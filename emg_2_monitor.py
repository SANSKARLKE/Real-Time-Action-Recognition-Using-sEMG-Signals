import time
import threading
import collections
import numpy as np
import serial
from scipy import signal
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
SERIAL_PORT = "COM11"              # ESP32 CH9102 COM port
BAUD_RATE   = 115200
MODEL_PATH  = "emg_2ch_model.tflite"

FS          = 1000.0               # 1000 Hz sampling rate
WINDOW_SIZE = 200                  # 200 ms inference window
PLOT_SECONDS = 3.0                 # 3-second display scope
PLOT_SAMPLES = int(PLOT_SECONDS * FS)

INFERENCE_INTERVAL_SEC = 0.06      # ~16 FPS prediction cadence

# RMS baseline gate: if raw signals are quiet, force REST to avoid noise classification
RMS_NOISE_FLOOR = 35.0

CLASS_NAMES = ["REST", "WRIST CLOSE", "WRIST OPEN"]
CLASS_COLORS = {
    "REST": "#888888",
    "WRIST CLOSE": "#00FF66",
    "WRIST OPEN": "#FFB300"
}

# -------------------------------------------------------------
# 1. Real-Time Digital Filter Setup (50 Hz Notch + 20 Hz HP)
# -------------------------------------------------------------
b_notch, a_notch = signal.iirnotch(w0=50.0, Q=30.0, fs=FS)
b_hp, a_hp       = signal.butter(N=2, Wn=20.0, btype='highpass', fs=FS)

zi_n_ch1 = signal.lfilter_zi(b_notch, a_notch) * 2048.0
zi_n_ch2 = signal.lfilter_zi(b_notch, a_notch) * 2048.0
zi_h_ch1 = signal.lfilter_zi(b_hp, a_hp) * 0.0
zi_h_ch2 = signal.lfilter_zi(b_hp, a_hp) * 0.0

def filter_sample(c1_raw, c2_raw):
    """Processes a streaming dual-channel sample pair through notch and HP filters."""
    global zi_n_ch1, zi_n_ch2, zi_h_ch1, zi_h_ch2

    # Step 1: 50 Hz Notch Filter
    c1_n, zi_n_ch1 = signal.lfilter(b_notch, a_notch, [c1_raw], zi=zi_n_ch1)
    c2_n, zi_n_ch2 = signal.lfilter(b_notch, a_notch, [c2_raw], zi=zi_n_ch2)

    # Step 2: 20 Hz High-Pass Filter
    c1_f, zi_h_ch1 = signal.lfilter(b_hp, a_hp, c1_n, zi=zi_h_ch1)
    c2_f, zi_h_ch2 = signal.lfilter(b_hp, a_hp, c2_n, zi=zi_h_ch2)

    return c1_f[0], c2_f[0]

# -------------------------------------------------------------
# 2. Shared Buffers & Model Loading
# -------------------------------------------------------------
ch1_plot_buffer = collections.deque([0.0] * PLOT_SAMPLES, maxlen=PLOT_SAMPLES)
ch2_plot_buffer = collections.deque([0.0] * PLOT_SAMPLES, maxlen=PLOT_SAMPLES)
window_buffer   = collections.deque(maxlen=WINDOW_SIZE)

current_label      = "REST"
current_confidence = 0.0
running            = True

# Load the newly trained 80-epoch TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Loaded TFLite model: {MODEL_PATH}")

# -------------------------------------------------------------
# 3. Background Thread: Serial Streaming & Inference Loop
# -------------------------------------------------------------
def serial_inference_worker():
    global current_label, current_confidence, running

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1.0)
        time.sleep(2.0)
        ser.reset_input_buffer()
        print(f"Serial link active on {SERIAL_PORT}. Streaming & detecting...\n")
    except Exception as e:
        print(f"Failed to open port {SERIAL_PORT}: {e}")
        running = False
        return

    last_inference_time = 0.0

    while running:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line or ',' not in line:
            continue

        try:
            parts = line.split(',')
            raw_c1 = float(parts[0])
            raw_c2 = float(parts[1])
        except (ValueError, IndexError):
            continue

        # Causal digital filtering
        f_c1, f_c2 = filter_sample(raw_c1, raw_c2)

        ch1_plot_buffer.append(f_c1)
        ch2_plot_buffer.append(f_c2)
        window_buffer.append([f_c1, f_c2])

        now = time.time()
        if len(window_buffer) == WINDOW_SIZE and (now - last_inference_time) >= INFERENCE_INTERVAL_SEC:
            last_inference_time = now

            win = np.array(window_buffer, dtype=np.float32)  # Shape: (200, 2)

            # Check signal energy (RMS)
            rms_c1 = np.sqrt(np.mean(win[:, 0] ** 2))
            rms_c2 = np.sqrt(np.mean(win[:, 1] ** 2))

            # Hardware noise guard
            if rms_c1 < RMS_NOISE_FLOOR and rms_c2 < RMS_NOISE_FLOOR:
                current_label = "REST"
                current_confidence = 100.0
                continue

            # Standardize channels per window exactly as done during training
            ch1_norm = (win[:, 0] - np.mean(win[:, 0])) / (np.std(win[:, 0]) + 1e-8)
            ch2_norm = (win[:, 1] - np.mean(win[:, 1])) / (np.std(win[:, 1]) + 1e-8)
            norm_tensor = np.column_stack((ch1_norm, ch2_norm)).reshape(1, WINDOW_SIZE, 2).astype(np.float32)

            # Execute forward pass
            interpreter.set_tensor(input_details[0]['index'], norm_tensor)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]

            cls_idx = int(np.argmax(preds))
            current_confidence = float(preds[cls_idx] * 100.0)
            current_label = CLASS_NAMES[cls_idx]

    ser.close()

# -------------------------------------------------------------
# 4. Oscilloscope GUI
# -------------------------------------------------------------
data_thread = threading.Thread(target=serial_inference_worker, daemon=True)
data_thread.start()

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig.canvas.manager.set_window_title("2-Channel Live EMG Classifier")

x_time = np.linspace(-PLOT_SECONDS, 0, PLOT_SAMPLES)

# Subplot 1: Flexors (CH1)
line1, = ax1.plot(x_time, list(ch1_plot_buffer), color='#00E5FF', lw=1.2, label="CH1: Flexors (GPIO 34)")
ax1.set_ylabel("Amplitude", fontsize=10, color='gray')
ax1.set_ylim(-1500, 1500)
ax1.grid(True, color='#222222', linestyle='--')
ax1.legend(loc="upper right", framealpha=0.3)

# Subplot 2: Extensors (CH2)
line2, = ax2.plot(x_time, list(ch2_plot_buffer), color='#FF9100', lw=1.2, label="CH2: Extensors (GPIO 35)")
ax2.set_ylabel("Amplitude", fontsize=10, color='gray')
ax2.set_xlabel("Time (seconds)", fontsize=10, color='gray')
ax2.set_ylim(-1500, 1500)
ax2.grid(True, color='#222222', linestyle='--')
ax2.legend(loc="upper right", framealpha=0.3)

fig.suptitle("PREDICTION: INITIALIZING...", fontsize=14, fontweight='bold', color='white', y=0.98)

def update_plot(frame):
    y1 = np.array(ch1_plot_buffer)
    y2 = np.array(ch2_plot_buffer)

    line1.set_ydata(y1)
    line2.set_ydata(y2)

    # Dynamic trace scaling
    peak1 = max(float(np.max(np.abs(y1))), 200.0)
    peak2 = max(float(np.max(np.abs(y2))), 200.0)
    ax1.set_ylim(-peak1 * 1.25, peak1 * 1.25)
    ax2.set_ylim(-peak2 * 1.25, peak2 * 1.25)

    color = CLASS_COLORS.get(current_label, "#FFFFFF")
    fig.suptitle(
        f"PREDICTION: {current_label} ({current_confidence:.1f}%)",
        fontsize=14,
        fontweight='bold',
        color=color,
        y=0.98
    )

    return line1, line2

# 30 FPS visual refresh rate
ani = animation.FuncAnimation(fig, update_plot, interval=33, blit=False)

try:
    plt.show()
finally:
    running = False
    data_thread.join(timeout=1.0)
    print("Real-time inference session closed.")