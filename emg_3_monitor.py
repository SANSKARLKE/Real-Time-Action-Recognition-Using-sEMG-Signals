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
BAUD_RATE   = 230400               # 3-channel 1000 Hz throughput
MODEL_PATH  = "emg_3ch_model.tflite"

FS          = 1000.0               # 1000 Hz sampling rate
WINDOW_SIZE = 200                  # 200 ms inference window
PLOT_SECONDS = 3.0                 # Display scope window
PLOT_SAMPLES = int(PLOT_SECONDS * FS)

INFERENCE_INTERVAL_SEC = 0.06      # ~16 FPS prediction cadence
RMS_NOISE_FLOOR = 35.0             # Noise gate to prevent idle drift classification

CLASS_NAMES = [
    "REST",
    "WRIST CLOSE",
    "WRIST OPEN",
    "WRIST UP",
    "WRIST DOWN"
]

CLASS_COLORS = {
    "REST": "#888888",
    "WRIST CLOSE": "#00FF66",
    "WRIST OPEN": "#FFB300",
    "WRIST UP": "#00E5FF",
    "WRIST DOWN": "#FF3366"
}

# -------------------------------------------------------------
# 1. Real-Time Digital Filter Setup (50 Hz Notch + 20 Hz HP)
# -------------------------------------------------------------
b_notch, a_notch = signal.iirnotch(w0=50.0, Q=30.0, fs=FS)
b_hp, a_hp       = signal.butter(N=2, Wn=20.0, btype='highpass', fs=FS)

# Initial filter states across all 3 channels
zi_n = [signal.lfilter_zi(b_notch, a_notch) * 2048.0 for _ in range(3)]
zi_h = [signal.lfilter_zi(b_hp, a_hp) * 0.0 for _ in range(3)]

def filter_3ch_sample(raw_triplet):
    """Causal, sample-by-sample notch and high-pass filtering across 3 channels."""
    global zi_n, zi_h
    filtered = []
    for ch in range(3):
        notched, zi_n[ch] = signal.lfilter(b_notch, a_notch, [raw_triplet[ch]], zi=zi_n[ch])
        hp_out, zi_h[ch]  = signal.lfilter(b_hp, a_hp, notched, zi=zi_h[ch])
        filtered.append(hp_out[0])
    return filtered

# -------------------------------------------------------------
# 2. Shared Buffers & TFLite Model Loading
# -------------------------------------------------------------
ch1_plot_buffer = collections.deque([0.0] * PLOT_SAMPLES, maxlen=PLOT_SAMPLES)
ch2_plot_buffer = collections.deque([0.0] * PLOT_SAMPLES, maxlen=PLOT_SAMPLES)
ch3_plot_buffer = collections.deque([0.0] * PLOT_SAMPLES, maxlen=PLOT_SAMPLES)
window_buffer   = collections.deque(maxlen=WINDOW_SIZE)

current_label      = "REST"
current_confidence = 0.0
running            = True

# Load the trained 3-channel TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Loaded model: {MODEL_PATH}")

# -------------------------------------------------------------
# 3. Background Thread: Serial Streaming & Inference Loop
# -------------------------------------------------------------
def serial_inference_worker():
    global current_label, current_confidence, running

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1.0)
        time.sleep(2.0)
        ser.reset_input_buffer()
        print(f"Serial link active on {SERIAL_PORT} @ {BAUD_RATE} baud. Live detection running...\n")
    except Exception as e:
        print(f"Failed to open port {SERIAL_PORT}: {e}")
        running = False
        return

    last_inference_time = 0.0

    while running:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue

        parts = line.split(',')
        if len(parts) != 3:
            continue

        try:
            raw_c1 = float(parts[0])
            raw_c2 = float(parts[1])
            raw_c3 = float(parts[2])
        except ValueError:
            continue

        # Real-time causal filtering
        f_c1, f_c2, f_c3 = filter_3ch_sample([raw_c1, raw_c2, raw_c3])

        ch1_plot_buffer.append(f_c1)
        ch2_plot_buffer.append(f_c2)
        ch3_plot_buffer.append(f_c3)
        window_buffer.append([f_c1, f_c2, f_c3])

        now = time.time()
        if len(window_buffer) == WINDOW_SIZE and (now - last_inference_time) >= INFERENCE_INTERVAL_SEC:
            last_inference_time = now

            win = np.array(window_buffer, dtype=np.float32)  # Shape: (200, 3)

            # Signal activity check (RMS) across all 3 channels
            rms_vals = [np.sqrt(np.mean(win[:, ch] ** 2)) for ch in range(3)]
            if all(rms < RMS_NOISE_FLOOR for rms in rms_vals):
                current_label = "REST"
                current_confidence = 100.0
                continue

            # Standardize channels per window identically to training
            norm_channels = []
            for ch in range(3):
                norm_c = (win[:, ch] - np.mean(win[:, ch])) / (np.std(win[:, ch]) + 1e-8)
                norm_channels.append(norm_c)

            norm_tensor = np.column_stack(norm_channels).reshape(1, WINDOW_SIZE, 3).astype(np.float32)

            # TFLite forward pass
            interpreter.set_tensor(input_details[0]['index'], norm_tensor)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]

            cls_idx = int(np.argmax(preds))
            current_confidence = float(preds[cls_idx] * 100.0)
            current_label = CLASS_NAMES[cls_idx]

    ser.close()

# -------------------------------------------------------------
# 4. Multi-Trace Live GUI
# -------------------------------------------------------------
data_thread = threading.Thread(target=serial_inference_worker, daemon=True)
data_thread.start()

plt.style.use('dark_background')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.canvas.manager.set_window_title("3-Channel EMG Real-Time Gesture Classifier")

x_time = np.linspace(-PLOT_SECONDS, 0, PLOT_SAMPLES)

# Subplot 1: Flexors (CH1)
line1, = ax1.plot(x_time, list(ch1_plot_buffer), color='#00E5FF', lw=1.1, label="CH1: Flexors (GPIO 34)")
ax1.set_ylabel("Amp", fontsize=9, color='gray')
ax1.set_ylim(-1500, 1500)
ax1.grid(True, color='#222222', linestyle='--')
ax1.legend(loc="upper right", framealpha=0.3, fontsize=8)

# Subplot 2: Extensors (CH2)
line2, = ax2.plot(x_time, list(ch2_plot_buffer), color='#FF9100', lw=1.1, label="CH2: Extensors (GPIO 35)")
ax2.set_ylabel("Amp", fontsize=9, color='gray')
ax2.set_ylim(-1500, 1500)
ax2.grid(True, color='#222222', linestyle='--')
ax2.legend(loc="upper right", framealpha=0.3, fontsize=8)

# Subplot 3: Lateral / Brachioradialis (CH3)
line3, = ax3.plot(x_time, list(ch3_plot_buffer), color='#D500F9', lw=1.1, label="CH3: Lateral/Brachioradialis (GPIO 32)")
ax3.set_ylabel("Amp", fontsize=9, color='gray')
ax3.set_xlabel("Time (seconds)", fontsize=9, color='gray')
ax3.set_ylim(-1500, 1500)
ax3.grid(True, color='#222222', linestyle='--')
ax3.legend(loc="upper right", framealpha=0.3, fontsize=8)

fig.suptitle("PREDICTION: INITIALIZING...", fontsize=14, fontweight='bold', color='white', y=0.98)

def update_plot(frame):
    y1 = np.array(ch1_plot_buffer)
    y2 = np.array(ch2_plot_buffer)
    y3 = np.array(ch3_plot_buffer)

    line1.set_ydata(y1)
    line2.set_ydata(y2)
    line3.set_ydata(y3)

    # Dynamic scaling per subplot
    for ax, y_arr in zip([ax1, ax2, ax3], [y1, y2, y3]):
        peak = max(float(np.max(np.abs(y_arr))), 200.0)
        ax.set_ylim(-peak * 1.25, peak * 1.25)

    color = CLASS_COLORS.get(current_label, "#FFFFFF")
    fig.suptitle(
        f"PREDICTION: {current_label} ({current_confidence:.1f}%)",
        fontsize=14,
        fontweight='bold',
        color=color,
        y=0.98
    )

    return line1, line2, line3

# 30 FPS display refresh
ani = animation.FuncAnimation(fig, update_plot, interval=33, blit=False)

try:
    plt.show()
finally:
    running = False
    data_thread.join(timeout=1.0)
    print("Real-time 3-channel session closed.")