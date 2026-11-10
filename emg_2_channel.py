import os
import time
import serial
import numpy as np
from scipy import signal

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
SERIAL_PORT = "COM11"         # ESP32 CH9102 COM port
BAUD_RATE   = 115200
FS          = 1000.0          # 1000 Hz sampling rate

WINDOW_SIZE = 200             # 200 ms window (200 samples per channel)
STEP_SIZE   = 50              # 75% overlap (50 ms stride)
WARMUP_DROP = 200             # Discard first 200 samples of each burst (filter settling)

# Doubled active acquisition duration
REPS_PER_CLASS = 12           # 12 trials per gesture
HOLD_DURATION  = 6.0          # Doubled from 3.0s to 6.0s active hold per trial
REST_DURATION  = 3.0          # 3.0s recovery pause between trials to prevent muscle fatigue

ACTIONS = {
    0: "REST (Arm completely neutral/relaxed)",
    1: "WRIST CLOSE (Fist / Flexor contraction)",
    2: "WRIST OPEN  (Hand open / Extensor contraction)"
}

OUTPUT_DIR = "./emg_2ch_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# Filter Design (50 Hz Notch + 20 Hz High-Pass)
# -------------------------------------------------------------
b_notch, a_notch = signal.iirnotch(w0=50.0, Q=30.0, fs=FS)
b_hp, a_hp       = signal.butter(N=2, Wn=20.0, btype='highpass', fs=FS)

def apply_dual_channel_filtering(burst_arr):
    """
    Applies zero-phase 50 Hz notch and 20 Hz high-pass filters along each channel.
    burst_arr shape: (num_samples, 2)
    """
    ch1_raw = burst_arr[:, 0]
    ch2_raw = burst_arr[:, 1]

    # Step 1: 50 Hz Notch Filter
    ch1_notched = signal.filtfilt(b_notch, a_notch, ch1_raw)
    ch2_notched = signal.filtfilt(b_notch, a_notch, ch2_raw)

    # Step 2: 20 Hz High-Pass Filter (removes DC baseline drift)
    ch1_filtered = signal.filtfilt(b_hp, a_hp, ch1_notched)
    ch2_filtered = signal.filtfilt(b_hp, a_hp, ch2_notched)

    return np.column_stack((ch1_filtered, ch2_filtered))

# -------------------------------------------------------------
# Serial Acquisition Helper
# -------------------------------------------------------------
def record_burst(ser, duration_seconds):
    """Captures synchronous dual-channel readings for duration_seconds."""
    samples = []
    ser.reset_input_buffer()
    t_end = time.time() + duration_seconds

    while time.time() < t_end:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line or ',' not in line:
            continue
        try:
            parts = line.split(',')
            c1 = float(parts[0])
            c2 = float(parts[1])
            samples.append([c1, c2])
        except (ValueError, IndexError):
            continue

    return np.array(samples, dtype=np.float32)

def extract_windows(filtered_burst, label):
    """
    Slices the filtered burst into overlapping 200 ms windows.
    Discards the initial warmup edge and applies per-window z-score scaling.
    """
    windows = []
    labels = []

    valid_data = filtered_burst[WARMUP_DROP:]
    num_samples = len(valid_data)

    for start in range(0, num_samples - WINDOW_SIZE, STEP_SIZE):
        win = valid_data[start : start + WINDOW_SIZE, :]  # Shape: (200, 2)

        # Standardize each channel independently within the window
        ch1 = (win[:, 0] - np.mean(win[:, 0])) / (np.std(win[:, 0]) + 1e-8)
        ch2 = (win[:, 1] - np.mean(win[:, 1])) / (np.std(win[:, 1]) + 1e-8)
        norm_win = np.column_stack((ch1, ch2))

        windows.append(norm_win)
        labels.append(label)

    return windows, labels

# -------------------------------------------------------------
# Main Data Collection Loop
# -------------------------------------------------------------
print(f"Connecting to ESP32 on {SERIAL_PORT}...")
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1.0)
time.sleep(2.0)
ser.reset_input_buffer()
print("Serial link established.")

all_windows = []
all_labels  = []

print("=" * 70)
print("DOUBLED-DURATION 2-CHANNEL EMG DATA COLLECTION")
print(f"Sampling       : {int(FS)} Hz")
print(f"Hold Duration  : {HOLD_DURATION}s per repetition (Doubled)")
print(f"Window / Stride: {WINDOW_SIZE} samples (200 ms) / {STEP_SIZE} samples (50 ms)")
print(f"Total Trials   : {REPS_PER_CLASS * len(ACTIONS)} ({REPS_PER_CLASS} per action)")
print("=" * 70)

for sec in range(3, 0, -1):
    print(f"Starting in {sec}...", end="\r")
    time.sleep(1.0)
print("\n")

try:
    for rep in range(1, REPS_PER_CLASS + 1):
        print(f"\n===================== TRIAL {rep}/{REPS_PER_CLASS} =====================")
        
        for class_id, class_name in ACTIONS.items():
            print(f"\nPrepare Action: >>> {class_name} <<<")
            time.sleep(1.0)

            print(f"--> HOLD GESTURE NOW ({HOLD_DURATION}s)! <--")
            raw_burst = record_burst(ser, HOLD_DURATION)

            if len(raw_burst) > (WARMUP_DROP + WINDOW_SIZE):
                filtered_burst = apply_dual_channel_filtering(raw_burst)
                windows, labels = extract_windows(filtered_burst, class_id)
                all_windows.extend(windows)
                all_labels.extend(labels)
                print(f"Captured {len(raw_burst)} samples -> Sliced {len(windows)} windows.")
            else:
                print("[WARNING] Incomplete packet stream, skipping burst.")

            print(f"Relax arm completely... ({REST_DURATION}s)")
            time.sleep(REST_DURATION)

    # ---------------------------------------------------------
    # Save Formatted Dataset
    # ---------------------------------------------------------
    X = np.array(all_windows, dtype=np.float32)  # Shape: (Total_Windows, 200, 2)
    y = np.array(all_labels, dtype=np.int64)     # Shape: (Total_Windows,)

    x_path = os.path.join(OUTPUT_DIR, "X_2ch.npy")
    y_path = os.path.join(OUTPUT_DIR, "y_2ch.npy")
    np.save(x_path, X)
    np.save(y_path, y)

    print("\n" + "=" * 70)
    print("DATASET COLLECTION COMPLETE")
    print(f"Feature Tensor Shape (X) : {X.shape}  [Windows, 200 samples, 2 channels]")
    print(f"Label Array Shape    (y) : {y.shape}")
    print(f"Class Distribution       : Rest(0)={np.sum(y == 0)} | Close(1)={np.sum(y == 1)} | Open(2)={np.sum(y == 2)}")
    print(f"Saved Files              :")
    print(f"  - {os.path.abspath(x_path)}")
    print(f"  - {os.path.abspath(y_path)}")
    print("=" * 70)

except KeyboardInterrupt:
    print("\nData collection interrupted by user. Saving captured data...")
    if len(all_windows) > 0:
        X = np.array(all_windows, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int64)
        np.save(os.path.join(OUTPUT_DIR, "X_2ch.npy"), X)
        np.save(os.path.join(OUTPUT_DIR, "y_2ch.npy"), y)
        print(f"Saved partial dataset with {len(X)} windows.")

finally:
    ser.close()
    print("Serial port released cleanly.")