import os
import time
import serial
import numpy as np
from scipy import signal

# -------------------------------------------------------------
# Configuration (Extended Acquisition Profile)
# -------------------------------------------------------------
SERIAL_PORT = "COM11"         # ESP32 CH9102 COM port
BAUD_RATE   = 230400          # Matches ESP32 3-channel stream rate
FS          = 1000.0          # 1000 Hz sampling rate

WINDOW_SIZE = 200             # 200 ms segment (200 samples)
STEP_SIZE   = 50              # 75% overlap (50 ms stride)
WARMUP_DROP = 200             # Discard initial filter settling samples

# Timing parameters
REPS_PER_CLASS = 12           # 12 repetitions per gesture (increased from 10)
HOLD_DURATION  = 8.0          # 8.0 seconds continuous hold per trial (increased from 5.0s)
REST_DURATION  = 4.0          # 4.0 seconds pause to prevent isometric muscle fatigue

ACTIONS = {
    0: "REST (Arm completely neutral/relaxed)",
    1: "WRIST CLOSE (Clench fist firmly)",
    2: "WRIST OPEN  (Splay fingers outward)",
    3: "WRIST UP    (Pull hand back / extend wrist)",
    4: "WRIST DOWN  (Curl hand down / flex wrist)"
}

OUTPUT_DIR = "./emg_3ch_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 50 Hz Notch + 20 Hz High-Pass Filter Design
# -------------------------------------------------------------
b_notch, a_notch = signal.iirnotch(w0=50.0, Q=30.0, fs=FS)
b_hp, a_hp       = signal.butter(N=2, Wn=20.0, btype='highpass', fs=FS)

def apply_3channel_filtering(burst_arr):
    """
    Applies zero-phase 50 Hz notch and 20 Hz high-pass filters across all 3 channels.
    burst_arr shape: (num_samples, 3)
    """
    filtered_channels = []
    for ch in range(3):
        raw_ch = burst_arr[:, ch]
        notched = signal.filtfilt(b_notch, a_notch, raw_ch)
        highpassed = signal.filtfilt(b_hp, a_hp, notched)
        filtered_channels.append(highpassed)

    return np.column_stack(filtered_channels)

# -------------------------------------------------------------
# Serial Ingestion Helper
# -------------------------------------------------------------
def record_burst(ser, duration_seconds):
    """Reads synchronous 3-channel samples for the given duration."""
    samples = []
    ser.reset_input_buffer()
    t_end = time.time() + duration_seconds

    while time.time() < t_end:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) != 3:
            continue
        try:
            c1 = float(parts[0])
            c2 = float(parts[1])
            c3 = float(parts[2])
            samples.append([c1, c2, c3])
        except ValueError:
            continue

    return np.array(samples, dtype=np.float32)

def extract_windows(filtered_burst, label):
    """
    Slices the filtered burst into overlapping windows and applies
    per-channel standard normal scaling within each window.
    """
    windows = []
    labels = []

    valid_data = filtered_burst[WARMUP_DROP:]
    num_samples = len(valid_data)

    for start in range(0, num_samples - WINDOW_SIZE, STEP_SIZE):
        win = valid_data[start : start + WINDOW_SIZE, :]

        # Normalize each channel independently: zero-mean, unit-variance
        norm_chans = []
        for ch in range(3):
            c = (win[:, ch] - np.mean(win[:, ch])) / (np.std(win[:, ch]) + 1e-8)
            norm_chans.append(c)

        norm_win = np.column_stack(norm_chans)
        windows.append(norm_win)
        labels.append(label)

    return windows, labels

# -------------------------------------------------------------
# Main Data Collection Routine
# -------------------------------------------------------------
print(f"Connecting to ESP32 on {SERIAL_PORT} at {BAUD_RATE} baud...")
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1.0)
time.sleep(2.0)
ser.reset_input_buffer()
print("Serial link established.")

all_windows = []
all_labels  = []

print("=" * 75)
print("EXTENDED 3-CHANNEL EMG DATASET ACQUISITION (5 GESTURES)")
print(f"Sampling       : {int(FS)} Hz | Hold: {HOLD_DURATION}s | Rest: {REST_DURATION}s")
print(f"Channels       : CH1 (GPIO 34), CH2 (GPIO 35), CH3 (GPIO 32)")
print(f"Window / Stride: {WINDOW_SIZE} samples / {STEP_SIZE} samples")
print(f"Total Trials   : {REPS_PER_CLASS * len(ACTIONS)} ({REPS_PER_CLASS} per class)")
print(f"Total Active Recording Time: ~{((HOLD_DURATION * REPS_PER_CLASS * len(ACTIONS)) / 60):.1f} minutes")
print("=" * 75)

for sec in range(3, 0, -1):
    print(f"Starting in {sec}...", end="\r")
    time.sleep(1.0)
print("\n")

try:
    for rep in range(1, REPS_PER_CLASS + 1):
        print(f"\n===================== TRIAL {rep}/{REPS_PER_CLASS} =====================")

        for class_id, class_name in ACTIONS.items():
            print(f"\nPrepare Action: >>> {class_name} <<<")
            time.sleep(1.5)

            print(f"--> HOLD GESTURE NOW ({HOLD_DURATION}s)! <--")
            raw_burst = record_burst(ser, HOLD_DURATION)

            if len(raw_burst) > (WARMUP_DROP + WINDOW_SIZE):
                filtered_burst = apply_3channel_filtering(raw_burst)
                windows, labels = extract_windows(filtered_burst, class_id)
                all_windows.extend(windows)
                all_labels.extend(labels)
                print(f"Captured {len(raw_burst)} samples -> Sliced {len(windows)} windows.")
            else:
                print(f"[WARNING] Incomplete stream ({len(raw_burst)} samples), skipping burst.")

            print(f"Relax arm completely... ({REST_DURATION}s)")
            time.sleep(REST_DURATION)

    # ---------------------------------------------------------
    # Save Dataset
    # ---------------------------------------------------------
    X = np.array(all_windows, dtype=np.float32)  # Shape: (Total_Windows, 200, 3)
    y = np.array(all_labels, dtype=np.int64)     # Shape: (Total_Windows,)

    x_path = os.path.join(OUTPUT_DIR, "X_3ch.npy")
    y_path = os.path.join(OUTPUT_DIR, "y_3ch.npy")
    np.save(x_path, X)
    np.save(y_path, y)

    print("\n" + "=" * 75)
    print("3-CHANNEL EXTENDED DATASET SAVED")
    print(f"Feature Array Shape (X) : {X.shape}  [Windows, 200 time steps, 3 channels]")
    print(f"Target Array Shape  (y) : {y.shape}")
    print("Class Window Distribution:")
    for cid, cname in ACTIONS.items():
        print(f"  [{cid}] {cname:<40}: {np.sum(y == cid)} windows")
    print(f"Saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 75)

except KeyboardInterrupt:
    print("\nCollection paused by user. Saving captured data...")
    if len(all_windows) > 0:
        X = np.array(all_windows, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int64)
        np.save(os.path.join(OUTPUT_DIR, "X_3ch.npy"), X)
        np.save(os.path.join(OUTPUT_DIR, "y_3ch.npy"), y)
        print(f"Saved partial dataset with {len(X)} windows.")

finally:
    ser.close()
    print("Serial port released.")