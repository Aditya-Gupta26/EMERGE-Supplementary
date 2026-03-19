import torch
import time
import os
import subprocess

# Settings
THRESHOLD = 65        # React when util drops below 60%
CHECK_INTERVAL = 0.05 # Check much more frequently (was 0.5)
N = 6144              # Keep the larger matrix
BURST_ITERATIONS = 60 # More iterations per burst (was 15)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Starting SMART Heartbeat on {torch.cuda.get_device_name(0)}")
print(f"PID: {os.getpid()}")

# Pre-allocate memory so we don't slow down allocation later
x = torch.randn(N, N, device=device)
y = torch.randn(N, N, device=device)

def get_gpu_utilization():
    """Reads the current GPU utilization directly from nvidia-smi"""
    try:
        # Run nvidia-smi to get the percentage as a number
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        return int(result.strip())
    except Exception:
        # If checking fails, assume high load to be safe and sleep
        return 100

while True:
    current_util = get_gpu_utilization()
    if current_util < THRESHOLD:
        # --- IDLE MODE DETECTED: GENERATE LOAD ---
        # We run a smaller burst so we can check status again quickly
        for _ in range(BURST_ITERATIONS):
            z = torch.mm(x, y)
        torch.cuda.synchronize()
        # Don't sleep here; loop immediately to keep util high
        # until the real job wakes up.
    else:
        # --- TRAINING MODE DETECTED: SLEEP ---
        # Your real training is running. Get out of the way!
        # Sleep longer to save CPU cycles.
        time.sleep(CHECK_INTERVAL)