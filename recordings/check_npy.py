import numpy as np
import os

folder = r"E:\read_EMG\recordings"

files = {
    "raw": os.path.join(
        folder, "trial_005_20261019_174932_emg.npy"
    ),
    "inference": os.path.join(
        folder, "trial_005_20261019_174932_inference_689hz.npy"
    ),
}

fs = 689.0625

for name, path in files.items():
    x = np.load(path)

    print(f"\n--- {name} ---")
    print("Shape:", x.shape)
    print("Data type:", x.dtype)
    print("Duration (seconds):", x.shape[0] / fs)
    print("Min:", np.min(x))
    print("Max:", np.max(x))
    print("Mean:", np.mean(x))
    print("Std:", np.std(x))