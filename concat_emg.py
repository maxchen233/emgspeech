import numpy as np
import os

# 使用者要求：將 22_emg.npy (後面這句) 放前面，194_emg.npy (前面那句) 放後面
file1 = 'emg_data/silent_parallel_data/5-10_silent/22_emg.npy' # 後面這句: "The hot water..."
file2 = 'emg_data/voiced_parallel_data/5-6/194_emg.npy'      # 前面這句: "After breakfast..."

print("Loading part 1:", file1)
arr1 = np.load(file1)
print("Loading part 2:", file2)
arr2 = np.load(file2)


combined = np.concatenate([arr1[:-800], arr2[400:]], axis=0)

out_path = 'emg_data/combined_emg.npy'
np.save(out_path, combined)

print(f"第一段 shape: {arr1.shape}")
print(f"第二段 shape: {arr2.shape}")
print(f"合併後 shape: {combined.shape}")
print(f"✅ 成功輸出合併檔案至: {out_path}")
