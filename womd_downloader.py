from datasets import load_dataset
import os

# Where to save everything
save_dir = "/scratch/ag11023/PufferDrive/pufferlib/data/processed/training"

# Force Hugging Face cache to stay in /scratch
os.environ["HF_HOME"] = save_dir
os.environ["HF_DATASETS_CACHE"] = save_dir
os.environ["HF_HUB_CACHE"] = save_dir

# Optional: force authentication if needed
# os.environ["HF_TOKEN"] = "hf_yourtokenhere"

print(f"Saving dataset to {save_dir}")
dataset = load_dataset("EMERGE-lab/GPUDrive_mini", cache_dir=save_dir)
dataset.save_to_disk(os.path.join(save_dir, "GPUDrive_mini"))
print("Done! Dataset downloaded and stored successfully.")
