import kagglehub
import shutil
import os

# Download dataset
dataset_path = kagglehub.dataset_download("ayuraj/asl-dataset")
print("Downloaded dataset to:", dataset_path)

# Source and destination paths
src_root = os.path.join(dataset_path, "asl_dataset")
dst_root = os.path.join(os.path.dirname(__file__), "images", "dataset")

# Folder mapping (0-9, a-z)
mapping = {
	"0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
	"a": "a", "b": "b", "c": "c", "d": "d", "e": "e", "f": "f", "g": "g", "h": "h", "i": "i", "j": "j",
	"k": "k", "l": "l", "m": "m", "n": "n", "o": "o", "p": "p", "q": "q", "r": "r", "s": "s", "t": "t",
	"u": "u", "v": "v", "w": "w", "x": "x", "y": "y", "z": "z"
}

# Create destination directory if it doesn't exist
os.makedirs(dst_root, exist_ok=True)

# Move and map folders
for src_folder, dst_folder in mapping.items():
	src_path = os.path.join(src_root, src_folder)
	dst_path = os.path.join(dst_root, dst_folder)
	if os.path.exists(src_path):
		shutil.move(src_path, dst_path)
		print(f"Moved {src_path} to {dst_path}")
	else:
		print(f"Source folder {src_path} does not exist.")
