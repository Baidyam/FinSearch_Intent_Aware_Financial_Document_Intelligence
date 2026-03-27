import gdown
import os

# Google Drive root folder ID
DRIVE_FOLDER_ID = "1YITrj2cPTqOxH7OAqirXb-svwZsuMH4E"
DATASET_DIR = "./Dataset"

def download_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)

    print("Downloading datasets from Google Drive...")
    print("This may take a while (~2GB). Please wait...\n")

    gdown.download_folder(
        id=DRIVE_FOLDER_ID,
        output=DATASET_DIR,
        quiet=False,
        use_cookies=False
    )

    print("\nAll datasets downloaded successfully!")
    print(f"Datasets are saved in: {os.path.abspath(DATASET_DIR)}/Data/")

if __name__ == "__main__":
    download_dataset()
