import os
import zipfile
import gdown

# REPLACE THIS WITH YOUR ACTUAL GOOGLE DRIVE FILE ID
# ⚠️ IMPORTANT: PASTE YOUR REAL ID BELOW OR IT WILL FAIL ⚠️
# Example: '1A2B3C4D5E6F7G8H9I0J'
file_id = "17YMBYdqVsIy9tIhjMGpVuhtyrPTGVZXN"
output = "models.zip"

def download_and_extract():
    # 1. Check if models already exist so we don't download twice
    if os.path.exists('models') and len(os.listdir('models')) > 0:
        print("Models folder already exists. Skipping download.")
        return

    # Safety check for the example ID
    if file_id == "17YMBYdqVsIy9tIhjMGpVuhtyrPTGVZXN" or "PASTE" in file_id:
        print("⚠️ WARNING: You are using the example ID. Skipping download to allow app startup.")
        return

    # 2. Download the zip file
    print("Downloading models from Google Drive...")
    gdown.download(id=file_id, output=output, quiet=False)

    # Check if the download was actually a zip file
    if not zipfile.is_zipfile(output):
        print("ERROR: The downloaded file is not a valid zip. Check your Google Drive Link permissions!")
        raise ValueError("Download failed - Invalid Zip")

    # 3. Unzip it
    print("Unzipping models...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        # Check if the zip already contains a 'models' folder
        file_names = zip_ref.namelist()
        has_models_folder = any(name.startswith('models/') or name.startswith('models\\') for name in file_names)
        
        if has_models_folder:
            zip_ref.extractall('.') # Extracts to current folder
        else:
            # If zip is flat, extract into a models/ directory
            os.makedirs('models', exist_ok=True)
            zip_ref.extractall('models')
            
    # 4. Fix Case Sensitivity (Windows vs Linux)
    # Rename all files in models/ to lowercase so the code can find them
    print("Normalizing filenames to lowercase...")
    for filename in os.listdir('models'):
        old_path = os.path.join('models', filename)
        new_path = os.path.join('models', filename.lower())
        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {filename.lower()}")

    # 5. Clean up (delete the zip file to save space)
    if os.path.exists(output):
        os.remove(output)
    
    print("Done! Models are ready.")

if __name__ == "__main__":
    download_and_extract()
