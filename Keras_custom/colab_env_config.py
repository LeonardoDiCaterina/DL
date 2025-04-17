import os
import subprocess
import requests
from tqdm import tqdm

# --- Configuration ---
DOWNLOAD_URL = "https://drive.google.com/file/d/1PyxqW_nsORX4PetkQo6OIL0mUL1pFsTD/view?usp=sharing"
OUTPUT_ZIP_FILENAME = "rare_species.zip"
EXTRACTED_FOLDER = "rare_species_data"
YAML_FILENAME = "DL.yaml"  # Assuming your YAML file is named this
CONDA_ENV_NAME = "DL"  # The name you want for your Conda environment

def download_file_from_google_drive(url, destination):
    """Downloads a file from Google Drive using its file ID."""
    try:
        file_id = url.split("/d/")[1].split("/")[0]
        download_url = f"https://drive.google.com/uc?id={file_id}&export=download"

        response = requests.get(download_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=block_size):
                progress_bar.update(len(chunk))
                f.write(chunk)
        progress_bar.close()
        print(f"Downloaded '{destination}' successfully.")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def unzip_file(zip_path, extract_path):
    """Unzips a file to the specified directory."""
    try:
        os.makedirs(extract_path, exist_ok=True)
        subprocess.run(["unzip", zip_path, "-d", extract_path], check=True)
        print(f"Unzipped '{zip_path}' to '{extract_path}' successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error unzipping file: {e}")
        return False
    except FileNotFoundError:
        print("Error: 'unzip' command not found. Make sure it's installed.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during unzipping: {e}")
        return False

def install_conda():
    """Installs Conda in the Colab environment."""
    try:
        print("Installing Conda...")
        subprocess.run(["pip", "install", "-q", "condacolab"], check=True)
        import condacolab
        condacolab.install()
        print("Conda installed successfully.")
        return True
    except Exception as e:
        print(f"Error installing Conda: {e}")
        return False

def create_conda_environment(yaml_path, env_name):
    """Creates a Conda environment from a YAML file."""
    try:
        print(f"Creating Conda environment '{env_name}' from '{yaml_path}'...")
        subprocess.run(["conda", "env", "create", "-f", yaml_path, "--name", env_name], check=True)
        print(f"Conda environment '{env_name}' created successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating Conda environment: {e}")
        return False
    except FileNotFoundError:
        print("Error: 'conda' command not found. Make sure Conda is installed.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during environment creation: {e}")
        return False

def main():
    # 1. Install Conda
    if not install_conda():
        return

    # 2. Download the ZIP file
    if not download_file_from_google_drive(DOWNLOAD_URL, OUTPUT_ZIP_FILENAME):
        return

    # 3. Create the extraction directory
    os.makedirs(EXTRACTED_FOLDER, exist_ok=True)

    # 4. Unzip the file
    if not unzip_file(OUTPUT_ZIP_FILENAME, EXTRACTED_FOLDER):
        return

    # 5. Check if the YAML file exists in the extracted folder (adjust path if needed)
    yaml_file_path = os.path.join(EXTRACTED_FOLDER, YAML_FILENAME)
    if not os.path.exists(yaml_file_path):
        print(f"Error: YAML file '{yaml_file_path}' not found in the extracted directory.")
        print("Please ensure your ZIP file contains the environment.yaml file.")
        return

    # 6. Create the Conda environment from the YAML file
    if not create_conda_environment(yaml_file_path, CONDA_ENV_NAME):
        return

    print("All steps completed successfully!")
    print(f"The ZIP file has been downloaded and unzipped to '{EXTRACTED_FOLDER}'.")
    print(f"The Conda environment '{CONDA_ENV_NAME}' has been created based on '{yaml_file_path}'.")
    print("You can now activate this environment to run your project.")
    print("To activate the environment in Colab, you might need to restart the runtime or use Colab-specific methods.")

if __name__ == "__main__":
    main()