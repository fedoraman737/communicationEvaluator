import os
import subprocess
from pathlib import Path

def download_model():
    """
    Downloads the Microsoft Phi-3-mini-4k-instruct model and saves it to the models directory.
    """
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Define model path
    model_path = models_dir / "phi3"
    
    # Check if model already exists
    if model_path.exists():
        print("Model already downloaded. Skipping download.")
        return
    
    print("Downloading Microsoft Phi-3-mini-4k-instruct model...")
    
    # Use huggingface-cli to download the model
    try:
        subprocess.run([
            "huggingface-cli", "download",
            "microsoft/Phi-3-mini-4k-instruct",
            "--local-dir", str(model_path),
            "--local-dir-use-symlinks", "False"
        ], check=True)
        print("Phi-3 model downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model: {e}")
        raise
    except FileNotFoundError:
        print("Error: huggingface-cli not found. Please install it first:")
        print("pip install huggingface_hub")

if __name__ == "__main__":
    download_model()