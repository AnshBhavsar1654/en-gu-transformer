# upload_model.py
import os

from huggingface_hub import HfApi, login

# best.pt from the Kaggle notebook output (self-contained: config + weights + tokenizer)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")

# Login to Hugging Face (you'll be prompted for your token)
login()

# Initialize the API
api = HfApi()

# Create repository (replace with your username and desired repo name)
repo_id = "anshbhavsar2674/english-gujarati-transformer"

try:
    # Create the repository
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        private=False  # Set to True if you want it private
    )
    print(f"Repository created: https://huggingface.co/{repo_id}")

    # Upload your model file
    api.upload_file(
        path_or_fileobj=MODEL_PATH,
        path_in_repo="best.pt",     # Name in the repo (app.py downloads this name)
        repo_id=repo_id,
        repo_type="model"
    )
    print("Model uploaded successfully!")

except Exception as e:
    print(f"Error: {e}")
