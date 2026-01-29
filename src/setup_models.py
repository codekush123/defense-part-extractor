import os
import urllib.request

def download_model():
    model_dir = "models"
    model_path = os.path.join(model_dir, "sam_vit_b_01ec64.pth")
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(model_path):
        print("Downloading SAM weights (375MB)... this may take a minute.")
        urllib.request.urlretrieve(url, model_path)
        print("Download complete!")
    else:
        print("Model already exists in /models.")

if __name__ == "__main__":
    download_model()