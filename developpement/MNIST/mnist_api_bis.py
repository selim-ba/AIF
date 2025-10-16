import argparse
import io
import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from model import MNISTNet

# ---------------------------------------------------------------------
# Configuration du device
# ---------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

# ---------------------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='weights/mnist_net.pth', help='Chemin vers le modèle')
args = parser.parse_args()
model_path = args.model_path

model = MNISTNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---------------------------------------------------------------------
# Préprocessing (cohérent avec MNIST)
# ---------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # stats officielles MNIST
])

# ---------------------------------------------------------------------
# Route /predict  → 1 image
# ---------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1️⃣ Si l'image vient d'un formulaire multipart (files)
        if 'file' in request.files:
            file_storage = request.files['file']
            img_pil = Image.open(file_storage.stream).convert('L')

        # 2️⃣ Sinon, cas d'octets bruts (raw bytes)
        else:
            img_binary = request.get_data()
            if not img_binary:
                return jsonify(error="Aucune image reçue"), 400
            img_pil = Image.open(io.BytesIO(img_binary)).convert('L')

        # Transformation et prédiction
        tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = outputs.max(1)

        return jsonify({"prediction": int(predicted.item())})

    except UnidentifiedImageError:
        return jsonify(error="Fichier non reconnu comme image"), 400
    except Exception as e:
        return jsonify(error=str(e)), 500

# ---------------------------------------------------------------------
# Route /batch_predict  → plusieurs images
# ---------------------------------------------------------------------
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        images_binary = request.files.getlist("images[]")
        if not images_binary:
            return jsonify(error="Aucune image dans la requête"), 400

        tensors = []
        for img_file in images_binary:
            img_pil = Image.open(img_file.stream).convert('L')
            tensors.append(transform(img_pil))

        batch_tensor = torch.stack(tensors, dim=0).to(device)

        with torch.no_grad():
            outputs = model(batch_tensor)
            _, predictions = outputs.max(1)

        return jsonify({"predictions": predictions.tolist()})

    except UnidentifiedImageError:
        return jsonify(error="Une ou plusieurs images invalides"), 400
    except Exception as e:
        return jsonify(error=str(e)), 500

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print(f"✅ Modèle chargé depuis : {model_path}")
    print("🚀 API en cours d'exécution sur http://127.0.0.1:5075/predict")
    app.run(host='0.0.0.0', port=5075, debug=True)
