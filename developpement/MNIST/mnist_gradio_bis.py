import io
import numpy as np
import requests
from PIL import Image, ImageOps
import gradio as gr

API_URL = "http://127.0.0.1:5075/predict"  # <- localhost, pas 0.0.0.0

def preprocess_from_sketchpad(img):
    if isinstance(img, dict) and 'composite' in img:
        img = img['composite']

    # img: (H, W, C?) numpy
    used_alpha = False
    if img.ndim == 3 and img.shape[2] >= 4:
        img = img[:, :, 3]           # alpha: trait=255, fond=0
        used_alpha = True
    elif img.ndim == 3:
        img = img.mean(axis=2)       # moyenne RGB
    # sinon déjà 2D

    img = np.clip(img, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img, mode="L").resize((28, 28), Image.NEAREST)

    # ⬇️ Inverser seulement si on n'a PAS utilisé l'alpha (cas RGB)
    if not used_alpha:
        pil_img = ImageOps.invert(pil_img)

    return pil_img


def recognize_digit(image):
    pil_img = preprocess_from_sketchpad(image)

    # Encoder en PNG dans un buffer
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    # VERSION la plus courante: l'API attend un fichier multipart "file"
    files = {"file": ("digit.png", buf.getvalue(), "image/png")}
    try:
        resp = requests.post(API_URL, files=files, timeout=5)
    except requests.RequestException as e:
        return f"API error: {e}"

    if resp.status_code != 200:
        return f"Bad response ({resp.status_code}): {resp.text}"

    # On suppose que l'API renvoie {"prediction": int}
    try:
        return resp.json()["prediction"]
    except Exception:
        return f"Unexpected JSON: {resp.text}"

if __name__ == "__main__":
    interface = gr.Interface(
        fn=recognize_digit,
        inputs=gr.Sketchpad(),   # explicite
        outputs=gr.Label(num_top_classes=1),
        live=False,
        description="Draw a number on the sketchpad to see the model's prediction.",
    )
    print("Starting Gradio app...")
    interface.launch(debug=True, share=True, server_name="0.0.0.0", server_port=7870)
