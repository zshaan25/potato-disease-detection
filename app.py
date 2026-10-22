import io
import os

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")  # config.json, preprocessor_config.json, model.safetensors
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp", "bmp"}
LOW_CONFIDENCE = 0.75

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
model = AutoModelForImageClassification.from_pretrained(MODEL_DIR).to(device).eval()
print(f"Model loaded on {device}. Classes: {model.config.id2label}")

# Display info keyed by the raw label names saved in config.json
DISEASE_INFO = {
    "Potato___Early_blight": {
        "name": "Early Blight",
        "severity": "moderate",
        "pathogen": "Alternaria solani",
        "summary": "A fungal disease that typically appears on older leaves first and spreads upward during warm, humid weather.",
        "symptoms": [
            "Dark brown spots with concentric rings (target-like pattern)",
            "Yellowing tissue around the lesions",
            "Lower, older leaves affected first; severe cases cause leaf drop",
        ],
        "treatment": [
            "Remove and destroy heavily infected leaves",
            "Apply a protectant fungicide (chlorothalonil or mancozeb) as per label",
            "Water at the base, not overhead, and avoid wet foliage at night",
            "Rotate crops; avoid planting potatoes or tomatoes in the same plot consecutively",
        ],
    },
    "Potato___Late_blight": {
        "name": "Late Blight",
        "severity": "severe",
        "pathogen": "Phytophthora infestans",
        "summary": "A fast-spreading and destructive disease that can destroy a crop within days in cool, wet conditions.",
        "symptoms": [
            "Water-soaked, pale green to dark brown patches on leaves",
            "White fuzzy growth on the underside of leaves in humid conditions",
            "Rapid blackening of leaves and stems; tuber rot",
        ],
        "treatment": [
            "Act immediately: isolate and remove infected plants",
            "Apply a systemic fungicide (e.g. metalaxyl or cymoxanil based) per label and local advice",
            "Do not compost infected material; destroy it",
            "Use certified disease-free seed tubers and resistant varieties next season",
        ],
    },
    "Potato___healthy": {
        "name": "Healthy Leaf",
        "severity": "none",
        "pathogen": "None detected",
        "summary": "No signs of early or late blight were detected in this leaf.",
        "symptoms": [
            "Uniform green colour",
            "No lesions, spots or discoloration",
        ],
        "treatment": [
            "Keep up regular scouting, at least weekly",
            "Maintain balanced fertilization and proper spacing for airflow",
            "Avoid overhead watering late in the day",
        ],
    },
}


def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(status="ok", device=str(device), classes=list(model.config.id2label.values()))


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="No image was uploaded."), 400
    if not allowed(file.filename):
        return jsonify(error="Unsupported file type. Use JPG, PNG or WEBP."), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify(error="The file could not be read as an image."), 400

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

    id2label = model.config.id2label
    ranked = sorted(
        ({"label": id2label[i], "name": DISEASE_INFO.get(id2label[i], {}).get("name", id2label[i]), "probability": p}
         for i, p in enumerate(probs)),
        key=lambda x: x["probability"],
        reverse=True,
    )
    top = ranked[0]
    info = DISEASE_INFO.get(top["label"], {})

    return jsonify(
        label=top["label"],
        name=top["name"],
        confidence=top["probability"],
        low_confidence=top["probability"] < LOW_CONFIDENCE,
        info=info,
        probabilities=ranked,
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image is too large. Maximum size is 8 MB."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
