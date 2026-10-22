# SpudScan: Potato Leaf Disease Detection

A Flask web application that classifies potato leaf images as **Early Blight**, **Late Blight**, or **Healthy** using a fine-tuned Vision Transformer (ViT). Upload a leaf photo and get a diagnosis with confidence scores, symptoms, and treatment guidance.

## Features

- Drag-and-drop image upload with live preview
- Instant prediction with confidence score and per-class probability bars
- Disease details: severity, causal agent, symptoms, and treatment advice
- Low-confidence warning for unclear or non-leaf images
- Recent scan history (stored locally in the browser)
- Printable report, responsive layout, automatic dark mode
- JSON API and a `/health` endpoint

## Model

| Item | Detail |
|---|---|
| Base model | `google/vit-base-patch16-224` |
| Task | 3-class image classification |
| Classes | Early Blight, Late Blight, Healthy |
| Dataset | [Potato Dataset (Kaggle)](https://www.kaggle.com/datasets/faysalmiah1721758/potato-dataset) |
| Split | 80% train (1,721 images) / 20% test (431 images), `random_state=42` |
| Training | 3 epochs, AdamW, lr 2e-5, batch size 8, cross-entropy loss |
| Test accuracy | 100% on the held-out split |

**Note on the results:** the test set is small (the healthy class has only 31 images) and the dataset consists of clean, lab-style leaf photos. Accuracy on real field images with cluttered backgrounds will likely be lower. The model also only knows these three classes, so a non-potato image will still be assigned one of them.

## Project Structure

```
potato_app/
├── app.py                 # Flask server and prediction endpoint
├── requirements.txt
├── model/
│   ├── config.json
│   ├── preprocessor_config.json
│   └── model.safetensors  # stored with Git LFS
└── templates/
    └── index.html         # Frontend interface
```

## Getting Started

Requires Python 3.9+ and [Git LFS](https://git-lfs.com) (the model file is about 340 MB).

```bash
git lfs install
git clone https://github.com/zshaan25/potato-disease-detection.git
cd potato-disease-detection

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 in your browser. The app runs on CPU; a GPU is used automatically if available.

## API

`POST /predict` with a multipart form field named `image`.

```bash
curl -X POST -F "image=@leaf.jpg" http://127.0.0.1:5000/predict
```

Example response:

```json
{
  "label": "Potato___Late_blight",
  "name": "Late Blight",
  "confidence": 0.98,
  "low_confidence": false,
  "probabilities": [
    {"label": "Potato___Late_blight", "name": "Late Blight", "probability": 0.98},
    {"label": "Potato___Early_blight", "name": "Early Blight", "probability": 0.01},
    {"label": "Potato___healthy", "name": "Healthy Leaf", "probability": 0.01}
  ]
}
```

`GET /health` returns the loaded classes and the device in use.

## Tech Stack

Python, PyTorch, Hugging Face Transformers, Flask, Pillow, HTML/CSS/JavaScript

## Limitations

- Trained on a single public dataset; not validated on field conditions
- Supports only three classes (no other potato diseases or pests)
- Treatment text is general guidance. Confirm serious diagnoses with a local agronomist before treating.

## Author

Zeeshan Qasim. BS Computer Science, University of Lahore.
