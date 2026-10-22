# SpudScan - Potato Leaf Disease Detection

## Structure
potato_app/
  app.py
  requirements.txt
  model/                      <- put your 3 files here
    config.json
    preprocessor_config.json
    model.safetensors
  templates/index.html

## Run
python -m venv venv
venv\Scripts\activate          (Windows)   |   source venv/bin/activate (Mac/Linux)
pip install -r requirements.txt
python app.py
Open http://127.0.0.1:5000
