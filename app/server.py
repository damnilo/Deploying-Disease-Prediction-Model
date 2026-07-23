import torch
from fastapi import FastAPI
import joblib
import numpy as np
from neural_network import SymptomClassifier

mlb = joblib.load('app/mlb.joblib')
le = joblib.load('app/le.joblib')

model = SymptomClassifier(input=len(mlb.classes_), output=len(le.classes_))
model.load_state_dict(torch.load('app/model.pt'))
model.eval()
app = FastAPI()

@app.get("/")
def read_root():
    return {"Message": "Disease classification model API"}

@app.post("/predict")
def predict(data: dict):
    symptoms = data["symptoms"]
    features = np.array([mlb.transform([symptoms])[0]], dtype=np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(features))
        pred_idx = torch.argmax(logits, dim=1).item()
    return {"Prediction": le.classes_[pred_idx]}