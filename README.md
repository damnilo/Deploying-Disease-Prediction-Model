# Symptom-Based Disease Classifier API

A PyTorch neural network that predicts a disease from a list of reported symptoms, served via FastAPI and containerized with Docker.

## Overview

- **Model**: Feedforward neural network (131 → 128 → 64 → 41) trained on multi-hot encoded symptoms
- **Dataset**: 41 diseases, 131 unique symptoms (Kaggle "Disease Prediction Using Machine Learning" dataset)
- **Serving**: FastAPI REST API with automatic Swagger docs at `/docs`
- **Deployment**: Docker container, runs locally on port 8000

## Project Structure
project/
├── model.py # Shared PyTorch model architecture (used by train.py and server.py)
├── train.py # Training script — trains the model and saves artifacts
├── client.py # Example script to call the running API
├── requirements.txt
├── Dockerfile
└── app/
├── server.py # FastAPI app
├── model.pt # Saved model weights (state_dict)
├── mlb.joblib # Fitted MultiLabelBinarizer (symptom vocabulary)
└── le.joblib # Fitted LabelEncoder (disease labels)

## Setup

### 1. Train the model

```bash
python train.py
```

This reads `Dataset/dataset.csv`, trains the model, and saves `model.pt`, `mlb.joblib`, and `le.joblib` into `app/`.

### 2. Build the Docker image

```bash
docker build -t symptom-classifier .
```

### 3. Run the container

```bash
docker run --name symptom-api -p 8000:8000 symptom-classifier
```

The API is now available at `http://localhost:8000`.

## API Endpoints

### `GET /`
Health check.

### `GET /symptoms`
Returns the list of valid symptom names the model was trained on.

```json
{ "available_symptoms": ["itching", "skin_rash", "..."] }
```

### `POST /predict`
Predicts a disease from a list of symptoms.

**Request body:**
```json
{
  "symptoms": ["itching", "skin_rash", "dischromic _patches"]
}
```

**Response:**
```json
{ "Prediction": "Fungal infection" }
```

> Note: symptom names must match exactly what `GET /symptoms` returns, including any embedded whitespace inherited from the original dataset (e.g. `"dischromic _patches"`).

## Interactive Docs

Once the container is running, visit:

http://localhost:8000/docs

for a Swagger UI where you can test the `/predict` endpoint directly in the browser.

## Testing the API from Python

```bash
python client.py
```

See `client.py` for a minimal example using the `requests` library.

## Known Limitations

- Trained on a small, deduplicated dataset (304 unique symptom-disease combinations)
- Dataset is synthetic; symptom combinations per disease may not reflect real-world clinical variability
- No input validation against unknown/misspelled symptom names beyond what `MultiLabelBinarizer` silently ignores

## Next Steps

- Push the image to a container registry (Docker Hub / GitHub Container Registry)
- Deploy to a hosting platform (Render, Railway, Fly.io, etc.)
- Add a simple HTML frontend for symptom selection instead of raw JSON requests
