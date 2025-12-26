import os

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer

app = FastAPI()

# Setup paths (adjusted for container or local run)
# In simpsons they use /triton/sources/... but we will try to be flexible
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.onnx")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Initialize model and tokenizer
print(f"Loading model from {MODEL_PATH}...")
ort_session = None
tokenizer = None

try:
    ort_session = ort.InferenceSession(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # We don't exit here to allow server to start and show error on request if needed,
    # but practically it's better to fail early.
    ort_session = None

templates = Jinja2Templates(directory=STATIC_DIR)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(text: str = Form(...)):  # noqa: B008
    if ort_session is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # 1. Tokenize
        encoding = tokenizer.encode_plus(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="np",
        )

        # 2. Prepare inputs dynamically (handling float/int types)
        inputs = {}
        for inp in ort_session.get_inputs():
            if "input_ids" in inp.name:
                dtype = np.float32 if "float" in inp.type else np.int64
                inputs[inp.name] = encoding["input_ids"].astype(dtype)
            elif "attention_mask" in inp.name:
                dtype = np.float32 if "float" in inp.type else np.int64
                inputs[inp.name] = encoding["attention_mask"].astype(dtype)

        # 3. Run inference
        output_name = ort_session.get_outputs()[0].name
        logits = ort_session.run([output_name], inputs)[0]

        # 4. Post-process
        prob = 1 / (1 + np.exp(-logits))
        prob = prob.squeeze().item()

        prediction = "FAKE" if prob > 0.5 else "REAL"
        confidence = prob if prob > 0.5 else 1 - prob

        return {
            "class": prediction,
            "probability": float(prob),
            "confidence": float(confidence),
            "text": text,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
