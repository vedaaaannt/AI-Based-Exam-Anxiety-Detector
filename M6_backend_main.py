# Run with: uvicorn milestone6.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs

import os
import time
import logging
from typing import Optional
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Activity 6.1 — FastAPI Selection ─────────────────────────
# FastAPI chosen for: async support, auto Swagger UI,
# Pydantic validation, high performance, clean Python typing

MODEL_DIR = os.getenv("MODEL_DIR", "model/saved_model")
MAX_LEN   = int(os.getenv("MAX_LEN", 128))
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

LABEL2ID  = {"Low Anxiety": 0, "Moderate Anxiety": 1, "High Anxiety": 2}
ID2LABEL  = {0: "Low Anxiety", 1: "Moderate Anxiety", 2: "High Anxiety"}

# Metadata returned with every prediction
ANXIETY_META = {
    "Low Anxiety": {
        "emoji": "😊", "color": "#27AE60", "level": 1,
        "message": "Your anxiety level is low. Keep up your positive mindset!",
        "tips": [
            "You're doing great! Maintain your study routine.",
            "Take short breaks every 45 minutes to stay fresh.",
            "Practice light breathing before the exam.",
            "Get a good night's sleep — you're well-prepared!",
        ],
    },
    "Moderate Anxiety": {
        "emoji": "😐", "color": "#F39C12", "level": 2,
        "message": "You're experiencing moderate anxiety. Some strategies can help.",
        "tips": [
            "Try the 4-7-8 breathing technique to calm nerves.",
            "Break revision into smaller chunks — one topic at a time.",
            "Write down your worries to clear your head.",
            "Avoid comparing yourself to classmates.",
        ],
    },
    "High Anxiety": {
        "emoji": "😰", "color": "#E74C3C", "level": 3,
        "message": "High anxiety detected. Please seek support — you are not alone.",
        "tips": [
            "Please speak with a counselor or trusted adult immediately.",
            "Ground yourself: name 5 things you can see right now.",
            "Step away from study materials for 30 minutes and breathe.",
            "Remember: one exam does not define your worth or future.",
        ],
    },
}

# ── Activity 6.2 — Loading the Trained BERT Model ────────────
# Global model and tokenizer loaded once at startup
tokenizer_global: Optional[BertTokenizer] = None
model_global: Optional[BertForSequenceClassification] = None

def load_bert_model():
    global tokenizer_global, model_global
    try:
        logger.info(f"Loading BERT from: {MODEL_DIR}")
        tokenizer_global = BertTokenizer.from_pretrained(MODEL_DIR)
        model_global     = BertForSequenceClassification.from_pretrained(MODEL_DIR)
        model_global.to(DEVICE)
        model_global.eval()   # inference mode — no dropout
        logger.info(f"Model loaded on {DEVICE}")
    except Exception as e:
        logger.warning(f"Could not load model: {e} — running in DEMO mode")


# ── Activity 6.3 — Request & Response Schemas ────────────────
# Pydantic models enforce types and auto-document the API

class PredictRequest(BaseModel):
    text: str = Field(
        ..., min_length=5, max_length=1000,
        example="I'm really nervous about tomorrow's exam. I can't stop worrying.",
    )
    student_id: Optional[str] = Field(None, example="STU_001")

class AnxietyResult(BaseModel):
    label: str
    confidence: float
    level: int                      # 1 = Low, 2 = Moderate, 3 = High
    emoji: str
    color: str
    message: str
    tips: list[str]
    probabilities: dict[str, float]
    inference_time_ms: float

class PredictResponse(BaseModel):
    success: bool
    student_id: Optional[str]
    input_text: str
    result: AnxietyResult

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str


# ── Activity 6.4 — Prediction Endpoint Logic ─────────────────
# Tokenize text → BERT forward pass → return label + probs
def run_inference(text):
    start = time.time()

    if model_global is None:
        # Demo mode — random output when model is not loaded
        probs    = np.random.dirichlet([1, 1, 1]).tolist()
        label_id = int(np.argmax(probs))
        return {"label": ID2LABEL[label_id], "confidence": round(max(probs), 4),
                "probabilities": {ID2LABEL[i]: round(p, 4) for i, p in enumerate(probs)},
                "inference_ms": round((time.time() - start) * 1000, 2)}

    inputs = tokenizer_global(text, return_tensors="pt", truncation=True,
                              padding="max_length", max_length=MAX_LEN)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():   # disable gradient tracking during inference
        probs = torch.softmax(model_global(**inputs).logits, dim=-1).squeeze().cpu().numpy()

    label_id = int(np.argmax(probs))
    return {
        "label":         ID2LABEL[label_id],
        "confidence":    round(float(probs[label_id]), 4),
        "probabilities": {ID2LABEL[i]: round(float(p), 4) for i, p in enumerate(probs)},
        "inference_ms":  round((time.time() - start) * 1000, 2),
    }


# ── Activity 6.5 — FastAPI App & Routes ──────────────────────
app = FastAPI(
    title="Exam Anxiety Detector API",
    description="BERT-powered exam anxiety classification: Low / Moderate / High",
    version="1.0.0",
)

# Allow requests from Streamlit frontend on any origin
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])

@app.on_event("startup")
async def startup():
    load_bert_model()   # load model when server starts

@app.get("/", tags=["Info"])
def root():
    return {"message": "Exam Anxiety Detector API 🎓", "docs": "/docs", "health": "/health"}

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(status="ok", model_loaded=model_global is not None,
                          device=DEVICE, version="1.0.0")

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Input text cannot be empty.")

    raw  = run_inference(request.text)
    meta = ANXIETY_META[raw["label"]]

    result = AnxietyResult(
        label             = raw["label"],
        confidence        = raw["confidence"],
        level             = meta["level"],
        emoji             = meta["emoji"],
        color             = meta["color"],
        message           = meta["message"],
        tips              = meta["tips"],
        probabilities     = raw["probabilities"],
        inference_time_ms = raw["inference_ms"],
    )
    logger.info(f"Predicted: {raw['label']} ({raw['confidence']:.2%}) for: '{request.text[:50]}'")
    return PredictResponse(success=True, student_id=request.student_id,
                           input_text=request.text, result=result)

# ── Activity 6.6 — Batch endpoint for institution monitoring ──
@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(requests_list: list[PredictRequest]):
    # Allows up to 50 texts in a single API call
    if len(requests_list) > 50:
        raise HTTPException(status_code=400, detail="Max 50 texts per batch.")
    return [predict(r) for r in requests_list]


if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI backend → http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
