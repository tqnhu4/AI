from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

# --- Configuration (Must match your training config) ---
MODEL_PATH = "./fine_tuned_model"
MAX_LEN = 128
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# --- Load Model and Tokenizer (Load once when app starts) ---
try:
    global model, tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # Move model to GPU if available
    if torch.cuda.is_available():
        model.to('cuda')
    model.eval() # Set model to evaluation mode
    print("Model and tokenizer loaded successfully for API.")
except Exception as e:
    print(f"Error loading model or tokenizer for API: {e}")
    # Exit if model can't be loaded, as the API won't function
    import sys
    sys.exit(1)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Toxic Comment Classifier API",
    description="API for classifying comment toxicity using a fine-tuned BERT model.",
    version="1.0.0"
)

# --- Pydantic Model for Request Body ---
class CommentRequest(BaseModel):
    comment_text: str

# --- Pydantic Model for Response Body ---
class ClassificationResult(BaseModel):
    category: str
    probability: float
    is_toxic: bool

class PredictionResponse(BaseModel):
    comment_text: str
    is_overall_toxic: bool
    predictions: list[ClassificationResult]

# --- Prediction Logic (similar to previous files) ---
def predict_comment_api(comment_text: str):
    inputs = tokenizer(comment_text, return_tensors="pt", truncation=True, padding='max_length', max_length=MAX_LEN)
    
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    predictions = (probabilities > 0.5).astype(int)

    results = []
    is_overall_toxic = False
    for i, label in enumerate(LABELS):
        is_toxic_category = bool(predictions[i])
        if is_toxic_category:
            is_overall_toxic = True # Mark true if any category is toxic
        results.append(ClassificationResult(
            category=label,
            probability=float(probabilities[i]),
            is_toxic=is_toxic_category
        ))
    
    return PredictionResponse(
        comment_text=comment_text,
        is_overall_toxic=is_overall_toxic,
        predictions=results
    )

# --- API Endpoint ---
@app.post("/classify", response_model=PredictionResponse)
async def classify_comment(request: CommentRequest):
    """
    Classify a given comment text into different toxicity categories.
    """
    if not request.comment_text.strip():
        raise HTTPException(status_code=400, detail="Comment text cannot be empty.")
    
    try:
        prediction = predict_comment_api(request.comment_text)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": True if 'model' in globals() else False}

# To run this file:
# uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
# Or run programmatically:
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)