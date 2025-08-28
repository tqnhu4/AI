
-----

## Toxic Comment Classifier Project Guide

This project aims to build a tool that classifies comments into different categories of toxicity (e.g., polite, toxic, severe toxic, obscene, threat, insult, identity hate). We'll leverage the powerful Hugging Face Transformers library for state-of-the-art NLP models like BERT or RoBERTa.

### Requirement
📚 Library used: HuggingFace Transformers (BERT, RoBERTa)

✅ Description:
A small tool that helps classify comments into: polite / negative / offensive / threatening,...

✅ Technique used:
Data: Use the "Jigsaw Toxic Comment" dataset or separate Vietnamese data.

Model: Fine-tune the BERT model (multilingual or PhoBERT if using Vietnamese)

Simple UI: enter comments, get instant reviews

✅ Extensibility:
Use FastAPI to deploy API

Integrate into web/chat app to block spam

-----

### Project Goals:

1.  **Data Preparation**: Acquire and preprocess a suitable dataset for toxic comment classification.
2.  **Model Fine-tuning**: Fine-tune a pre-trained BERT-like model (e.g., `bert-base-uncased`, `bert-base-multilingual-cased`, or a language-specific model like PhoBERT for Vietnamese) on the classification task.
3.  **Simple UI**: Create a basic web interface using Streamlit for real-time comment classification.

### Technologies & Libraries:

  * **Hugging Face Transformers**: For pre-trained models and easy fine-tuning.
  * **PyTorch / TensorFlow**: The backend for Transformers (we'll primarily use PyTorch in this guide).
  * **Scikit-learn**: For data splitting and evaluation metrics.
  * **Pandas**: For data manipulation.
  * **Streamlit**: For creating a simple, interactive web UI.
  * **FastAPI** (Optional Extension): For building a robust API endpoint.
  * **Uvicorn** (Optional Extension): ASGI server for FastAPI.

-----

### Part 1: Core Classifier Development (Python Script)

First, let's build the core logic for data loading, model fine-tuning, and prediction.

#### Step 1: Set Up Your Environment

Create a new directory for your project and install the necessary libraries:

```bash
mkdir toxic_comment_classifier
cd toxic_comment_classifier
pip install torch transformers pandas scikit-learn accelerate datasets
# If you plan to use Streamlit later
pip install streamlit
# If you plan to use FastAPI later
pip install fastapi uvicorn
```

**Note:** `accelerate` is recommended by Hugging Face for faster training, especially on GPUs. `datasets` is useful for loading HF datasets easily.

#### Step 2: Choose Your Dataset

**Option A: Jigsaw Toxic Comment Classification (English)**
This is a standard dataset available on Kaggle. You'll need to download `train.csv` and `test.csv` from the competition page.

  * **Source:** [Jigsaw Toxic Comment Classification Challenge - Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

**Option B: Vietnamese Data (e.g., PhoBERT for Vietnamese)**
If you want to handle Vietnamese comments, you'll need a Vietnamese toxic comment dataset. You might have to create one or find publicly available ones. For this guide, we'll assume a similar structure to Jigsaw.

For this guide, we'll proceed assuming a structure similar to the Jigsaw dataset: a 'comment\_text' column and binary labels for multiple toxicity types (e.g., 'toxic', 'severe\_toxic', 'obscene', 'threat', 'insult', 'identity\_hate').

#### Step 3: Data Loading and Preprocessing

Create a Python file named `classifier_model.py`.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset # HuggingFace Dataset object
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# --- Configuration ---
MODEL_NAME = "bert-base-uncased" # Or "bert-base-multilingual-cased" for multi-language, or "vinai/phobert-base" for Vietnamese
MAX_LEN = 128 # Maximum sequence length for the model
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5 # Standard for fine-tuning
DATA_PATH = "train.csv" # Path to your Jigsaw train.csv or similar

# --- 1. Load Data ---
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Define labels based on Jigsaw dataset columns
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    # If your data has different labels or is single-label, adjust this
    # For a single label classifier (e.g., "toxic" vs "non-toxic"), adjust:
    # df['label'] = df['toxic'] # For binary classification
    return df, labels

# --- 2. Tokenization and Dataset Creation ---
def tokenize_function(examples, tokenizer, text_column, label_columns):
    # Tokenize the text
    tokenized_inputs = tokenizer(examples[text_column], truncation=True, padding='max_length', max_length=MAX_LEN)

    # Prepare labels for multi-label classification
    labels_batch = []
    for i in range(len(examples[text_column])):
        labels_batch.append([examples[col][i] for col in label_columns])
    tokenized_inputs["labels"] = labels_batch
    return tokenized_inputs

def create_datasets(df, labels, tokenizer, text_column='comment_text'):
    # Convert pandas DataFrame to HuggingFace Dataset
    hf_dataset = Dataset.from_pandas(df)

    # Split the dataset
    train_test_split_dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split_dataset['train']
    eval_dataset = train_test_split_dataset['test']

    # Apply tokenization
    train_tokenized_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, text_column, labels),
        batched=True,
        remove_columns=[text_column] + labels # Remove original text and label columns
    )
    eval_tokenized_dataset = eval_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, text_column, labels),
        batched=True,
        remove_columns=[text_column] + labels
    )

    # Set formats to PyTorch tensors
    train_tokenized_dataset.set_format("torch")
    eval_tokenized_dataset.set_format("torch")

    return train_tokenized_dataset, eval_tokenized_dataset

# --- 3. Define Metrics (for multi-label classification) ---
def compute_metrics(p):
    predictions, labels = p
    # Apply sigmoid to predictions if output from model is logits
    preds = (torch.sigmoid(torch.tensor(predictions)) > 0.5).int().numpy()
    labels = labels.astype(int)

    # For multi-label, compute metrics per label and then average (e.g., micro or macro)
    f1_micro = f1_score(labels, preds, average='micro')
    f1_macro = f1_score(labels, preds, average='macro')
    accuracy = accuracy_score(labels, preds)
    
    # AUC score per label if applicable, then average
    # Note: ROC AUC requires probabilities, not binary predictions
    # This might require careful handling if some labels have no positive samples in a batch.
    # For simplicity, we'll focus on F1 and Accuracy here.
    # To properly compute AUC for multi-label, you often calculate it per class and average.
    # roc_auc = roc_auc_score(labels, predictions, average='macro') # Use raw predictions for AUC

    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'accuracy': accuracy,
        # 'roc_auc': roc_auc,
    }


# --- 4. Model Training ---
def train_model(train_dataset, eval_dataset, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels, problem_type="multi_label_classification")
    # For PhoBERT, you might need to specify the tokenizer based on its config, e.g., AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        load_best_model_at_end=True, # Load the best model found during training
        metric_for_best_model="f1_micro", # Metric to monitor for best model
        save_strategy="epoch",
        # For GPU usage, ensure you have CUDA set up and PyTorch built with CUDA support.
        # This will automatically use GPU if available.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer, # Pass tokenizer to Trainer for automatic padding
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("./fine_tuned_model") # Save the fine-tuned model
    tokenizer.save_pretrained("./fine_tuned_model") # Save tokenizer alongside the model

    return model

# --- 5. Prediction Function ---
def predict_comment(comment_text, model, tokenizer, labels):
    inputs = tokenizer(comment_text, return_tensors="pt", truncation=True, padding='max_length', max_length=MAX_LEN)
    
    # Move inputs to the same device as model (GPU if available)
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        model.to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)

    # Apply sigmoid to get probabilities for multi-label classification
    probabilities = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    
    # Threshold for prediction (e.g., 0.5)
    predictions = (probabilities > 0.5).astype(int)

    results = {}
    for i, label in enumerate(labels):
        results[label] = {
            'probability': float(probabilities[i]),
            'is_toxic': bool(predictions[i])
        }
    return results

if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load data
    df, labels = load_data(DATA_PATH)
    num_labels = len(labels)

    st.write(f"Loaded {len(df)} comments with {num_labels} labels: {labels}")

    # Create datasets
    train_dataset, eval_dataset = create_datasets(df, labels, tokenizer)
    st.write(f"Training samples: {len(train_dataset)}, Evaluation samples: {len(eval_dataset)}")

    # Train model
    # Uncomment the following lines to train the model
    # st.write("Starting model training...")
    # trained_model = train_model(train_dataset, eval_dataset, num_labels)
    # st.write("Model training complete and saved to './fine_tuned_model'")

    # --- For demonstration, load a pre-trained model instead of training every time ---
    # Assuming you have run the training once and saved the model
    try:
        st.write("Attempting to load a pre-trained model from './fine_tuned_model'...")
        loaded_model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")
        loaded_tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
        st.success("Model and tokenizer loaded successfully!")
        
        # Example prediction
        sample_comment = "This is a great movie, I love it!"
        # sample_comment = "You are an idiot and I hate you." # For a toxic example
        st.write(f"\nExample prediction for: '{sample_comment}'")
        results = predict_comment(sample_comment, loaded_model, loaded_tokenizer, labels)
        for label, data in results.items():
            st.write(f"- {label}: Probability={data['probability']:.4f}, Is Toxic={data['is_toxic']}")

    except Exception as e:
        st.error(f"Could not load pre-trained model. Please ensure you have run the training once and saved it. Error: {e}")
        st.info("You need to uncomment the `train_model` lines above and run this script once to train and save the model.")

```

**Explanation of `classifier_model.py`:**

1.  **Configuration**: Defines constants like `MODEL_NAME`, `MAX_LEN`, `BATCH_SIZE`, etc.
      * **`MODEL_NAME`**:
          * `"bert-base-uncased"`: Standard English BERT.
          * `"bert-base-multilingual-cased"`: For multiple languages, including English. This is a good choice if you have a mix of languages or want to generalize.
          * `"vinai/phobert-base"`: Specifically for Vietnamese. If using this, ensure your `DATA_PATH` points to a Vietnamese dataset, and you might need `underthesea` or `vncorenlp` for proper tokenization/word segmentation *before* passing to PhoBERT if not already segmented (PhoBERT expects segmented input).
2.  **`load_data()`**: Reads your CSV and identifies the label columns.
3.  **`tokenize_function()`**: This function is mapped over your dataset.
      * It uses `tokenizer()` to convert text into numerical IDs, attention masks, and token type IDs.
      * `truncation=True` and `padding='max_length'` ensure all sequences are the same length (`MAX_LEN`).
      * It also prepares the `labels` for multi-label classification (a list of 0s and 1s for each sample).
4.  **`create_datasets()`**:
      * Converts the Pandas DataFrame to a `HuggingFace Dataset` object, which is optimized for training.
      * Splits the dataset into train and evaluation sets.
      * Applies `tokenize_function` using `.map()` for efficient processing.
      * `set_format("torch")` makes the dataset return PyTorch tensors, ready for the model.
5.  **`compute_metrics()`**: Defines how to evaluate your model during training.
      * For multi-label classification, `torch.sigmoid()` is applied to the raw logits to get probabilities.
      * A threshold (0.5) converts probabilities to binary predictions.
      * `f1_score(average='micro')` and `f1_score(average='macro')` are common metrics for multi-label.
6.  **`train_model()`**:
      * `AutoModelForSequenceClassification.from_pretrained()`: Loads the pre-trained model.
      * `num_labels`: Crucial for multi-label classification, tells the model how many output labels to expect.
      * `problem_type="multi_label_classification"`: Tells the model to use appropriate loss functions (e.g., BCEWithLogitsLoss).
      * `TrainingArguments`: Configures the training process (epochs, batch size, learning rate, saving strategy, evaluation strategy).
      * `Trainer`: Hugging Face's high-level API for training models. Simplifies the training loop significantly.
      * `trainer.train()`: Starts the training process.
      * `trainer.save_model()` and `tokenizer.save_pretrained()`: Saves the fine-tuned model weights and the tokenizer configuration for later use.
7.  **`predict_comment()`**:
      * Takes a raw comment, tokenizes it, and passes it to the model.
      * Uses `torch.no_grad()` to disable gradient calculation during inference, saving memory and speeding up.
      * Applies `torch.sigmoid()` to the model's output (logits) to get probabilities for each label.
      * Applies a 0.5 threshold to determine if a label is active (`is_toxic`).

#### To Run `classifier_model.py`:

1.  **Download `train.csv`** from the Jigsaw Kaggle competition and place it in the same directory as `classifier_model.py`.
2.  **Uncomment the `train_model` lines** in the `if __name__ == "__main__":` block.
3.  Run the script from your terminal: `python classifier_model.py`
4.  This will train the model and save it to a folder named `fine_tuned_model`. After the first run, you can comment out the `train_model` lines again to just load and test.

-----

### Part 2: Simple UI with Streamlit

Now, let's create a user interface to interact with your trained model.

Create a new Python file named `app.py`.

```python
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd # Just for labels loading, if needed from CSV structure

# --- Configuration (Must match your training config) ---
MODEL_PATH = "./fine_tuned_model" # Path where your trained model is saved
MAX_LEN = 128
# Define labels (Must be the same order as used during training)
# If your labels come from a specific CSV, you might load them differently
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

@st.cache_resource # Cache the model loading to prevent reloading on every rerun
def load_model_and_tokenizer():
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        # Move model to GPU if available
        if torch.cuda.is_available():
            model.to('cuda')
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        st.stop() # Stop the app if model can't be loaded

# --- Prediction Function (similar to classifier_model.py) ---
def predict_comment(comment_text, model, tokenizer, labels):
    inputs = tokenizer(comment_text, return_tensors="pt", truncation=True, padding='max_length', max_length=MAX_LEN)
    
    # Move inputs to the same device as model
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        model.to('cuda') # Ensure model is on GPU if not already

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    predictions = (probabilities > 0.5).astype(int)

    results = {}
    for i, label in enumerate(labels):
        results[label] = {
            'probability': float(probabilities[i]),
            'is_toxic': bool(predictions[i])
        }
    return results

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Toxic Comment Classifier", layout="centered", icon="💬")
    st.title("💬 Toxic Comment Classifier")
    st.markdown("A tool to classify comments into different toxicity categories using a fine-tuned BERT model.")
    st.markdown("---")

    st.info(f"Model used: `{MODEL_PATH}` | Labels: `{', '.join(LABELS)}`")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    st.subheader("Enter Your Comment Below:")
    comment_input = st.text_area("Type your comment here...", height=150, help="Enter any text to classify its toxicity.")

    if st.button("Classify Comment"):
        if comment_input.strip() == "":
            st.warning("Please enter a comment to classify.")
        else:
            with st.spinner("Classifying..."):
                results = predict_comment(comment_input, model, tokenizer, LABELS)
            
            st.subheader("Classification Results:")
            
            # Check if any toxic label is predicted
            is_any_toxic = any(res['is_toxic'] for res in results.values())
            
            if is_any_toxic:
                st.error("This comment is detected as **TOXIC**!")
            else:
                st.success("This comment is detected as **POLITE**.")

            st.markdown("---")
            st.write("Detailed Probabilities:")
            
            # Display results in a table
            display_data = []
            for label, data in results.items():
                display_data.append({
                    "Category": label.replace('_', ' ').title(),
                    "Probability": f"{data['probability']:.2%}",
                    "Is Toxic": "✅ Yes" if data['is_toxic'] else "❌ No"
                })
            
            st.dataframe(pd.DataFrame(display_data).set_index("Category"))

            st.markdown("---")
            st.write("Disclaimer: This is a simplified model for demonstration purposes. Its accuracy depends on the training data and model complexity.")

if __name__ == "__main__":
    main()
```

**Explanation of `app.py`:**

1.  **`@st.cache_resource`**: This decorator is crucial\! It tells Streamlit to load the model and tokenizer only once when the application starts, even if the user interacts with other widgets. Without this, the model would reload every time, making the app very slow.
2.  **`load_model_and_tokenizer()`**: Handles loading the pre-trained model and tokenizer from the `MODEL_PATH`. It also checks for GPU availability.
3.  **`predict_comment()`**: This is almost identical to the function in `classifier_model.py`, adapted for the Streamlit environment.
4.  **Streamlit UI (`main()` function)**:
      * `st.set_page_config()`: Configures the app's appearance.
      * `st.title()`, `st.markdown()`: For text display.
      * `st.text_area()`: The input field where users type their comments.
      * `st.button("Classify Comment")`: Triggers the classification process.
      * `st.spinner("Classifying...")`: Shows a loading spinner while the prediction is running.
      * `st.error()`, `st.success()`: Display prominent messages based on whether toxicity is detected.
      * `st.dataframe()`: Presents the detailed probabilities and toxicity status for each category in a neat table.

#### To Run `app.py`:

1.  Ensure you have run `classifier_model.py` at least once and the `fine_tuned_model` directory exists.
2.  Run the Streamlit app from your terminal in the project directory:
    ```bash
    streamlit run app.py
    ```
    Your browser will open, showing the comment classifier interface.

-----

### Part 3: Optional Extensions

#### Extension 1: Deploying as an API with FastAPI

For more robust integration with other applications or for production deployment, an API is ideal.

Create a new Python file named `api_server.py`.

```python
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
```

**Explanation of `api_server.py`:**

1.  **Model Loading**: The model and tokenizer are loaded globally *once* when the FastAPI application starts, similar to `st.cache_resource`. This is crucial for performance. `model.eval()` sets the model to evaluation mode, disabling dropout and batch normalization updates.
2.  **`FastAPI` App**: Initializes the FastAPI application.
3.  **`Pydantic` Models**: `CommentRequest`, `ClassificationResult`, and `PredictionResponse` define the structure of the incoming request body and the outgoing response body. This provides automatic data validation and clear API documentation.
4.  **`predict_comment_api()`**: The core prediction logic, adapted to return the Pydantic `PredictionResponse`.
5.  **`@app.post("/classify", ...)`**: Defines a POST API endpoint.
      * `/classify`: The URL path.
      * `response_model=PredictionResponse`: Ensures the response conforms to the defined Pydantic model.
      * `async def classify_comment(request: CommentRequest)`: Defines the asynchronous function handling the request, expecting a `CommentRequest` object.
      * `HTTPException`: Used to return HTTP error codes (e.g., 400 for bad request, 500 for internal server error).
6.  **`@app.get("/health")`**: A simple endpoint to check if the API is running and if the model is loaded correctly.
7.  **`uvicorn.run(app, ...)`**: This block allows you to run the FastAPI application programmatically.

#### To Run FastAPI Server:

1.  Ensure you have `fastapi` and `uvicorn` installed.

2.  Run the API server from your terminal:

    ```bash
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
    ```

      * `api_server`: The Python file name.
      * `app`: The FastAPI application instance within that file.
      * `--host 0.0.0.0`: Makes the API accessible from other devices on your network (for testing).
      * `--port 8000`: Runs the API on port 8000.
      * `--reload`: Reloads the server automatically on code changes (useful during development).

    You can then test the API by going to `http://127.0.0.1:8000/docs` in your browser to access the interactive OpenAPI (Swagger UI) documentation.

#### Extension 2: Integration into a Web/Chat App

Once you have the FastAPI endpoint, integrating it into a web or chat application is straightforward:

  * **Web App (Frontend)**: Use JavaScript's `fetch` API or `axios` to send POST requests to your `/classify` endpoint with the comment text, and then display the returned JSON data.
  * **Chatbot**: If you have a chatbot framework (e.g., Rasa, Botpress, custom solution), you can configure it to send user messages to your FastAPI endpoint and then use the `is_overall_toxic` flag or specific categories to decide on actions (e.g., block the message, flag it for review, respond with a warning).

**Example (Conceptual JavaScript for a simple web page):**

```html
<textarea id="commentInput" rows="5" cols="50"></textarea><br>
<button onclick="classifyComment()">Classify</button>
<pre id="results"></pre>

<script>
async function classifyComment() {
    const commentText = document.getElementById('commentInput').value;
    const resultsDiv = document.getElementById('results');
    resultsDiv.textContent = 'Classifying...';

    try {
        const response = await fetch('http://127.0.0.1:8000/classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ comment_text: commentText })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(`HTTP error! status: ${response.status}, detail: ${errorData.detail || response.statusText}`);
        }

        const data = await response.json();
        resultsDiv.textContent = JSON.stringify(data, null, 2); // Pretty print JSON

        // You can then parse 'data' and display it nicely in your UI
        // For example: if (data.is_overall_toxic) { alert('Toxic comment!'); }
        // document.getElementById('toxicityStatus').textContent = data.is_overall_toxic ? 'Toxic' : 'Polite';

    } catch (error) {
        console.error('Error classifying comment:', error);
        resultsDiv.textContent = `Error: ${error.message}`;
    }
}
</script>
```

-----

### Final Project Structure:

```
toxic_comment_classifier/
├── train.csv                 # Your Jigsaw (or Vietnamese) training data
├── classifier_model.py       # Script for data prep, training, and saving the model
├── app.py                    # Streamlit web application
├── api_server.py             # FastAPI server for API deployment
└── fine_tuned_model/         # Directory created after training
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    ├── vocab.txt
    └── ... (other tokenizer files)
```
