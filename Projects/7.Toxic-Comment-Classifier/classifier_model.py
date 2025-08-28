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