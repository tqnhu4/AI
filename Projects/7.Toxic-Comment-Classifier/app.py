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