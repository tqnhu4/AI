
## 🎙️ Voice-to-Text with Emotion Analysis Application

This is an exciting project that showcases advanced skills\! We'll build a **Voice-to-Text application with Emotion Analysis** using modern technologies.

Since integrating direct browser recording, managing API keys, and deploying a local AI model require multiple steps and a complex environment (e.g., a backend server is needed, not just a simple Python script), I'll focus on describing the application structure and providing the core code snippets for the main components: **Automatic Speech Recognition (ASR)** and **Emotion Analysis**. You'll need to combine these parts into a complete web application (e.g., Flask or Streamlit) or a desktop app.

-----

## ⚙️ Technologies Used

Our application will follow this flow:

1.  **Audio Recording/Upload**: The user provides an audio file (either by direct recording or uploading a file).
2.  **Voice-to-Text (ASR)**: An AI model converts the audio into text.
3.  **Emotion Analysis**: Another AI model analyzes the transcribed text to predict emotions (e.g., positive, negative, neutral).
4.  **Display Results**: The original text and emotion analysis results are presented to the user.

-----

## 🛠️ Implementation Steps & Core Code

### Step 1: Install Necessary Libraries

First, let's install the Python libraries we'll be using.

```bash
pip install openai # For Whisper API
pip install transformers # For HuggingFace models
pip install torch # PyTorch, a deep learning framework needed by Transformers
pip install -U huggingface_hub # Ensure you have the latest version for model download
```

**Important Note**: You'll need an OpenAI account to get an **API Key**. Keep this API Key secure; do not share it publicly.

```python
# Set OpenAI API Key
import os
# Set as an environment variable or replace directly (not recommended for production)
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" # Replace with your actual API Key
```

### Step 2: Voice-to-Text (ASR) with Whisper API

We'll create a function to perform this.

```python
import openai

def transcribe_audio_whisper(audio_file_path):
    """
    Uses the OpenAI Whisper API to convert speech to text.

    Args:
        audio_file_path (str): The path to the audio file.

    Returns:
        str: The transcribed text, or None if an error occurs.
    """
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            return transcript.text
    except Exception as e:
        print(f"Error during speech-to-text conversion: {e}")
        return None

# --- ASR Usage Example ---
# Assume you have a sample audio file, e.g., 'sample_audio.mp3'
# You can record one yourself or download a small sample file.
# Example: an audio file containing the sentence "Hello, how are you today? I am feeling good."

# To test, create a dummy audio file or ensure you have a real one:
# content = "This is a test audio file. I am happy to demonstrate voice-to-text."
# # In a real-world scenario, you'd need a tool to create audio from text
# # or record voice. Example with gTTS (Google Text-to-Speech) to create a test file:
# # pip install gtts
# from gtts import gTTS
# tts = gTTS(content, lang='en')
# tts.save("sample_audio.mp3")

# audio_path = "sample_audio.mp3" # Change this to your audio file path
# if os.path.exists(audio_path):
#     transcribed_text = transcribe_audio_whisper(audio_path)
#     if transcribed_text:
#         print("\n--- Speech-to-Text (ASR) Result ---")
#         print(f"Text: \"{transcribed_text}\"")
# else:
#     print(f"Error: Audio file not found at {audio_path}. Please create or provide the file.")
```

### Step 3: Emotion Analysis with Hugging Face Transformers

We'll use a pre-trained sentiment analysis model. The `cardiffnlp/twitter-roberta-base-sentiment-latest` model is a good choice for general sentiment analysis.

```python
from transformers import pipeline

def analyze_sentiment(text):
    """
    Analyzes the sentiment of text using a Hugging Face Transformers model.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: The sentiment analysis result (label and score), or None if an error occurs.
    """
    if not text:
        return None

    try:
        # Load pre-trained sentiment analysis pipeline
        # You can try other models like "dair-ai/emotion" for more detailed emotion classification
        sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        # Or use a model specialized for detailed emotions:
        # sentiment_analyzer = pipeline("text-classification", model="dair-ai/emotion", top_k=None)

        result = sentiment_analyzer(text)
        return result[0] # The pipeline returns a list of results
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return None

# --- Sentiment Analysis Usage Example ---
# Use the transcribed text or another sample text
sample_text_positive = "I love this product, it's absolutely amazing and wonderful!"
sample_text_negative = "This is a terrible service, I'm very disappointed."
sample_text_neutral = "The weather is cloudy today."

print("\n--- Sentiment Analysis Results ---")
sentiment_pos = analyze_sentiment(sample_text_positive)
if sentiment_pos:
    print(f"Text: \"{sample_text_positive}\"")
    print(f"Sentiment: {sentiment_pos['label']} (Score: {sentiment_pos['score']:.4f})")

sentiment_neg = analyze_sentiment(sample_text_negative)
if sentiment_neg:
    print(f"Text: \"{sample_text_negative}\"")
    print(f"Sentiment: {sentiment_neg['label']} (Score: {sentiment_neg['score']:.4f})")

sentiment_neu = analyze_sentiment(sample_text_neutral)
if sentiment_neu:
    print(f"Text: \"{sample_text_neutral}\"")
    print(f"Sentiment: {sentiment_neu['label']} (Score: {sentiment_neu['score']:.4f})")

# Combine with ASR results
# if transcribed_text:
#     sentiment_from_asr = analyze_sentiment(transcribed_text)
#     if sentiment_from_asr:
#         print(f"\n--- Sentiment Analysis from ASR ---")
#         print(f"ASR Text: \"{transcribed_text}\"")
#         print(f"Sentiment: {sentiment_from_asr['label']} (Score: {sentiment_from_asr['score']:.4f})")
```

-----

## 🚀 Completing the Application (Deployment Ideas)

To have a complete application, you'll need a user interface. Here are a few options:

### 1\. Web Application with Flask (or Django/FastAPI)

  * **Backend (Python)**: Handles audio file uploads, calls the ASR function, calls the emotion analysis function, and returns results.
  * **Frontend (HTML/CSS/JavaScript)**:
      * Provides an interface for users to select/record audio files.
      * Sends audio files to the backend.
      * Displays the transcribed text and emotion analysis results.
      * Can use a JavaScript library like `MediaRecorder` for direct browser recording.

**Suggested Flask File Structure:**

```
your_app_name/
├── app.py           # Main Flask logic (routes, calls ASR/Sentiment functions)
├── templates/
│   └── index.html   # User interface
├── static/
│   └── style.css    # CSS
│   └── script.js    # JavaScript for recording/uploading
└── requirements.txt # List of Python libraries
```

**Basic Flask Code for `app.py`:**

```python
# app.py (A simple example to give you an idea)
from flask import Flask, request, render_template, flash, redirect, url_for
import os
# Import your ASR and sentiment analysis functions
# from your_asr_module import transcribe_audio_whisper
# from your_sentiment_module import analyze_sentiment

app = Flask(__name__)
app.secret_key = 'supersecretkey' # Needed for flash messages
app.config['UPLOAD_FOLDER'] = 'uploads' # Where temporary audio files will be stored
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set OpenAI API key (for illustration only, use environment variables in production)
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

@app.route('/', methods=['GET', 'POST'])
def index():
    transcribed_text = None
    sentiment_result = None

    if request.method == 'POST':
        if 'audio_file' not in request.files:
            flash('No audio file part', 'error')
            return redirect(request.url)

        file = request.files['audio_file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Step 1: Speech-to-Text conversion
            transcribed_text = transcribe_audio_whisper(filepath)
            os.remove(filepath) # Delete temporary file after processing

            if transcribed_text:
                # Step 2: Emotion Analysis
                sentiment_result = analyze_sentiment(transcribed_text)
                if not sentiment_result:
                    flash('Error during sentiment analysis.', 'error')
            else:
                flash('Error during speech-to-text conversion.', 'error')

    return render_template('index.html',
                           transcribed_text=transcribed_text,
                           sentiment_result=sentiment_result)

if __name__ == '__main__':
    # Ensure ASR and sentiment analysis functions are defined/imported correctly
    # And API key is set
    app.run(debug=True)
```

### 2\. Application with Streamlit (Simple and Fast)

Streamlit allows you to create interactive web applications with simple Python scripts, making it ideal for demos and rapid development.

```bash
pip install streamlit
pip install pydub # Useful for audio processing in Streamlit
```

**`app_streamlit.py`:**

```python
import streamlit as st
import os
from pydub import AudioSegment # To process audio files, convert formats if needed

# Import your ASR and sentiment analysis functions
# from your_asr_module import transcribe_audio_whisper
# from your_sentiment_module import analyze_sentiment

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Your transcribe_audio_whisper and analyze_sentiment functions go here

st.set_page_config(page_title="Voice-to-Text & Emotion Analysis", layout="centered")

st.title("🎙️ Voice-to-Text & Emotion Analysis")

st.markdown("""
This application will convert speech from your audio file into text,
then analyze the emotion within that text.
""")

uploaded_file = st.file_uploader("Upload an audio file (mp3, wav, m4a, etc.)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # Save the file temporarily
    temp_audio_path = os.path.join("temp_audio", uploaded_file.name)
    os.makedirs(os.path.dirname(temp_audio_path), exist_ok=True)
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file, format=uploaded_file.type)

    if st.button("Analyze"):
        with st.spinner("Converting speech and analyzing emotion..."):
            # Convert to suitable format if needed (Whisper generally supports well)
            # If format errors occur, you can use pydub to convert
            # if not uploaded_file.name.lower().endswith(('.mp3', '.wav')):
            #     audio = AudioSegment.from_file(temp_audio_path)
            #     temp_audio_path = temp_audio_path.rsplit('.', 1)[0] + '.mp3'
            #     audio.export(temp_audio_path, format="mp3")

            # Step 1: Speech-to-Text conversion
            transcribed_text = transcribe_audio_whisper(temp_audio_path)

            if transcribed_text:
                st.subheader("Transcribed Text:")
                st.success(transcribed_text)

                # Step 2: Emotion Analysis
                sentiment_result = analyze_sentiment(transcribed_text)
                if sentiment_result:
                    st.subheader("Emotion Analysis Result:")
                    st.write(f"**Emotion:** {sentiment_result['label']}")
                    st.progress(sentiment_result['score'])
                    st.write(f"**Confidence Score:** {sentiment_result['score']:.4f}")
                else:
                    st.warning("Could not analyze sentiment from the text.")
            else:
                st.error("Could not convert speech to text. Please check API key or file format.")

        # Delete the temporary file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

# To run the Streamlit app:
# streamlit run app_streamlit.py
```

-----

## 💪 Project Strengths

  * **Audio Processing Skills**: Using libraries to process and send audio files.
  * **Natural Language Processing (NLP)**: Applying AI models to understand and analyze text (emotions).
  * **AI API Integration**: Working with external AI services like OpenAI Whisper.
  * **Multimedia Application**: Combining audio and text to create a useful solution.
  * **Web/Application Development Skills**: If you build a user interface, you'll demonstrate your ability to create a complete application.

This project is an excellent way to showcase your capabilities in AI and software development. If you need to delve deeper into any specific part (e.g., how to record directly from the browser using JavaScript, or deploying to the cloud), let me know\!