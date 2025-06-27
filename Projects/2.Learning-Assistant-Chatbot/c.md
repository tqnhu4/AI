
# 🤖 AI-Powered Learning Assistant Chatbot: Core Code Structure
(LangChain + OpenAI API + FAISS (RAG))

This code will demonstrate the main components:

1.  **Data Preparation (Simulated)**
2.  **Embedding Creation**
3.  **FAISS Indexing**
4.  **LangChain RAG Pipeline (Retrieval and Generation)**
5.  **Basic Chat Interface**

**Before you start:**

  * **Install Libraries:**
    ```bash
    pip install langchain openai faiss-cpu sentence-transformers tiktoken python-dotenv
    ```
    *(Note: `faiss-cpu` for CPU-only, use `faiss-gpu` if you have a compatible GPU).*
  * **OpenAI API Key:** Get your API key from [OpenAI](https://platform.openai.com/account/api-keys). Store it securely, preferably in a `.env` file.
  * **Knowledge Base Data:** For a real project, you'd replace the `docs` list with actual loaded data (e.g., from PDF, Markdown, text files).

-----
## 🔁 Overview of steps:

```
[Start]
   ↓
[0. Config & Environment Setup]
   ↓
[1. Data Preparation]
   ↓
[2. Text Splitting & Embedding Creation]
   ↓
[3. FAISS Indexing (Vector Store)]
   ↓
[4. LangChain RAG Pipeline (QA Chain)]
   ↓
[5. Chat Interface with Streamlit]
   ↓
[User Interaction + LLM Response]
   ↓
[End]

```
-----

### 🔄 Project Flow

```mermaid
graph TD
    A[Start Application] --> B{0. Environment Setup};
    B --> C[Load .env & Check OpenAI API Key];
    C -- API Key Missing? --> D{Display Error & Stop};
    C -- API Key OK --> E[1. Data Preparation];
    E --> F[Define/Load Sample Learning Content];
    F --> G[2. Text Splitting];
    G --> H[Split Content into Chunks (CharacterTextSplitter)];
    H --> I[3. Embeddings Creation];
    I --> J[Generate Embeddings for Chunks (OpenAIEmbeddings)];
    J --> K[4. FAISS Indexing (Vector Store)];
    K --> L{Create FAISS Vector Store from Embeddings};
    L -- Success --> M[5. Initialize LLM & Prompt];
    L -- Error --> D;
    M --> N[Initialize OpenAI LLM];
    N --> O[Define Custom Prompt Template];
    O --> P[6. Build LangChain RAG Pipeline];
    P --> Q[Create RetrievalQA Chain (LLM + Retriever + Prompt)];
    Q --> R[7. Initialize Streamlit UI];
    R --> S[Set Page Config & Display Header];
    S --> T[Initialize/Display Chat History];
    T --> U{8. User Input};
    U -- User Enters Query --> V[Display User Message];
    V --> W[Add User Message to Chat History];
    W --> X[9. Process Query];
    X --> Y[Retrieve Relevant Context (Vector Store FAISS)];
    Y --> Z[Generate LLM Response (RAG Chain)];
    Z --> AA[10. Display Response];
    AA --> BB[Display Assistant Response];
    BB --> CC[Add Assistant Message to Chat History];
    CC --> U;
    D[End Application (Error)]
```

## 📁 Project Structure (Recommended)

```
ai_learning_chatbot/
├── .env                  # For storing API keys securely
├── data/
│   └── learning_content.md # Your structured learning material
├── app.py                # Main application logic
├── requirements.txt      # List of dependencies
└── README.md             # Project description and setup instructions
```

-----

## 📄 `requirements.txt`

```
langchain
openai
faiss-cpu
sentence-transformers
tiktoken
python-dotenv
streamlit # For a simple web UI
```

-----

## 🔒 `.env` file

Create this file in the root directory and add your OpenAI API key:

```
OPENAI_API_KEY="your_openai_api_key_here"
```

-----

## 🐍 `app.py` (Main Chatbot Logic)

```python
import os
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st # For a simple UI

# --- 0. Configuration and Environment Setup ---
load_dotenv() # Load environment variables from .env file

# Ensure the OpenAI API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found in .env file. Please add it.")
    st.stop() # Stop execution if key is missing

# --- 1. Simulate Data Acquisition & Preparation ---
# In a real project, you would load data from files (PDFs, Markdown, etc.)
# For this example, we'll use a simple list of strings.
# You'd typically load from 'data/learning_content.md' and process it.

# This structured data could be from a programming tutorial, math concepts, etc.
# Example content for a Python programming assistant:
sample_learning_content = [
    "Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum and first released in 1991, Python's design philosophy emphasizes code readability with its notable use of significant whitespace.",
    "A 'for' loop in Python is used for iterating over a sequence (that is, a list, tuple, dictionary, set, or string). It works like this:\n\nfor item in sequence:\n    # do something with item\n\nExample:\nfruits = ['apple', 'banana', 'cherry']\nfor x in fruits:\n    print(x)",
    "A function in Python is a block of organized, reusable code that is used to perform a single, related action. Functions provide better modularity for your application and a high degree of code reusing. You define a function using the 'def' keyword:\n\ndef my_function(parameter1, parameter2):\n    # do something\n    return result",
    "Recursion in programming is a technique where a function calls itself. This is often used for problems that can be broken down into smaller, self-similar sub-problems. A recursive function must have a base case to stop the recursion. Without a base case, it would lead to infinite recursion.\n\nExample of factorial recursion:\ndef factorial(n):\n    if n == 1:\n        return 1\n    else:\n        return n * factorial(n-1)",
    "What is the Big O notation? Big O notation is a mathematical notation that describes the limiting behavior of a function when the argument tends towards a particular value or infinity. It is a way of classifying algorithms according to how their running time or space requirements grow as the input size grows. For example, O(n) means linear time, O(n^2) means quadratic time.",
    "What is an array in data structures? An array is a collection of items stored at contiguous memory locations. The idea is to store multiple items of the same type together. This makes it easier to calculate the position of each element by simply adding an offset to a base value, i.e., the memory location of the first element.",
    "In mathematics, a derivative is a fundamental tool of calculus that measures how a function changes as its input changes. The derivative of a function y = f(x) with respect to x is denoted as dy/dx or f'(x). For example, the derivative of f(x) = x^2 is f'(x) = 2x. The derivative of sin(x) is cos(x).",
    "Integration is the reverse process of differentiation. It is used to find the area under a curve, volume of solids, and other applications. The integral of a function f(x) is denoted as ∫f(x)dx. For example, the integral of 2x is x^2 + C, where C is the constant of integration. The integral of cos(x)dx is sin(x) + C."
]

# --- 2. Text Splitting & Embedding Creation ---
# CharacterTextSplitter is good for general text. For code, you might use RecursiveCharacterTextSplitter
# or language-specific splitters (e.g., PythonRecursiveTextSplitter in LangChain).
text_splitter = CharacterTextSplitter(
    separator="\n\n", # Split by double newline
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# Split the documents into chunks
docs = []
for content in sample_learning_content:
    docs.extend(text_splitter.create_documents([content]))

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings()

# --- 3. FAISS Indexing (Knowledge Base Creation) ---
# Create a FAISS vector store from the document chunks and their embeddings
# This step creates and stores the numerical representation of your knowledge.
# In a real app, you'd likely save and load this index.
try:
    vectorstore = FAISS.from_documents(docs, embeddings)
    st.success("FAISS Vector Store created successfully!")
except Exception as e:
    st.error(f"Error creating FAISS Vector Store: {e}")
    st.stop()

# --- 4. LangChain RAG Pipeline (Retrieval and Generation) ---
# Define the LLM (OpenAI's gpt-3.5-turbo is a good starting point)
llm = OpenAI(temperature=0.7) # Temperature controls creativity (0.0 = deterministic, 1.0 = very creative)

# Create a custom prompt template for the learning assistant
# This helps guide the LLM's response based on the retrieved context.
template = """You are an AI-powered learning assistant specializing in programming and mathematics.
Use the following context to answer the student's question concisely and clearly.
If the answer is not in the provided context, state that you don't have enough information but try to guide them based on general knowledge if possible.
Provide code examples or mathematical formulas when relevant and make sure they are well-formatted.

Context:
{context}

Student's Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

# Create a RetrievalQA chain.
# This chain takes a question, retrieves relevant documents from the vectorstore,
# then passes them to the LLM along with the question to generate an answer.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # 'stuff' means it will "stuff" all retrieved documents into the prompt
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True # Optionally return the documents that were used
)

# --- 5. Basic Chat Interface (using Streamlit) ---
st.set_page_config(page_title="🤖 AI Learning Assistant")
st.header("🤖 AI Learning Assistant")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me about Python, math, or programming concepts..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        # Get response from the RAG chain
        response = qa_chain.invoke({"query": prompt})
        bot_response = response["result"]

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(bot_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

        # Optionally, display source documents for transparency
        # with st.expander("Source Documents Used"):
        #     for doc in response["source_documents"]:
        #         st.write(doc.page_content)
        #         st.write(f"Source: {doc.metadata.get('source', 'N/A')}")
```

-----

## ▶️ How to Run This Project

1.  **Save the Code:** Save the code above as `app.py`.
2.  **Create `.env`:** Create a file named `.env` in the same directory as `app.py` and paste your `OPENAI_API_KEY="your_api_key_here"` into it.
3.  **Install Dependencies:** Open your terminal or command prompt, navigate to the project directory, and run:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    This will open the chatbot interface in your web browser.

-----

## 💡 Explanation of Key Sections:

  * **`0. Configuration and Environment Setup`**: Loads your OpenAI API key securely using `python-dotenv`.
  * **`1. Simulate Data Acquisition & Preparation`**:
      * `sample_learning_content`: This is your *structured data*. In a real project, you'd load this from actual files (e.g., a folder full of Markdown files for programming concepts, or a database for math formulas).
      * `CharacterTextSplitter`: Breaks down your large text content into smaller, manageable "chunks." This is crucial because LLMs have token limits, and smaller chunks lead to more precise retrieval.
  * **`2. Text Splitting & Embedding Creation`**:
      * `OpenAIEmbeddings()`: This uses OpenAI's API to convert your text chunks into numerical vectors (embeddings). These vectors capture the semantic meaning of the text.
  * **`3. FAISS Indexing (Knowledge Base Creation)`**:
      * `FAISS.from_documents()`: This takes your text chunks and their embeddings and builds a searchable index using FAISS. When a user asks a question, FAISS will quickly find the most semantically similar chunks from this index.
  * **`4. LangChain RAG Pipeline`**:
      * `OpenAI(temperature=0.7)`: Initializes the Large Language Model. `temperature` controls how creative/random the output is.
      * `PromptTemplate`: This defines the instructions for the LLM. It includes placeholders for the `context` (retrieved by FAISS) and the `question` (user's input). This is key to making the LLM use your specific data.
      * `RetrievalQA.from_chain_type`: This is the core RAG chain from LangChain.
          * `retriever=vectorstore.as_retriever()`: Tells the chain to use your FAISS vector store to find relevant documents.
          * `chain_type="stuff"`: A common method where all retrieved documents are "stuffed" directly into the LLM's prompt.
          * `chain_type_kwargs={"prompt": PROMPT}`: Applies your custom prompt template.
  * **`5. Basic Chat Interface (Streamlit)`**:
      * **Streamlit** is used for quickly creating a simple web-based chat interface. It manages the input, output, and displays the conversation history.

This code provides a functional prototype. For a production system, you'd expand on error handling, persistent storage of the FAISS index, more sophisticated data loading, and potentially user authentication/management.