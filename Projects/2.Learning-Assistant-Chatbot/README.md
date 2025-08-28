
## 📚 Study Assistant Chatbot: Overview

This chatbot will help students learn a specific skill (e.g., Python programming, high school math) by answering their questions based on a provided set of structured learning materials.

**Core Idea:**
When a user asks a question, the chatbot will:

1.  **Retrieve**: Find the most relevant snippets of information from the given library of learning materials.
2.  **Generate**: Use the retrieved information and a large language model (LLM) to create a natural and accurate answer.

This is the **RAG (Retrieval-Augmented Generation)** model.

-----

## ⚙️ Technologies Used

  * **LangChain**: A powerful framework that helps build LLM-powered applications. It connects various components of the RAG pipeline (LLM integration, vector store management, chain creation).
  * **OpenAI API**: Provides the large language model (LLM) to understand questions and generate answers (e.g., GPT-3.5 Turbo, GPT-4). You'll need an **API Key** from OpenAI.
  * **Vector Search (FAISS)**: A library for efficient similarity search among a large collection of vectors. We'll use it to store and query the **embeddings** of the learning materials.
  * **Text Splitters**: To break down large documents into smaller chunks before creating embeddings.
  * **OpenAI Embeddings**: To convert text into numerical vectors (embeddings) that FAISS can search.

-----

## 🛠️ Implementation Steps & Core Code

### Step 1: Environment and Data Preparation

1.  **Install Libraries:**

    ```bash
    pip install langchain openai faiss-cpu tiktoken
    ```

      * `langchain`: The core framework.
      * `openai`: Python library to interact with the OpenAI API.
      * `faiss-cpu`: Vector search library. Use `faiss-gpu` if you have a GPU for acceleration.
      * `tiktoken`: Used by OpenAI to calculate tokens.

2.  **Set Up OpenAI API Key:**
    Make sure you have your OpenAI API Key. You can set it as an environment variable (recommended) or pass it directly in the code (not recommended for production).

    ```python
    import os
    os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" # Replace with your actual API Key
    ```

3.  **Structured Learning Data:**
    This is a strong point of this application. You'll need real learning materials.

      * **Example:** You could have `.txt`, `.pdf`, `.md` files containing Python programming knowledge.
      * Let's assume we have a `python_basics.txt` file with the following content (place it in a `data/` directory):
        ```
        # data/python_basics.txt
        Python is a high-level, general-purpose programming language created by Guido van Rossum. It was first released in 1991.
        Python is known for its clear and readable syntax. It is widely used in web development, data science, artificial intelligence, and automation.

        Variables in Python:
        Variables are used to store data. Explicit data type declaration is not required.
        Example:
        age = 30
        name = "Alice"
        is_student = True

        If-else statements:
        Used to execute conditional code blocks.
        if condition:
            # code block if condition is True
        else:
            # code block if condition is False

        For loops:
        Used to iterate over sequences (list, tuple, string) or other iterable objects.
        fruits = ["apple", "banana", "cherry"]
        for x in fruits:
            print(x)
        ```
      * Create a `data/` directory and place `python_basics.txt` inside it.

### Step 2: Prepare Data for the Vector Store (RAG Pipeline)

We need to load the data, split it into chunks, and create embeddings.

```python
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# 1. Load Data
# Assuming your data is in the 'data/' directory
# You can use DirectoryLoader to load multiple files at once
loader = TextLoader("data/python_basics.txt", encoding="utf-8")
documents = loader.load()

# 2. Split Documents
# Split large documents into smaller chunks for efficient searching
text_splitter = CharacterTextSplitter(
    chunk_size=1000, # Size of each chunk
    chunk_overlap=200 # Overlap between chunks to maintain context
)
docs = text_splitter.split_documents(documents)

# 3. Create Embeddings and Vector Store
# Use OpenAIEmbeddings to convert text into numerical vectors
embeddings = OpenAIEmbeddings()

# Create a FAISS vector store from the document chunks and embeddings
# This is where the document vectors are stored and indexed for search
db = FAISS.from_documents(docs, embeddings)

print(f"Vector Store created with {len(docs)} document chunks.")
```

### Step 3: Build the Chatbot (RAG Chain)

Now we'll combine the LLM with the Vector Store to create the RAG Chain.

```python
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

# 1. Initialize LLM
# We'll use a model from OpenAI
# Lower temperature means less "creative" and more data-focused answers
llm = OpenAI(temperature=0)

# 2. Create RetrievalQA Chain
# This chain will take a question, search in the db, and then pass the results to the LLM to generate an answer
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # "stuff" means stuffing all relevant chunks into the LLM's prompt
    retriever=db.as_retriever(), # Convert the FAISS db into a retriever
    return_source_documents=True # To see the source documents used by the LLM
)

print("Chatbot is ready!")
```

### Step 4: Interact with the Chatbot

Now you can ask the chatbot questions and get answers based on the data you've provided.

```python
def ask_chatbot(question):
    """
    Asks the chatbot a question and prints the answer along with source documents.
    """
    print(f"\nYour Question: {question}")
    result = qa_chain({"query": question})
    print(f"Chatbot Answer: {result['result']}")
    print("\n--- Source Documents Used ---")
    for doc in result['source_documents']:
        print(f"  - {doc.metadata.get('source', 'Unknown source')}: \"{doc.page_content[:150]}...\"")
    print("-----------------------------------")

# --- Example Questions ---
ask_chatbot("What is Python?")
ask_chatbot("How do variables work in Python?")
ask_chatbot("How to use a for loop in Python?")
ask_chatbot("Do I need to declare data types for variables in Python?")
ask_chatbot("Who created Python?")
ask_chatbot("I want to learn about lambda functions in Python.") # Out-of-scope question
```

-----

## 💪 Project Strengths

  * **Integrating AI with Real-world Data (RAG)**: This is a powerful and crucial technique in modern AI. It allows LLMs to answer specific questions, avoid "hallucinations," and rely on trustworthy information.
  * **NLP & Vector Search Skills**: Shows your ability to process text, create embeddings, and use vector databases for semantic search.
  * **Using an AI Framework (LangChain)**: Demonstrates your capability to build complex AI pipelines in a structured and efficient manner.
  * **Product Thinking**: This project solves a real-world problem (study assistance) and has significant potential for expansion. You can discuss how you would:
      * Expand data sources (add PDFs, web pages, etc.).
      * Add other skills (e.g., code debugging, explaining math concepts).
      * Build a user interface (web app with Flask/Streamlit, mobile app).
      * Integrate user feedback to improve answer quality.

