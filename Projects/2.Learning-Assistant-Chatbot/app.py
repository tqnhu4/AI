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