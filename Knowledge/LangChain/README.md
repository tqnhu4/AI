
# LangChain Cheat Sheet

LangChain is a powerful framework that streamlines building applications with large language models (LLMs). It helps you connect LLMs with external data sources and tools, making complex AI applications much simpler to develop.

-----

## 🎯 Key Concepts

  * **Models**:

      * **LLMs**: Core language models for text generation (e.g., OpenAI's GPT models).
      * **Chat Models**: LLMs optimized for conversations (e.g., GPT-3.5-turbo).
      * **Embeddings**: Convert text into numerical vectors, capturing their semantic meaning.

  * **Prompts**:

      * **`PromptTemplate`**: Formats text prompts for LLMs, allowing dynamic variable insertion.
      * **`ChatPromptTemplate`**: Similar to `PromptTemplate` but specifically for chat models, using roles like System, Human, and AI.

  * **Chains**:

      * Sequences of components that run in a defined order. They let you link LLMs with other tools or data.
      * **`LLMChain`**: The most basic chain, taking a prompt and an LLM to get a response.
      * **`RetrievalQA`**: A common chain for Question Answering over a custom knowledge base (often used in RAG).

  * **Retrievers**:

      * Components that fetch relevant documents from a data source (like a vector store) based on a query. Essential for **Retrieval Augmented Generation (RAG)**.

  * **Agents**:

      * LLMs that can dynamically use a set of **Tools** to decide what actions to take based on user input, offering more flexibility than fixed chains.

  * **Document Loaders**:

      * Load data from various sources (PDFs, text files, web pages) into `Document` objects for processing.

  * **Text Splitters**:

      * Break down large `Document` objects into smaller chunks, important for managing LLM token limits and improving retrieval accuracy.

  * **Vector Stores**:

      * Databases that store and index embeddings, enabling fast similarity searches (e.g., FAISS, Chroma).

-----

## 🔁 Common Workflow: Retrieval Augmented Generation (RAG)

RAG is a popular pattern for building LLM applications that can answer questions using specific, custom data rather than just the LLM's pre-trained knowledge.

1.  **Load Data**: Use **`DocumentLoaders`** to ingest your custom data (e.g., PDFs, Markdown files) into LangChain `Document` objects.
2.  **Split Documents**: Employ **`TextSplitters`** to divide large documents into smaller, manageable chunks.
3.  **Create Embeddings**: Use an **`Embeddings`** model to convert each text chunk into a numerical vector.
4.  **Store in Vector Store**: Save these embeddings in a **`VectorStore`** (like FAISS) to create a searchable index.
5.  **Retrieve**: When a user asks a question, a **`Retriever`** finds the most relevant document chunks from your `VectorStore`.
6.  **Generate**: Pass the retrieved chunks (as context) along with the user's question to an **LLM** (often via a `RetrievalQA` chain) to generate an informed answer.

-----

## 🐍 Key Code Snippets

### 1\. Loading Data & Splitting

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Load documents from a file
loader = TextLoader("your_data.txt")
documents = loader.load()

# Split documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
```

### 2\. Creating Embeddings & Vector Store

```python
from langchain_community.embeddings import OpenAIEmbeddings # Or HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS # Or Chroma, Pinecone

# Initialize the embeddings model
embeddings = OpenAIEmbeddings()

# Create a FAISS vector store from your document chunks and embeddings
vectorstore = FAISS.from_documents(chunks, embeddings)

# (Optional) Save and Load your vector store for persistence
# vectorstore.save_local("faiss_index")
# loaded_vectorstore = FAISS.load_local("faiss_index", embeddings)
```

### 3\. Basic LLM Interaction

```python
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize the LLM (e.g., OpenAI's text-davinci-003)
llm = OpenAI(temperature=0.7) # Temperature controls creativity (0.0 = deterministic)

# Define a simple prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short paragraph about {topic}."
)

# Create an LLM Chain to combine the prompt and LLM
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain with your input
response = llm_chain.invoke({"topic": "artificial intelligence"})
print(response["text"])
```

### 4\. RetrievalQA Chain (RAG)

```python
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

# Assuming 'vectorstore' is already created as shown above
retriever = vectorstore.as_retriever() # Get a retriever from your vector store

# Initialize the LLM for Q&A
llm = OpenAI(temperature=0.0) # Often lower temperature for factual answers

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # 'stuff' combines all retrieved docs into one prompt
    retriever=retriever,
    return_source_documents=True # Optionally return the source chunks used
)

# Ask a question to your RAG system
query = "What is Python's design philosophy?"
response = qa_chain.invoke({"query": query})
print(response["result"])
# print(response["source_documents"]) # Uncomment to see the retrieved text
```

### 5\. Chat Models & ChatPromptTemplate

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Initialize a chat model (e.g., GPT-3.5-turbo)
chat_model = ChatOpenAI(temperature=0.7)

# Create a chat prompt template with system and human roles
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{user_input}")
])

# Format messages and invoke the chat model
messages = chat_prompt.format_messages(user_input="Hello, how are you?")
response = chat_model.invoke(messages)
print(response.content)

# For a conversational chain with memory
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

conversation = ConversationChain(
    llm=chat_model,
    memory=ConversationBufferMemory() # Stores chat history
)

print(conversation.invoke({"input": "What is recursion?"})["response"])
print(conversation.invoke({"input": "Give me a Python example of it."})["response"])
```

-----

## 🔗 Useful Links

  * **LangChain Documentation**: The official hub for all things LangChain.
    [liên kết đáng ngờ đã bị xóa]
  * **LangChain API Reference**: Detailed information on all classes and functions.
    [https://api.python.langchain.com/en/latest/](https://api.python.langchain.com/en/latest/)
  * **OpenAI API Keys**: Where to get your API key for OpenAI models.
    [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)

