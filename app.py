!pip install gradio langchain
!pip install langchain_groq langchain_core langchain_community
!pip install pypdf
!pip install chromadb
!pip install sentence_transformers
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import gradio as gr

# Initialize OpenAI LLM
def initialize_llm():
    llm = ChatGroq(
        temperature=0,
        groq_api_key="-----------PLEASE PASTE YOUR GROQ API KEY -------------------", # ( CREATE A GROQ ACCOUNT AND CREATE AN API KEY AND PASTE IT IN THE VARIABLE   -> Link  : https://console.groq.com) 
        model_name="llama-3.3-70b-versatile"
    )
    return llm

# Create vector database
def create_vector_db():
    loader = DirectoryLoader("/content/data", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory="/content/chroma_db")
    vector_db.persist()
    return vector_db

# Setup QA chain
def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_templates = """You are a compassionate mental health chatbot. Respond thoughtfully:
        {context}
        User: {question}
        Chatbot: """
    PROMPT = PromptTemplate(template=prompt_templates, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# Initialize
llm = initialize_llm()
if not os.path.exists("/content/chroma_db"):
    vector_db = create_vector_db()
else:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="/content/chroma_db", embedding_function=embeddings)
qa_chain = setup_qa_chain(vector_db, llm)

with gr.Blocks() as app:
    gr.Markdown("# 🧠 EmotiBot🤖")
    chatbot = gr.Chatbot()  # Gradio Chatbot component
    input_box = gr.Textbox(placeholder="Type your message here...")
    submit_button = gr.Button("Send")

    # Chatbot response function
    def chatbot_response(user_input, history=[]):
        if not user_input.strip():
            return "Please provide a valid input", history
        response = qa_chain.run(user_input)
        history.append([user_input, response])  # Append as a list, not a tuple
        return history, history

    # Link input and output to the button
    submit_button.click(
        chatbot_response,  # Function to call
        inputs=[input_box, chatbot],  # Input components
        outputs=[chatbot, chatbot],  # Output components
    )

app.launch()
