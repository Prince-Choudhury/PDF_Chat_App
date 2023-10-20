# Import necessary libraries
import streamlit as st  # Streamlit library for creating web apps
from dotenv import load_dotenv  # For loading environment variables from a .env file
import pickle  # For serializing and deserializing Python objects
from PyPDF2 import PdfReader  # For reading PDF files
from streamlit_extras.add_vertical_space import add_vertical_space  # A Streamlit extension for adding vertical space
from langchain.text_splitter import RecursiveCharacterTextSplitter  # LangChain library for text splitting
from langchain.embeddings.openai import OpenAIEmbeddings  # LangChain library for OpenAI text embeddings
from langchain.vectorstores import FAISS  # LangChain library for vector stores
from langchain.llms import OpenAI  # LangChain library for OpenAI models
from langchain.chains.question_answering import load_qa_chain  # LangChain library for question-answering chains
from langchain.callbacks import get_openai_callback  # LangChain callback for OpenAI models
import os  # Python's OS module for working with the file system


# Sidebar contents
with st.sidebar:
    st.title('🤖 PDF Chat App')  # Sidebar title
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/) - Streamlit library for creating web apps.
    - [LangChain](https://python.langchain.com/) - LangChain provides natural language processing capabilities.
    - [OpenAI](https://platform.openai.com/docs/models) LLM model.
    ''')
    add_vertical_space(5)  # Adds vertical space to the sidebar
    st.write('Crafted with ✨ by [Prince Choudhury](https://www.linkedin.com/in/prince-choudhury26/)') 
 
# Load environment variables
load_dotenv()

# Define the main function for the Streamlit app
def main():
    st.header("Chat with PDF 📚")  # Main app header

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)  # Read the uploaded PDF file
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()  # Extract text from PDF pages

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Split text into chunks of 1000 characters
            chunk_overlap=200,  # Overlap between chunks
            length_function=len  # Function to calculate text length
        )
        chunks = text_splitter.split_text(text=text)  # Split the text into manageable chunks

        store_name = pdf.name[:-4]  # Remove the ".pdf" extension from the PDF file name
        st.write(f'{store_name}')   # Display the PDF file name without the extension


        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)  # Load vector store from a saved file
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)  # Create a vector store from text chunks
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)  # Serialize and save the vector store

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")  # Input field for user's query

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)  # Search for similar documents based on the query
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")  # Load a question-answering chain
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)  # Execute the question-answering chain
                print(cb) # Print the callback information
            st.write(response)  # Display the response

if __name__ == '__main__':
    main()  # Run the Streamlit app
