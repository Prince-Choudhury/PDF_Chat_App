# PDF Chat App

Welcome to the PDF Chat App! This web application allows you to chat with a chatbot powered by the OpenAI language model. It is designed to extract text from uploaded PDF files and answer your questions based on the content of the PDF.

## Getting Started

Follow these steps to set up the PDF Chat App on your local machine:

### Prerequisites

1. Python: Make sure you have Python installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

### Installation

1. Clone this repository to your local machine.

   ```bash
   git clone https://github.com/yourusername/pdf-chat-app.git

2. Create a virtual environment (optional but recommended) for the project. You can use venv or virtualenv.
   
   ```bash
   python -m venv venv
   
3. Activate the virtual environment:

   ```bash
   venv\Scripts\activate

4. Install the required Python libraries from the requirements.txt file.

   ```bash
   pip install -r requirements.txt
   
5. Create a .env file in the project directory to store your OpenAI API key.
   OPENAI_API_KEY=your-api-key-here
   
6. Run the PDF Chat App using Streamlit.
   
    ```bash
   streamlit run app.py
    
7. The app should now be running locally. You can access it in your web browser at http://localhost:XXXX.

### Usage

   1. Upload a PDF file by clicking the "Upload your PDF" button.
   2. Wait for the app to process the PDF and create a searchable index.
   3. Enter your questions about the PDF in the text input field.
   4. The app will provide answers based on the content of the PDF.

### Contributing

If you'd like to contribute to this project, please open an issue and discuss your ideas or create a pull request with your proposed changes.

 
