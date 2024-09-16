# gemini-quizify
Gemini Quiz Builder is an interactive application that generates quizzes based on a given PDF document. Users can specify the number of questions and the topic, and the program will create a quiz tailored to those requirements. This project leverages various advanced technologies, including Google Gemini, Vertex AI API, embeddings, Google Service Account, Langchain, PDF loader, and Streamlit. It was developed as part of the challenges provided by Radical AI. The project aims to simplify the quiz creation process by automating it through AI and machine learning. Users can quickly generate quizzes for educational or training purposes by providing a PDF and selecting the quiz parameters.

# Installation:

### 1. With Docker (Suggested)
##### 1.1. Clone the repository:
```bash
git clone https://github.com/yourusername/gemini-quiz-builder.git
cd gemini-quiz-builder
```

##### 1.2. Change your Google Service Account Key File and Adjust the Environment Variables
1.2.1. Change the service_account_key.json with your Google Service Account Key's key file.
1.2.2. Change the environment variables in `.env` file with your information.

##### 1.3. Build the Container with Your Credentials
```bash
docker build --tag gemini-quizify .
```

##### 1.4. Run the Container
```bash
docker run --rm --name gemini-quizify --env-file .env --publish 8501:8501 gemini-quizify
```

### 2. Without Docker

##### 2.1. Clone the repository:
```bash
git clone https://github.com/yourusername/gemini-quiz-builder.git
cd gemini-quiz-builder
```

##### 2.2. Create and activate a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

##### 2.3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

##### 2.4. Set up Google Service Account:

Obtain the service account JSON file from your Google Cloud Console.
Place the JSON file in the project directory and set the GOOGLE_APPLICATION_CREDENTIALS environment variable to its path:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service_account_key.json"
```

##### 2.5. Run the application:
```bash
streamlit run main.py
```

# Usage
**Upload PDF:** Open the application and upload the PDF file from which you want to generate the quiz.  
**Select Parameters:** Choose the number of questions and the specific topic for the quiz.  
**Generate Quiz:** Click on the "Generate" button, and the program will create a quiz based on the provided inputs.  
**Review and Use:** Review the generated quiz questions and use them for your desired purpose.

# Acknowledgements
This project is part of the challenges provided by Radical AI. Special thanks to all contributors who have helped implement various steps in this project.
