from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from chroma_collection_creator import ChromaCollectionCreator
from embedding_client import EmbeddingClient
from document_processor import DocumentProcessor
import os
import sys
import json
import streamlit as st
from dotenv import load_dotenv
sys.path.append(os.path.abspath('../../'))

load_dotenv()


class QuizGenerator:
    def __init__(self, topic=None, num_questions=1, vectorstore=None):
        """
        Initializes the QuizGenerator with a required topic, the number of questions for the quiz,
        and an optional vectorstore for querying related information.

        :param topic: A string representing the required topic of the quiz.
        :param num_questions: An integer representing the number of questions to generate for the quiz, up to a maximum of 10.
        :param vectorstore: An optional vectorstore instance (e.g., ChromaDB) to be used for querying information related to the quiz topic.
        """
        if not topic:
            self.topic = "General Knowledge"
        else:
            self.topic = topic

        if num_questions > 10:
            raise ValueError("Number of questions cannot exceed 10.")
        self.num_questions = num_questions

        self.vectorstore = vectorstore
        self.llm = None
        self.question_bank = []  # Initialize the question bank to store questions
        self.system_template = """
            You are a subject matter expert on the topic: {topic}
            
            Follow the instructions to create a quiz question:
            1. Generate a question based on the topic provided and context as key "question"
            2. Provide 4 multiple choice answers to the question as a list of key-value pairs "choices"
            3. Provide the correct answer for the question from the list of answers as key "answer"
            4. Provide an explanation as to why the answer is correct as key "explanation"
            
            You must respond as a JSON object with the following structure:
            {{
                "question": "<question>",
                "choices": [
                    {{"key": "A", "value": "<choice>"}},
                    {{"key": "B", "value": "<choice>"}},
                    {{"key": "C", "value": "<choice>"}},
                    {{"key": "D", "value": "<choice>"}}
                ],
                "answer": "<answer key from choices list>",
                "explanation": "<explanation as to why the answer is correct>"
            }}
            
            Context: {context}
            """

    def init_llm(self):
        """
        Initializes and configures the Large Language Model (LLM) for generating quiz questions.

        This method should handle any setup required to interact with the LLM, including authentication,
        setting up any necessary parameters, or selecting a specific model.

        :return: An instance or configuration for the LLM.
        """
        self.llm = VertexAI(
            model_name="gemini-pro",
            temperature=0.8,  # Increased for less deterministic questions
            max_output_tokens=500
        )

    def generate_question_with_vectorstore(self):
        """
        Generates a quiz question based on the topic provided using a vectorstore

        :return: A JSON object representing the generated quiz question.
        """
        if not self.llm:
            self.init_llm()
        if not self.vectorstore:
            raise ValueError("Vectorstore not provided.")

        # Enable a Retriever
        retriever = self.vectorstore.db.as_retriever()

        # Use the system template to create a PromptTemplate
        prompt = PromptTemplate.from_template(self.system_template)

        # RunnableParallel allows Retriever to get relevant documents
        # RunnablePassthrough allows chain.invoke to send self.topic to LLM
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "topic": RunnablePassthrough()}
        )
        # Create a chain with the Retriever, PromptTemplate, and LLM
        chain = setup_and_retrieval | prompt | self.llm

        # Invoke the chain with the topic as input
        response = chain.invoke(self.topic)
        return response


# Test Generating the Quiz
if __name__ == "__main__":

    embed_config = {
        "model_name": os.getenv('MODEL_NAME'),
        "project": os.getenv('GOOGLE_PROJECT_ID'),
        "location": os.getenv('LOCATION')
    }

    screen = st.empty()
    with screen.container():
        st.header("Quiz Builder")
        processor = DocumentProcessor()
        processor.ingest_documents()

        embed_client = EmbeddingClient(**embed_config)

        chroma_creator = ChromaCollectionCreator(processor, embed_client)

        question = None

        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write(
                "Select PDFs for Ingestion, the topic for the quiz, and click Generate!")

            topic_input = st.text_input(
                "Topic for Generative Quiz", placeholder="Enter the topic of the document")
            questions = st.slider("Number of Questions",
                                  min_value=1, max_value=10, value=1)

            submitted = st.form_submit_button("Submit")
            if submitted:
                chroma_creator.create_chroma_collection()

                st.write(topic_input)

                # Test the Quiz Generator
                generator = QuizGenerator(
                    topic_input, questions, chroma_creator)
                question = generator.generate_question_with_vectorstore()

    if question:
        screen.empty()
        with st.container():
            st.header("Generated Quiz Question: ")
            st.write(question)
