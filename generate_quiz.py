from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate
from chroma_collection_creator import ChromaCollectionCreator
from embedding_client import EmbeddingClient
from document_processor import DocumentProcessor
import streamlit as st
import os
import sys
import json
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
        try:
            self.llm = VertexAI(
                model_name="gemini-pro",
                temperature=0.8,  # Increased for less deterministic questions
                max_output_tokens=500
            )
        except Exception as e:
            print("Failed to initialize VertexAI:", e)

    def generate_question_with_vectorstore(self):
        """
        Generates a quiz question based on the topic provided using a vectorstore

        :return: A JSON object representing the generated quiz question.
        """
        self.init_llm()  # Assuming init_llm() initializes self.llm properly
        if not self.llm:
            raise Exception("Failed to initialize llm")
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
        if not setup_and_retrieval:
            raise Exception("Failed to initialize the setup_and_retrieval")

        # Create a chain with the Retriever, PromptTemplate, and LLM
        chain = setup_and_retrieval | prompt | self.llm
        if not chain:
            raise Exception("Failed to initialize the chain")

        # Invoke the chain with the topic as input
        response = chain.invoke(self.topic)
        return response

    def generate_quiz(self) -> list:
        """
        Task: Generate a list of unique quiz questions based on the specified topic and number of questions.

        This method orchestrates the quiz generation process by utilizing the `generate_question_with_vectorstore` method to generate each question and the `validate_question` method to ensure its uniqueness before adding it to the quiz.

        Steps:
            1. Initialize an empty list to store the unique quiz questions.
            2. Loop through the desired number of questions (`num_questions`), generating each question via `generate_question_with_vectorstore`.
            3. For each generated question, validate its uniqueness using `validate_question`.
            4. If the question is unique, add it to the quiz; if not, attempt to generate a new question (consider implementing a retry limit).
            5. Return the compiled list of unique quiz questions.

        Returns:
        - A list of dictionaries, where each dictionary represents a unique quiz question generated based on the topic.

        Note: This method relies on `generate_question_with_vectorstore` for question generation and `validate_question` for ensuring question uniqueness. Ensure `question_bank` is properly initialized and managed.
        """
        # self.question_bank = [] # Reset the question bank

        for _ in range(self.num_questions):
            # Try maximum 10 times when the json string could not be converted to a dictionary.
            for _ in range(0, 10):
                # Use class method to generate question
                question_str = self.generate_question_with_vectorstore()

                try:
                    # Convert the JSON String to a dictionary
                    question = json.loads(question_str)

                except json.JSONDecodeError:
                    print("Failed to decode question JSON.")
                    continue  # Skip this iteration if JSON decoding fails

                # Validate the question using the validate_question method
                if question and self.validate_question(question):
                    print("Successfully generated unique question")
                    # Add the valid and unique question to the bank
                    self.question_bank.append(question)
                    break
                else:
                    print("Duplicate or invalid question detected.")
                    continue
        return self.question_bank

    def validate_question(self, question: dict) -> bool:
        """
        Task: Validate a quiz question for uniqueness within the generated quiz.

        This method checks if the provided question (as a dictionary) is unique based on its text content compared to previously generated questions stored in `question_bank`. The goal is to ensure that no duplicate questions are added to the quiz.

        Steps:
            1. Extract the question text from the provided dictionary.
            2. Iterate over the existing questions in `question_bank` and compare their texts to the current question's text.
            3. If a duplicate is found, return False to indicate the question is not unique.
            4. If no duplicates are found, return True, indicating the question is unique and can be added to the quiz.

        Parameters:
        - question: A dictionary representing the generated quiz question, expected to contain at least a "question" key.

        Returns:
        - A boolean value: True if the question is unique, False otherwise.

        Note: This method assumes `question` is a valid dictionary and `question_bank` has been properly initialized.
        """

        is_unique = True
        question_text = question['question']
        if not question_text:
            is_unique = False
        else:
            for dictionary in self.question_bank:
                if dictionary['question'] == question_text:
                    is_unique = False
        # is_unique = any(question in self.question_bank for item in self.question_bank)
        return is_unique
