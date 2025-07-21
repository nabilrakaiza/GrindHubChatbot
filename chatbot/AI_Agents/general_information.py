"""
This file is for an AI Agent that provides general information to users.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Dict, Any
from langchain_core.output_parsers import JsonOutputParser

class GeneralInformation(): # Renamed the class for clarity
    """
    Main class for the General Information AI Agent.
    Provides factual and concise answers to a wide range of general queries.
    """

    def __init__(self, api_key: str):
        """
        Initialize the agent with the API key.
        :param api_key: API key for Google Generative AI.
        """
        self.api_key = api_key
        self.chat_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.api_key,
            temperature=0.2 # Keeping temperature low for factual accuracy
        )
        self.parser = JsonOutputParser(pydantic_object=BaseModel) # Parser retained, though not strictly used for simple text output
    
    def process_message(self, user_message, context=None): # Added default None for context
        """
        Process the user message and return a general information response.
        :param user_message: The message from the user.
        :param context: Optional additional context for the query.
        :return: Response from the AI agent providing general information.
        """

        llm = self.chat_model

        # --- System Prompt for the General Information Agent ---
        system_prompt = """
        You are a highly knowledgeable and helpful AI assistant.
        Your primary role is to provide factual, accurate, and concise general information on a wide variety of topics.
        Answer questions directly and clearly. If a question is outside your knowledge base, or requires real-time, highly personal, or subjective opinions, please politely state that you cannot provide that specific information.
        Your responses should be easily readable from a mobile device, so keep them concise and to the point.
        """

        user_prompt = f"""
        User: {user_message}
        {f"Context: {context}" if context else ""}
        Please provide a general information response to the user's query.
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        result = llm.invoke(messages)
        text_result = result.content
        print(f"text result: {text_result}")
        return text_result
    