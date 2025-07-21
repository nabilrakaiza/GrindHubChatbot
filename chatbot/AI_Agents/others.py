"""
This file is for an AI Agent that handles miscellaneous queries not covered by other AI agents.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Dict, Any
from langchain_core.output_parsers import JsonOutputParser
import json
import requests # This might be removed if the new agent doesn't need external data

class Others():
    """
    Main class for the Others Query AI Agent.
    Handles miscellaneous queries that don't fit into other specialized agents.
    """

    def __init__(self, userid, api_key: str):
        """
        Initialize the agent with the API key.
        :param api_key: API key for Google Generative AI.
        """
        self.userid = userid # userid might not be strictly necessary for a general agent, but keeping it for consistency
        self.api_key = api_key
        self.chat_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.api_key,
            temperature=0.2 # You might adjust this temperature based on desired response creativity
        )
        self.parser = JsonOutputParser(pydantic_object=BaseModel) # Parser might not be strictly needed unless you expect structured output

    def process_message(self, user_message, context):
        """
        Process the user message and return a response for general queries.
        :param user_message: The message from the user.
        :param context: Additional context for the query, if available.
        :return: Response from the AI agent.
        """

        llm = self.chat_model

        # --- System Prompt for the "Others Query" Agent ---
        system_prompt = """
        You are a versatile AI assistant designed to handle a wide range of queries that are not addressed by other specialized AI agents.
        Your goal is to provide helpful, informative, and concise responses.
        If you are unsure or the query is outside your capabilities, you should politely state that you cannot assist with that specific request.
        Keep your responses clear and easy to understand.
        """

        user_prompt = f"""
        User: {user_message}
        Context: {context if context else "No additional context provided."}
        Please provide a helpful response to the user's message.
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        result = llm.invoke(messages)
        text_result = result.content
        print(f"text result: {text_result}")
        return text_result
