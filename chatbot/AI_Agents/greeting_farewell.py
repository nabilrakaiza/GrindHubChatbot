"""
This file is for an AI Agent that handles user greetings and farewells.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Dict, Any
from langchain_core.output_parsers import JsonOutputParser

class GreetingFarewell(): # Renamed the class for clarity
    """
    Main class for the Greeting and Farewell AI Agent.
    Handles user greetings and farewells with appropriate and friendly responses.
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
            temperature=0.7 # Increased temperature slightly for more natural, varied greetings/farewells
        )
        self.parser = JsonOutputParser(pydantic_object=BaseModel) # Parser retained, though not strictly used for simple text output
    
    def process_message(self, user_message, context=None): # Added default None for context
        """
        Process the user message and return an appropriate greeting or farewell response.
        :param user_message: The message from the user.
        :param context: Optional additional context (e.g., current time of day for "Good morning").
        :return: Response from the AI agent (greeting or farewell).
        """

        llm = self.chat_model

        # --- System Prompt for the Greeting/Farewell Agent ---
        system_prompt = f"""
        You are a friendly and polite AI assistant specialized in handling greetings and farewells.
        Your task is to respond appropriately and courteously to user messages that are clearly a greeting or a farewell.
        - If the user's message is a greeting (e.g., "hello", "hi", "good morning", "good afternoon", "good evening", "how are you"), respond with a warm and friendly greeting back.
        - If the user's message is a farewell (e.g., "goodbye", "bye", "see you", "farewell", "talk to you later"), respond with a polite and friendly farewell.
        - Acknowledge the current time of day (morning, afternoon, evening) if the greeting implies it.
        - Keep your responses brief, positive, and natural.
        - Your response should also be easily readable from a mobile device, so keep it concise and to the point.
        """

        user_prompt = f"""
        User: {user_message}
        Please provide an appropriate greeting or farewell response to the user's message.
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        result = llm.invoke(messages)
        text_result = result.content
        print(f"text result: {text_result}")
        return text_result
        
    # The 'extract_user_message' function from the previous context is not needed
    # for a dedicated greeting/farewell agent, as the LLM directly handles the intent.
    # Therefore, it has been removed.