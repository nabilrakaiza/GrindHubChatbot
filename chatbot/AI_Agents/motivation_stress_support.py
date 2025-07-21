"""
This file is for AI Agent that provides motivation and stress support to users.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Dict, Any
from langchain_core.output_parsers import JsonOutputParser
import json
import requests

class MotivationStressSupport():
    """
    Main class for the Motivation and Stress Support AI Agent.
    """

    def __init__(self, userid, api_key: str):
        """
        Initialize the agent with the API key.
        :param api_key: API key for Google Generative AI.
        """
        self.userid = userid
        self.api_key = api_key
        self.chat_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.api_key,
            temperature=0.2
        )
        self.parser = JsonOutputParser(pydantic_object=BaseModel)
    
    def process_message(self, user_message, context):
        """
        Process the user message and return a response.
        :param user_message: The message from the user.
        :return: Response from the AI agent.
        """

        llm = self.chat_model

        system_prompt = """
        You are an AI assistant that provides motivation and stress support to users.
        Users are most likely stressed due to academic pressures, personal issues, or general life challenges.
        You should respond with empathetic and supportive messages that help alleviate stress and provide motivation.
        Your responses should be encouraging and uplifting, focusing on positive reinforcement and practical advice.
        Your response should also be easily readable from a mobile device, so keep it concise and to the point.
        """

        user_prompt = f"""
        User: {user_message}
        Context: {context}
        Please provide a motivational and supportive response to the user's message.
        """

        # formatted_prompt = prompt_template.format(user_message=user_message)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        result = llm.invoke(messages)
        text_result = result.content
        print(f"text result: {text_result}")
        return text_result
    
    def get_data_from_database(self, time_range: str, class_name: str):
        """
        Based on user message, get some data from database to add more context. 
        Input : time range, userid
        Output : data from database
        """
        try:
            url = "https://grindhub-production.up.railway.app/api/auth/getPerformanceAssignmentQuery"
            headers = {'Content-Type': 'application/json'}
            payload = {'userid': self.userid, "time_range":time_range, "class":class_name}  # Replace with actual user ID

            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()

            if data.get('success'):
                print("Data fetched successfully:", data)
            else:
                print("Failed to fetch data:", data.get('message', 'No message provided.'))
        
        except requests.exceptions.RequestException as e:
            print(f"Error during API request: {e}")
            return {"success": False, "message": f"Request error: {e}"}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            return {"success": False, "message": f"JSON parsing error: {e}"}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"success": False, "message": f"An unexpected error occurred: {e}"}
        