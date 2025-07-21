"""
This file is for an AI Agent that provides assignment and class performance data to users.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from langchain_core.output_parsers import JsonOutputParser
import requests
import json

class ExtractedPerformanceQueryComponent(BaseModel): # Renamed for clarity
    """
    Represents extracted components from the user's query about performance.
    """
    time_range: Optional[str] = Field(None, description="Time range for the performance query, e.g., 'today', 'this week', 'last month'.")
    module_class: Optional[str] = Field(None, description="Specific module or class name, e.g., 'Math', 'History 101'.")
    assignment_type: Optional[str] = Field(None, description="Type of assignment, e.g., 'essay', 'project', 'exam', 'quiz'.")

class PerformanceAssignmentQuery(): # Renamed the class for clarity
    """
    Main class for the Assignment Performance Query AI Agent.
    Fetches and summarizes user's assignment scores, class performance, and study statistics.
    """

    def __init__(self, userid: str, api_key: str): # Added userid to init as it's used in database calls
        """
        Initialize the agent with the user ID and API key.
        :param userid: The ID of the current user.
        :param api_key: API key for Google Generative AI.
        """
        self.userid = userid # Store userid for database calls
        self.api_key = api_key
        self.chat_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.api_key,
            temperature=0.2 # Low temperature for factual, consistent responses
        )
        self.parser = JsonOutputParser(pydantic_object=ExtractedPerformanceQueryComponent) # Parser for structured extraction
    
    def process_message(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process the user message to understand the performance query and return a summary.
        This method orchestrates data extraction and response generation.

        :param user_message: The message from the user, e.g., "How did I do on my last math assignment?"
        :param context: Additional context (e.g., previous turns in conversation).
        :return: A summary of the user's performance metrics.
        """
        llm = self.chat_model

        # Step 1: Extract relevant parameters from the user message
        # extracted_params = self.extract_user_message(user_message)
        # time_range = extracted_params.get("time_range")
        # module_class = extracted_params.get("module_class")
        # assignment_type = extracted_params.get("assignment_type")

        # # Step 2: Fetch data from the database using extracted parameters
        # performance_data = self.get_data_from_database(
        #     time_range=time_range,
        #     class_name=module_class # Using module_class for the database function's class_name parameter
        # )

        # Append fetched data to context for the LLM
        # if context is None:
        #     context = {}
        # context['performance_data'] = performance_data # Add fetched data to context

        system_prompt = """
        You are an AI assistant specialized in providing users with their academic performance data.
        You can access study statistics, assignment scores, and class performance from the user's records.
        Your task is to calculate and summarize performance metrics based on the user's query and the provided data.
        When providing the summary:
        - Be concise and to the point.
        - Highlight key achievements or areas for improvement.
        - If specific data is requested (e.g., "my score on the last Physics quiz"), provide it clearly.
        - If data is not available for a requested period or class, politely inform the user.
        - Your response should be easily readable from a mobile device.
        """

        user_prompt = f"""
        User query: "{user_message}"
        
        Please provide a concise summary of the user's academic performance based on the query and the provided data.
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        result = llm.invoke(messages)
        text_result = result.content
        print(f"Generated response: {text_result}")
        return text_result
        
    def extract_user_message(self, user_message: str) -> ExtractedPerformanceQueryComponent:
        """
        Extracts relevant components (time range, module/class, assignment type)
        from the user's message using the LLM.

        :param user_message: The message from the user.
        :return: An instance of ExtractedPerformanceQueryComponent containing the extracted details.
        """
        extraction_prompt = f"""
        Extract the following information from the user's message:
        - time_range (e.g., 'today', 'this week', 'last month', 'all time')
        - module_class (e.g., 'Math', 'History 101', 'Physics')
        - assignment_type (e.g., 'essay', 'project', 'exam', 'quiz')

        If a piece of information is not explicitly mentioned, leave it as null.
        
        User message: "{user_message}"

        Return the information as a JSON object adhering to the following schema:
        {{"time_range": "string or null", "module_class": "string or null", "assignment_type": "string or null"}}
        """

        messages = [
            SystemMessage(content="You are an expert at extracting specific data points from user queries."),
            HumanMessage(content=extraction_prompt)
        ]

        # Use the LLM to perform the extraction
        extraction_result = self.chat_model.invoke(messages)
        
        try:
            # Parse the JSON string from the LLM's response
            extracted_data = json.loads(extraction_result.content)
            # Validate with Pydantic model
            return self.parser.parse(json.dumps(extracted_data)) # Ensure it's a JSON string
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing extracted message components: {e}")
            return ExtractedPerformanceQueryComponent(time_range=None, module_class=None, assignment_type=None)


    def get_data_from_database(self, time_range: Optional[str] = None, class_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetches assignment and performance data from the database.
        
        :param time_range: The desired time range for the data (e.g., 'today', 'this week').
        :param class_name: The name of the class or module.
        :return: A dictionary containing the fetched data or an error message.
        """
        try:
            url = "https://grindhub-production.up.railway.app/api/auth/getPerformanceAssignmentQuery"
            headers = {'Content-Type': 'application/json'}
            payload = {'userid': self.userid}
            
            if time_range:
                payload['time_range'] = time_range
            if class_name:
                payload['class'] = class_name
            
            print(f"Requesting data from database with payload: {payload}")
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            if data.get('success'):
                print("Data fetched successfully.")
                return data
            else:
                print("Failed to fetch data:", data.get('message', 'No message provided.'))
                return {"success": False, "message": data.get('message', 'Failed to fetch data from API.')}
        
        except requests.exceptions.RequestException as e:
            print(f"Error during API request: {e}")
            return {"success": False, "message": f"Network or API error: {e}"}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            return {"success": False, "message": f"Invalid JSON response: {e}"}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"success": False, "message": f"An unexpected error occurred: {e}"}