"""
This file is for an AI Agent that generates personalized study plans for users.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from langchain_core.output_parsers import JsonOutputParser
import requests
import json

class ExtractedStudyPlanComponent(BaseModel): # Renamed for clarity and specificity
    """
    Represents extracted components from the user message for generating a study plan.
    """
    time_range: Optional[str] = Field(None, description="Desired time range for the study plan, e.g., 'today', 'this week', 'next month'.")
    focus_area: Optional[str] = Field(None, description="Specific subject, module, or topic the user wants to focus on.")
    plan_type: Optional[str] = Field(None, description="Type of plan requested, e.g., 'detailed', 'summary', 'prioritized'.")
    # You can add more fields here like 'specific_assignment', 'deadline', 'available_hours', etc.

class StudyPlanRequest(): # Renamed the class for clarity
    """
    Main class for the Study Plan Request AI Agent.
    Helps users manage assignments, classes, study logs, and generates prioritized study plans.
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
            temperature=0.2 # Low temperature for structured and accurate plan generation
        )
        self.parser = JsonOutputParser(pydantic_object=ExtractedStudyPlanComponent) # Parser for structured extraction
    
    def process_message(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process the user message to generate a personalized study plan.
        This method orchestrates data extraction, data fetching, and response generation.

        :param user_message: The message from the user, e.g., "Can you make a study plan for this week?"
        :param context: Additional context (e.g., current date, time, user preferences).
        :return: A detailed study plan for the user.
        """
        llm = self.chat_model
        
        study_data = None

        # Step 1: Extract relevant parameters from the user message
        # extracted_params = self.extract_user_message(user_message)
        # time_range = extracted_params.time_range
        # focus_area = extracted_params.focus_area # New extracted field
        # plan_type = extracted_params.plan_type # New extracted field

        # # Step 2: Fetch user-specific study data from the database
        # study_data = self.get_data_from_database(time_range=time_range)

        # # Append fetched data and extracted params to context for the LLM
        if context is None:
            context = {}
        # context['study_data'] = study_data
        # context['extracted_plan_params'] = extracted_params

        system_prompt = f"""
        You are an intelligent AI assistant dedicated to creating personalized study plans.
        You have access to the user's assignments, classes, study logs, past performance, and preferences.
        Your goal is to analyze deadlines, user's past performance, and current tasks to suggest a prioritized list of tasks or a structured study schedule.
        When generating the study plan:
        - Make it detailed, practical, and actionable based on the user's request and provided data.
        - Prioritize tasks effectively, considering deadlines and difficulty.
        - If 'time_range' is provided, ensure the plan fits that duration.
        - If 'focus_area' is provided, emphasize tasks related to that area.
        - If 'plan_type' is specific (e.g., 'summary', 'detailed'), adhere to it.
        - If key information is missing for a good plan (e.g., no upcoming assignments), inform the user.
        - Ensure the response is easily readable from a mobile device, keeping it concise and clearly formatted.
        """

        user_prompt = f"""
        User request for study plan: "{user_message}"
        User's study data (assignments, classes, logs, preferences): {json.dumps(study_data, indent=2) if study_data else "No specific study data available."}
        
        Please generate a personalized and detailed study plan based on this information.
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        result = llm.invoke(messages)
        text_result = result.content
        print(f"Generated study plan: {text_result}")
        return text_result
        
    def extract_user_message(self, user_message: str) -> ExtractedStudyPlanComponent:
        """
        Extracts relevant components (time range, focus area, plan type)
        from the user's message using the LLM.

        :param user_message: The message from the user.
        :return: An instance of ExtractedStudyPlanComponent containing the extracted details.
        """
        extraction_prompt = f"""
        Extract the following information from the user's message for a study plan request:
        - time_range (e.g., 'today', 'this week', 'tomorrow', 'next month', 'upcoming')
        - focus_area (e.g., 'Math', 'Physics chapter 3', 'essay writing', 'History exam')
        - plan_type (e.g., 'detailed', 'summary', 'prioritized list', 'daily schedule')

        If a piece of information is not explicitly mentioned, leave it as null.
        
        User message: "{user_message}"

        Return the information as a JSON object adhering to the following schema:
        {{"time_range": "string or null", "focus_area": "string or null", "plan_type": "string or null"}}
        """

        messages = [
            SystemMessage(content="You are an expert at extracting specific data points from user queries for planning purposes."),
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
            print(f"Error parsing extracted study plan components: {e}")
            return ExtractedStudyPlanComponent(time_range=None, focus_area=None, plan_type=None)


    def get_data_from_database(self, time_range: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetches user's assignments, classes, study logs, and preferences from the database
        to inform the study plan generation.
        
        :param time_range: The desired time range for fetching relevant data.
        :return: A dictionary containing the fetched data or an error message.
        """
        try:
            url = "https://grindhub-production.up.railway.app/api/auth/getStudyPlanData" # Already correct endpoint
            headers = {'Content-Type': 'application/json'}
            payload = {'userid': self.userid}
            
            if time_range:
                payload['time_range'] = time_range
            
            print(f"Requesting study plan data from database with payload: {payload}")
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            if data.get('success'):
                print("Study plan data fetched successfully.")
                return data
            else:
                print("Failed to fetch study plan data:", data.get('message', 'No message provided.'))
                return {"success": False, "message": data.get('message', 'Failed to fetch study plan data from API.')}
        
        except requests.exceptions.RequestException as e:
            print(f"Error during API request: {e}")
            return {"success": False, "message": f"Network or API error: {e}"}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            return {"success": False, "message": f"Invalid JSON response: {e}"}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"success": False, "message": f"An unexpected error occurred: {e}"}