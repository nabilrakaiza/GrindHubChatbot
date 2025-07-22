"""
File Explanation: core.py

This file will be the core of chatbot feature. This will handle create/find thread based on user email, intent classification,
delegation to specific agent, and sends the message to the user. 

"""
# Library Needed
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
# from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import ConversationChain
import requests
import os

from .AI_Agents.general_information import GeneralInformation
from .AI_Agents.greeting_farewell import GreetingFarewell
from .AI_Agents.motivation_stress_support import MotivationStressSupport
from .AI_Agents.performance_assignment_query import PerformanceAssignmentQuery
from .AI_Agents.study_plan_request import StudyPlanRequest
from .AI_Agents.others import Others
from .helpers import *

class UserIntent(BaseModel):
    """
    This class will define the user intent based on the user message.
    It will be used to classify the intent of the user message.
    """
    intent: str = Field(..., description="Intent of the user message")

def get_user_message():
    """
    This function will get the user message from the input.

    Input : User message (string)
    Output: User message (string)
    """
    user_message = input("User: ")
    return user_message

def save_user_message(user_message):
    """
    This function will save the user message to the database.

    Input : User message (string)
    Output: None
    """
    pass

def classify_intent(user_message):
    """
    This function will classify the intent of the user message.

    Input : User message (string)
    Output: Intent (string)
    """

    prompt_template = """
    You are an AI assistant that classifies user intents based on their messages.
    Given the user message, classify the intent into one of the following categories:
    1. General Information
    2. Greeting and Farewell
    3. Motivation and Stress Support
    4. Performance and Assignment Query
    5. Study Plan Request
    6. Others
    
    User Message: {user_message}
    Return the intent as a string. Please make sure that the string is exactly the same as the categories above. 
    """
    
    intent = call_llm(
        prompt_template=prompt_template,
        model=UserIntent,
        params={"user_message": user_message}
    )

    return intent.intent

def create_or_find_thread():
    """
    This function will create a new thread or find an existing thread based on user email.

    Input : User email (string)
    Output: Thread ID (string)
    """
    pass

def delegation_to_specific_ai_agent(intent: str, user_message: str, context: str = None):
    """
    This function will delegate the user message to a specific AI agent based on the intent classification.

    Input : User message (string), Intent (string)
    Output: AI Agent response (object)
    """
    
    if intent == "Motivation and Stress Support":
        ai_agent = MotivationStressSupport(api_key=os.getenv("GOOGLE_API_KEY"), userid="nabilrakaiza")
        response = ai_agent.process_message(user_message=user_message, context=context)
        return response
    
    elif intent == "Performance and Assignment Query":
        ai_agent = PerformanceAssignmentQuery(api_key=os.getenv("GOOGLE_API_KEY"), userid="nabilrakaiza")
        response = ai_agent.process_message(user_message=user_message, context=context)
        return response
    
    elif intent == "Study Plan Request":
        ai_agent = StudyPlanRequest(api_key=os.getenv("GOOGLE_API_KEY"), userid="nabilrakaiza")
        response = ai_agent.process_message(user_message=user_message, context=context)
        return response
    
    elif intent == "General Information":
        ai_agent = GeneralInformation(api_key=os.getenv("GOOGLE_API_KEY"))
        response = ai_agent.process_message(user_message=user_message, context=context)
        return response
    
    elif intent == "Greeting and Farewell":
        ai_agent = GreetingFarewell(api_key=os.getenv("GOOGLE_API_KEY"))
        response = ai_agent.process_message(user_message=user_message, context=context)
        return response
    
    elif intent == "Others":
        # Handle other intents or fallback to a default response
        # You can create a generic AI agent or return a predefined message
        ai_agent = Others(api_key=os.getenv("GOOGLE_API_KEY"), userid="nabilrakaiza")
        response = ai_agent.process_message(user_message=user_message, context=context)
        return response
    
    else:
        # If the intent is not recognized, return a default response or handle accordingly
        return "I'm sorry, I don't understand your request. Can you please rephrase it?"

def send_message_to_user():
    """
    This function will send the AI agent's response to the user.

    Input : AI Agent response (string)
    Output: None
    """
    pass

def save_ai_agent_response():
    """
    This function will save the AI agent's response to the database.

    Input : AI Agent response (string)
    Output: None
    """
    pass

def process_message(user_message: str, context: str = None):
    """
    This function will be the main function of the chatbot feature.
    It will handle the user message, intent classification, delegation to specific AI agent,
    and sending the message to the user.
    """
    intent = classify_intent(user_message=user_message)
    response = delegation_to_specific_ai_agent(intent=intent, user_message=user_message, context=context)
    # context = call_llm_text_response(
    #     prompt_template="""Summarise the following conversation between a user and a chatbot AI Agent. Please make sure that current context is included in the summary.
    #     context : {context}
    #     user_message : {user_message}
    #     chatbot response : {response}""",
    #     params={
    #         "context": context,
    #         "user_message": user_message,
    #         "response": response
    #     }
    # ) ## update context to database after every chatbot response?, and reset context to empty if no message from user for 30 minutes ?
    # # so there will be a database dedicated to holding the context of the conversation, and delete the context if no message from user for 30 minutes
    # print(context, intent)

    return response

def local_test():
    """
    This function is for local testing of the chatbot core functionality.
    It will simulate a conversation with the AI agent.
    """
    user_username = input("Enter your username: ")
    # user_message = input("User message: ")
    # intent = classify_intent(user_message=user_message)

    # probably will be a summary, or a compilation of last several messages. 
    context = "" # need to implement this somehow

    
    # print(f"Classified Intent: {intent}")

    user_message = ""

    while user_message != "exit":
        # if intent == "Motivation and Stress Support":
            user_message = input("User message: ")
            intent = classify_intent(user_message=user_message)
            response = delegation_to_specific_ai_agent(intent=intent, user_message=user_message, context=context)
            context = call_llm_text_response(
                prompt_template="""Summarise the following conversation between a user and a chatbot AI Agent. Please make sure that current context is included in the summary.
                context : {context}
                user_message : {user_message}
                chatbot response : {response}""",
                params={
                    "context": context,
                    "user_message": user_message,
                    "response": response
                }
            ) ## update context to database after every chatbot response?, and reset context to empty if no message from user for 30 minutes ?
            # so there will be a database dedicated to holding the context of the conversation, and delete the context if no message from user for 30 minutes
            print(context, intent)
    # context = None # get new context by summarising last context + last user message + last chatbot response

    

if __name__ == "__main__":
    local_test()

    # Uncomment the line below to run the main function
    # main()
