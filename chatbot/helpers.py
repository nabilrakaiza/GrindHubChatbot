# this code was originally taken from my internship codebase and modifed it 
import os
from typing import Any, TypeVar
from typing import Dict, Optional, Type
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers.pydantic import TBaseModel
from langchain_google_genai import ChatGoogleGenerativeAI # Changed import

T = TypeVar('T', bound=BaseModel)

def call_gemini_with_prompt_model( # Renamed function
    api_key: str | None,
    prompt: str,
    pydantic_object: Optional[Type[TBaseModel]],
    model: str = "gemini-2.0-flash", # Changed default model
) -> Dict:
    """
    Generic function to ask for Gemini's inference. Return value will be in pydantic_object
    :param api_key:
    :param prompt:
    :param pydantic_object:
    :return:
    """
    if api_key is None:
        api_key = os.getenv("GOOGLE_API_KEY") # Changed environment variable
    if api_key is None:
        raise ValueError("GOOGLE_API_KEY is not set") # Changed error message
    
    # Changed ChatOpenAI to ChatGoogleGenerativeAI and api_key to google_api_key
    chat = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.2) 
    parser = JsonOutputParser(pydantic_object=pydantic_object)
    format_instructions = parser.get_format_instructions()
    _prompt = f"Answer the user query.\n{format_instructions}\n{prompt}\n"
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": _prompt},
            ]
        )
    ]

    text_result = chat.invoke(messages)
    return parser.invoke(text_result)


def call_llm(
    prompt_template: str,
    model: type[T],
    params: dict[str, Any] | None = None
) -> T:
    if params is None:
        params = {}
    RETRY_COUNT = 0
    error_message = ""
    while RETRY_COUNT < 3:
        try:
            res_dict = call_gemini_with_prompt_model( # Changed function call
                api_key=os.getenv("GOOGLE_API_KEY"), # Changed environment variable
                prompt=prompt_template.format(**params),
                pydantic_object=model
            )
            return model.model_validate(res_dict)
        except Exception as e:
            error_message = str(e)
            RETRY_COUNT += 1
            print(f"Error: {e}")
            print(f"Retrying... ({RETRY_COUNT}/3)")

    raise Exception(f"Failed to call LLM after {RETRY_COUNT} attempts because of {error_message}")


def call_llm_text_response(
        prompt_template: str, params: dict[str, Any] | None = None) -> str:
    if params is None:
        params = {}

    class TextResponse(BaseModel):
        text: str

    res_dict = call_gemini_with_prompt_model( # Changed function call
        api_key=os.getenv("GOOGLE_API_KEY"), # Changed environment variable
        prompt=prompt_template.format(**params),
        pydantic_object=TextResponse
    )

    return res_dict['text']