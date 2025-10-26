import os
import traceback
from abc import ABC

import google.generativeai as genai
from google.api_core import exceptions


class BaseLLM(ABC):

    def __init__(self, *args, **kwargs):
        pass

    def generate(self, prompt: str, **kwargs):
        raise NotImplementedError

    def chat(self, prompt: str, **kwargs):
        raise NotImplementedError


class GeminiLLM(BaseLLM):
    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: str = None, temperature: float = 0.0):
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or api_key
        if not GOOGLE_API_KEY:
            raise ValueError(
                "Please set GOOGLE_API_KEY environment variable or pass api_key parameter."
            )
        
        genai.configure(api_key=GOOGLE_API_KEY)
        
        generation_config = genai.GenerationConfig(temperature=temperature)
        self.model = genai.GenerativeModel(model_name, generation_config=generation_config)
    
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        retries = 0
        
        while retries <= max_retries:
            try:
                response = self.model.generate_content(prompt)
                if not response.text:
                    print(f"Warning: Empty response. Prompt filter reason: {response.prompt_feedback}")
                    return "not found #1"
                return response.text
            except exceptions.InternalServerError as e:
                retries += 1
                print(f"Internal server error. Retry {retries}/{max_retries}...")
                if retries > max_retries:
                    raise Exception(f"Failed after {max_retries} retries") from e
            except exceptions.ResourceExhausted as e:
                print("Rate limit exceeded. Please wait and try again.")
                raise e
            except Exception as e:
                print(f"An error occurred: {e}")
                traceback.print_exc()
                retries += 1
                if retries > max_retries:
                    raise Exception(f"Failed after {max_retries} retries") from e
        
        return "not found #2"
    
    def chat(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)