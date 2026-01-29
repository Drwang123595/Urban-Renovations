import time
from typing import List, Dict, Any, Optional
from openai import OpenAI, APIError, RateLimitError
from .config import Config

class DeepSeekClient:
    """
    Generic LLM Client wrapper compatible with OpenAI SDK.
    Despite the name (legacy), it supports any OpenAI-compatible provider.
    """
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or Config.API_KEY
        self.base_url = base_url or Config.BASE_URL
        self.model = model or Config.MODEL_NAME
        
        if not self.api_key:
            print("Warning: API Key is not set. API calls will fail.")
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=Config.TIMEOUT
        )
            
    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.1, max_retries: int = 3) -> Optional[str]:
        """
        Call LLM API for chat completion using OpenAI SDK.
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=Config.MAX_TOKENS,
                    stream=False
                )
                
                content = response.choices[0].message.content
                return content
                
            except RateLimitError as e:
                print(f"Rate Limit Hit (Attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(5 * (attempt + 1)) # Aggressive backoff
                
            except APIError as e:
                print(f"API Error (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    return None
                    
            except Exception as e:
                print(f"Unexpected Error (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    return None
                    
        return None
