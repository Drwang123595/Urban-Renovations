import json
import requests
import time
from typing import List, Dict, Any, Optional
from .config import Config

class DeepSeekClient:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or Config.API_KEY
        self.base_url = base_url or Config.BASE_URL
        self.model = model or Config.MODEL_NAME
        
        if not self.api_key:
            print("Warning: API Key is not set. API calls will fail.")
            
    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.1, max_retries: int = 3) -> Optional[str]:
        """
        Call DeepSeek API for chat completion.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": Config.MAX_TOKENS,
            "stream": False 
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=Config.TIMEOUT)
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return content
                
            except Exception as e:
                print(f"API Call Failed (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    return None
                    
        return None
