import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.llm_client import DeepSeekClient
from src.memory import ConversationMemory

def test_multiturn():
    # Load env
    Config.load_env()
    
    if not Config.API_KEY:
        print("Error: DEEPSEEK_API_KEY not found.")
        return

    client = DeepSeekClient()
    memory = ConversationMemory()

    print("=== Starting Multi-turn Dialogue Test ===\n")

    # Round 1
    question1 = "What is the capital of France?"
    print(f"[User]: {question1}")
    memory.add_user_message(question1)
    
    # Send to API (Round 1)
    # Note: DeepSeekClient.chat_completion expects a list of messages
    response1 = client.chat_completion(memory.get_messages())
    if not response1:
        print("Error: No response from API.")
        return
        
    print(f"[Assistant]: {response1}")
    memory.add_assistant_message(response1)
    
    print(f"\nCurrent History Length: {len(memory.get_messages())}")
    print("-" * 30 + "\n")

    # Round 2
    question2 = "What is its population?"
    print(f"[User]: {question2}")
    memory.add_user_message(question2)
    
    # Send to API (Round 2)
    # This call includes: User Q1, Assistant A1, User Q2
    response2 = client.chat_completion(memory.get_messages())
    if not response2:
        print("Error: No response from API.")
        return

    print(f"[Assistant]: {response2}")
    memory.add_assistant_message(response2)

    print(f"\nFinal History Length: {len(memory.get_messages())}")
    print("\nFull Conversation History:")
    for msg in memory.get_messages():
        print(f"  {msg['role'].upper()}: {msg['content']}")

    print("\n=== Test Completed ===")

if __name__ == "__main__":
    test_multiturn()
