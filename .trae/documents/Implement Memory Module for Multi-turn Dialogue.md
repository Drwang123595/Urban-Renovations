# Implement Memory Module and Verify Multi-turn Dialogue

I will implement a `ConversationMemory` class to manage dialogue history and refactor the existing strategies to use it. I will also create a test script to verify that the context is correctly preserved during multi-turn conversations.

## Implementation Steps

1.  **Create Memory Module (`src/memory.py`)**
    *   Define `ConversationMemory` class.
    *   Methods: `add_system_message`, `add_user_message`, `add_assistant_message`, `get_messages`, `clear`.
    *   This ensures structured management of the `messages` list.

2.  **Create Test Script (`test_multiturn_dialogue.py`)**
    *   Initialize `DeepSeekClient` and `ConversationMemory`.
    *   Simulate a 2-round dialogue (similar to the DeepSeek example).
    *   **Round 1**: User asks a question -> Call API -> Store response.
    *   **Round 2**: User asks a follow-up question -> Call API (sending full history) -> Store response.
    *   Print the `messages` list at each step to demonstrate that history is preserved.

3.  **Refactor Strategies to Use Memory**
    *   Update `src/strategies/stepwise.py` to use `ConversationMemory` for managing its 3-step dialogue.
    *   Update `src/strategies/single.py` to use `ConversationMemory` for consistency (even though it's single-turn).

4.  **Verification**
    *   Run `test_multiturn_dialogue.py` to confirm the API handles context correctly.
