import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.memory import ConversationMemory

def list_sessions():
    Config.load_env()
    index_file = Config.INDEX_FILE
    
    if not index_file.exists():
        print("No conversation history found.")
        return

    with open(index_file, "r", encoding="utf-8") as f:
        entries = json.load(f)

    print(f"\nFound {len(entries)} sessions:\n")
    print(f"{'ID':<38} {'Date':<20} {'Msgs':<5} {'Title'}")
    print("-" * 80)
    
    for e in entries:
        dt = datetime.fromtimestamp(e["updated_at"]).strftime("%Y-%m-%d %H:%M")
        print(f"{e['session_id']:<38} {dt:<20} {e['message_count']:<5} {e['title']}")
    print("")

def view_session(session_id):
    Config.load_env()
    memory = ConversationMemory(session_id=session_id)
    
    if not memory.messages:
        print(f"Session {session_id} not found or empty.")
        return

    print(f"\n=== Session: {session_id} ===\n")
    for msg in memory.get_messages():
        role = msg["role"].upper()
        content = msg["content"]
        print(f"[{role}]: {content}\n{'-'*40}")

def delete_session(session_id):
    Config.load_env()
    session_file = Config.SESSIONS_DIR / f"{session_id}.json"
    index_file = Config.INDEX_FILE
    
    if session_file.exists():
        session_file.unlink()
        print(f"Deleted session file: {session_id}")
    else:
        print(f"Session file not found: {session_id}")
        
    # Update index
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            entries = json.load(f)
        
        entries = [e for e in entries if e["session_id"] != session_id]
        
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
            
        print("Index updated.")

def main():
    parser = argparse.ArgumentParser(description="Chat History Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # List
    subparsers.add_parser("list", help="List all sessions")
    
    # View
    view_parser = subparsers.add_parser("view", help="View a session")
    view_parser.add_argument("session_id", help="Session ID")
    
    # Delete
    del_parser = subparsers.add_parser("delete", help="Delete a session")
    del_parser.add_argument("session_id", help="Session ID")

    args = parser.parse_args()
    
    if args.command == "list":
        list_sessions()
    elif args.command == "view":
        view_session(args.session_id)
    elif args.command == "delete":
        delete_session(args.session_id)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
