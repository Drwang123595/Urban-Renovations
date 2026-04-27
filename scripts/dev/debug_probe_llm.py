import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.config import Config
from src.runtime.llm_client import DeepSeekClient


def _build_env_snapshot(include_sensitive: bool = False):
    lines = [
        "=== Runtime Config Snapshot ===",
        f"BASE_URL: {Config.BASE_URL}",
        f"MODEL_NAME: {Config.MODEL_NAME}",
        f"TIMEOUT: {Config.TIMEOUT}",
        f"MAX_TOKENS: {Config.MAX_TOKENS}",
        f"API_KEY_PRESENT: {'yes' if Config.API_KEY else 'no'}",
    ]
    if include_sensitive:
        lines.extend(
            [
                "=== Filtered Proxy Snapshot ===",
                f"HTTP_PROXY_SET: {'yes' if os.environ.get('HTTP_PROXY') else 'no'}",
                f"HTTPS_PROXY_SET: {'yes' if os.environ.get('HTTPS_PROXY') else 'no'}",
                f"NO_PROXY_SET: {'yes' if os.environ.get('NO_PROXY') else 'no'}",
            ]
        )
    return lines


def _print_env_snapshot(include_sensitive: bool = False):
    for line in _build_env_snapshot(include_sensitive=include_sensitive):
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Minimal probe for OpenAI-compatible LLM endpoint")
    parser.add_argument("--proxy", help="Proxy URL, e.g. http://127.0.0.1:8888")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument(
        "--debug-sensitive",
        action="store_true",
        help="Show additional filtered diagnostics for troubleshooting.",
    )
    args = parser.parse_args()

    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy

    Config.load_env()
    Config.DEBUG_SENSITIVE_LOGGING = Config.DEBUG_SENSITIVE_LOGGING or args.debug_sensitive
    Config.validate_runtime_environment(
        require_py313=True,
        warn_on_minor_drift=True,
        required_modules=("openai",),
    )

    if not Config.API_KEY:
        print("Error: API key is empty.")
        return 1

    if args.model:
        Config.MODEL_NAME = args.model

    _print_env_snapshot(include_sensitive=args.debug_sensitive)

    client = DeepSeekClient(model=Config.MODEL_NAME)
    messages = [{"role": "user", "content": "Reply with OK only."}]
    print("=== Sending Probe Request ===")
    result = client.chat_completion(messages=messages, temperature=0.0, max_retries=1)
    if result is None:
        print("Probe result: FAILED")
        return 2

    print("Probe result: SUCCESS")
    print(f"Response: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
