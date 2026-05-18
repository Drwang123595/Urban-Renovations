from __future__ import annotations

from scripts.pipeline import main_py313


def main(argv: list[str] | None = None) -> int:
    return int(main_py313.main(argv) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
