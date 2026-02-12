import argparse
import sys

from .runtime import MultiAgentRuntime


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-agent CLI")
    parser.add_argument(
        "--mode",
        choices=("admin", "user"),
        required=True,
        help="CLI mode. admin routes via supervisor; user routes to customer.",
    )
    parser.add_argument(
        "--model-mode",
        choices=("auto", "openrouter", "offline"),
        default="auto",
        help="Model selection mode. auto uses .env OpenRouter when available.",
    )
    parser.add_argument(
        "--once",
        help="Run a single turn and exit.",
    )
    return parser


def _run_loop(mode: str, runtime: MultiAgentRuntime) -> int:
    prompt = "admin> " if mode == "admin" else "user> "
    handler = runtime.run_admin_turn if mode == "admin" else runtime.run_user_turn

    while True:
        try:
            line = input(prompt).strip()
        except EOFError:
            return 0

        if line.lower() in {"exit", "quit"}:
            return 0
        if not line:
            continue

        print(handler(line))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    runtime = MultiAgentRuntime.create(model_mode=args.model_mode)

    handler = runtime.run_admin_turn if args.mode == "admin" else runtime.run_user_turn
    if args.once is not None:
        print(handler(args.once))
        return 0

    return _run_loop(args.mode, runtime)


if __name__ == "__main__":
    sys.exit(main())
