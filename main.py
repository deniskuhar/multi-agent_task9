from __future__ import annotations

from config import APP_TITLE, SEPARATOR
from supervisor import (
    new_thread_id,
    revise_report_with_feedback,
    run_supervisor,
    save_report_via_mcp,
)


def _handle_save_flow(report: dict, thread_id: str) -> str:
    while True:
        print("\n[Supervisor → MCP → save_report]")
        print("    ACTION REQUIRES APPROVAL")
        print(f"  Filename: {report['filename']}")

        preview = report["content"][:1200]
        print("  Preview:")
        print(preview)
        if len(report["content"]) > 1200:
            print("\n  ...")

        decision = input("\n  approve / edit / reject: ").strip().lower()

        if decision not in {"approve", "edit", "reject"}:
            print("  Please enter approve, edit, or reject.")
            continue

        if decision == "approve":
            result = save_report_via_mcp(report["filename"], report["content"])
            return f" {result}"

        if decision == "edit":
            feedback = input("    Your feedback: ").strip()
            report = revise_report_with_feedback(report, feedback)
            print("\n  Report revised based on your feedback.")
            continue

        return " Report saving was cancelled."


def main() -> None:
    thread_id = new_thread_id()

    print(SEPARATOR)
    print(APP_TITLE)
    print("Type 'exit' or 'quit' to leave. Type 'new' to reset the session.")
    print(SEPARATOR)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if user_input.lower() == "new":
            thread_id = new_thread_id()
            print("Started a new session.")
            continue

        try:
            report = run_supervisor(user_input)

            print("\n" + SEPARATOR)
            print("Draft report prepared")
            print(SEPARATOR)
            print(f"Filename: {report['filename']}")
            preview = report["content"][:1500]
            print(preview)
            if len(report["content"]) > 1500:
                print("\n...")

            final_message = _handle_save_flow(report, thread_id)
            print(f"\nAgent: {final_message}")

        except Exception as exc:
            print(f"\nAgent error: {exc}")


if __name__ == "__main__":
    main()