"""
Local Coding Agent - terminal interface.

Uses core.py for tools/model/logging. Risky tools pause for y/N approval.
"""

import core


def run_tool_with_approval(name: str, args: dict) -> str:
    print("\n[APPROVAL NEEDED]")
    print(core.describe_tool_call(name, args))

    answer = input("Allow? [y/N]: ").strip().lower()

    if answer != "y":
        core.log_event({
            "type": "approval_decision",
            "tool": name,
            "args": args,
            "approved": False,
        })
        return "DENIED by user."

    core.log_event({
        "type": "approval_decision",
        "tool": name,
        "args": args,
        "approved": True,
    })

    if name == "run_command":
        return core.execute_command(args["command"])

    return core.run_tool(name, args)


def run_agent(user_prompt: str, max_turns: int = 10):
    messages = [
        {"role": "system", "content": core.SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    core.log_event({"type": "user_prompt", "content": user_prompt})

    last_call_signature = None
    last_result_was_error = False

    for _ in range(max_turns):
        message = core.call_model(messages)
        messages.append(message)

        tool_calls = message.get("tool_calls")

        if not tool_calls:
            if last_result_was_error:
                print(
                    "\n[WARNING] The previous tool call failed or was denied. "
                    "Verify the final answer carefully."
                )

            print(f"\n[AGENT]: {message['content']}")
            core.log_event({"type": "final_response", "content": message["content"]})
            return message["content"]

        for call in tool_calls:
            name, args = core.parse_tool_call(call)

            print(f"\n[TOOL CALL] {name}({args})")
            core.log_event({"type": "tool_call", "name": name, "args": args})

            if core.tool_requires_approval(name):
                result = run_tool_with_approval(name, args)
            else:
                result = core.run_tool(name, args)

            signature = (name, str(args))

            if result.startswith("ERROR") and signature == last_call_signature:
                result += (
                    " REPEATED IDENTICAL FAILED CALL DETECTED. Do not retry "
                    "the exact same arguments again - they will fail again. "
                    "Re-read the file first, or try a different approach."
                )

            last_call_signature = signature
            last_result_was_error = result.startswith("ERROR") or result.startswith("DENIED")

            print(f"[TOOL RESULT] {result[:300]}")
            core.log_event({"type": "tool_result", "name": name, "result": result})

            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result,
            })

    print("[AGENT] Hit max turns without finishing.")
    return None


if __name__ == "__main__":
    print("Local Coding Agent - type your request (Ctrl+C to quit)")

    while True:
        try:
            prompt = input("\n> ")
        except (KeyboardInterrupt, EOFError):
            break

        run_agent(prompt)
