"""
Local Coding Agent — terminal interface.

Uses core.py for tools/model/logging. Behavior is identical to before:
run_command still pauses for a y/N approval right here in the terminal.
"""

import core


def run_command_with_approval(command: str) -> str:
    print(f"\n[APPROVAL NEEDED] Agent wants to run: {command}")
    answer = input("Allow? [y/N]: ").strip().lower()
    if answer != "y":
        return "DENIED by user."
    return core.execute_command(command)


def run_agent(user_prompt: str, max_turns: int = 10):
    messages = [
        {"role": "system", "content": core.SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    core.log_event({"type": "user_prompt", "content": user_prompt})
    last_call_signature = None

    for turn in range(max_turns):
        message = core.call_model(messages)
        messages.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            print(f"\n[AGENT]: {message['content']}")
            core.log_event({"type": "final_response", "content": message["content"]})
            return message["content"]

        for call in tool_calls:
            name, args = core.parse_tool_call(call)
            print(f"\n[TOOL CALL] {name}({args})")
            core.log_event({"type": "tool_call", "name": name, "args": args})

            if name == "run_command":
                result = run_command_with_approval(args["command"])
            else:
                result = core.run_tool(name, args)

            # Stop the model from retrying an identical failed call forever
            signature = (name, str(args))
            if result.startswith("ERROR") and signature == last_call_signature:
                result += (
                    " REPEATED IDENTICAL FAILED CALL DETECTED. Do not retry "
                    "the exact same arguments again — they will fail again. "
                    "Re-read the file first, or try a different approach."
                )
            last_call_signature = signature

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
    print("Local Coding Agent — type your request (Ctrl+C to quit)")
    while True:
        try:
            prompt = input("\n> ")
        except (KeyboardInterrupt, EOFError):
            break
        run_agent(prompt)
