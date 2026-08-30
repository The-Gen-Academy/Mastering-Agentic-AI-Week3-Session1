import config
from orchestrator_agent import build_orchestrator_agent


def run_agent(question: str):

    agent = build_orchestrator_agent()

    initial_state = {
        "messages": [
            {
                "role": "user",
                "content": question,
            }
        ],
        "files": {
            "research.md": "",
            "analysis.md": "",
            "final.md": "",
        },
        config={
            "recursion_limit": config.GRAPH_RECURSION_LIMIT
        }
    }

    result = agent.invoke(initial_state)

    return result


def print_results(result):

    print("\n" + "=" * 70)
    print("FINAL RESPONSE")
    print("=" * 70)

    messages = result.get("messages", [])

    if messages:
        final_message = messages[-1]

        if hasattr(final_message, "content"):
            print(final_message.content)
        else:
            print(final_message)

    print("\n" + "=" * 70)
    print("AGENT WORKSPACE")
    print("=" * 70)

    files = result.get("files", {})

    for filename, content in files.items():
        print(f"\n--- {filename} ---\n")

        if content:
            print(content[:5000])
        else:
            print("[empty]")


if __name__ == "__main__":

    print("\nDeepAgent Multi-Agent System")
    print("=" * 70)

    question = input("\nEnter your question:\n\n")

    result = run_agent(question)

    print_results(result)