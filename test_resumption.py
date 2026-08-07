"""Reproduces a disconnect and resume: tell the agent something, drop the
connection, reconnect with the saved conversation id, and ask about it.

Run: python test_resumption.py
"""
import asyncio

from assistant import SupportAgent


async def main():
    first = SupportAgent()
    await first.run_turn("Whats the status of order ORD-1042?")
    conversation_id = first.client.conversation_id
    for role, line in first.transcript:
        print(f"{role}: {line}")
    await first.close()
    print(f"Saved conversation_id: {conversation_id}")

    # Simulate a dropped call: wait a moment, then reconnect with the id.
    await asyncio.sleep(2)

    second = SupportAgent(resume_conversation_id=conversation_id)
    await second.run_turn("Sorry, my call dropped for a second there. What was the ETA again?")
    for role, line in second.transcript:
        print(f"{role}: {line}")
    await second.close()


if __name__ == "__main__":
    asyncio.run(main())
