"""Text-mode demo of the support agent, run scenario by scenario.

This drives the same WebSocket path a microphone would, just with typed
text instead of audio input, so the tool loop, confirmation flow, and
cost tracking are all visible in a plain terminal without any audio
hardware. Swap send_text() for streamed input_audio_buffer.append() calls
once you are capturing real microphone audio.

Run: python run_demo.py
"""
import asyncio

from assistant import SupportAgent

SCENARIOS = [
    "Whats the status of order ORD-1042?",
    "Do you have any info on order ORD-9999?",
    "Please change the delivery instructions for order ORD-1042 to leave it "
    "with the doorman, yes I confirm please do it now.",
    "My order ORD-2210 never arrived and tracking hasn't moved in a week, "
    "please open a ticket about it.",
]


async def run_scenario(text: str):
    agent = SupportAgent()
    print(f"\nCaller: {text}")
    await agent.run_turn(text)
    for role, line in agent.transcript:
        print(f"{role}: {line}")
    print(f"(audio so far: {agent.client.cost.audio_seconds:.2f}s, "
          f"est. cost: ${agent.client.cost.total_usd:.5f})")
    await agent.close()


async def main():
    for scenario in SCENARIOS:
        await run_scenario(scenario)


if __name__ == "__main__":
    asyncio.run(main())
