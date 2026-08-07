"""Minimal server endpoint that mints ephemeral tokens for client apps.

Run with: uvicorn token_server:app --reload
A browser or mobile client calls POST /session and gets back a short-lived
client secret instead of your permanent XAI_API_KEY.
"""
import httpx
from fastapi import FastAPI

from config import CLIENT_SECRETS_URL, XAI_API_KEY

app = FastAPI()


@app.post("/session")
async def create_session():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            CLIENT_SECRETS_URL,
            headers={"Authorization": f"Bearer {XAI_API_KEY}"},
            json={"expires_after": {"seconds": 300}},
        )
    # Response body looks like {"value": "xai-realtime-client-secret-...", "expires_at": 1785870204}
    return response.json()
