import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

client = OpenAI(
    api_key=os.environ.get("SUPER_MIND_API_KEY"),
    base_url="https://space.ai-builders.com/backend/v1",
)


class ChatRequest(BaseModel):
    user_message: str


@app.get("/")
def root():
    return {
        "message": "FastAPI is running!",
        "try": "/hello/YourName",
        "chat": "POST /chat with {\"user_message\": \"...\"}",
        "docs": "/docs",
    }


@app.get("/hello/{name}")
def hello(name: str):
    return {"message": f"Hello, World {name}"}


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        completion = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": request.user_message}],
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upstream API error: {exc}")

    return {"response": completion.choices[0].message.content}
