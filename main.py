import os

# import gradio as gr
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel

load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]  # from .env or the environment
URL = "https://api.groq.com/openai/v1/chat/completions"


app = FastAPI()


class bodyToLlm(BaseModel):
    message: str


def apiCall(message, *args):
    response = requests.post(
        url=URL,
        headers={"Authorization": f"Bearer {groq_api_key}"},
        json={
            "model": "openai/gpt-oss-20b",
            "messages": [
                {
                    "role": "user",
                    "content": message,
                }
            ],
        },
    )
    return response.json()["choices"][0]["message"]["content"]


@app.post("/llm")
def response(data: bodyToLlm, request: Request):
    response = apiCall(data.message)
    return {"message": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
