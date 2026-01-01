import asyncio
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from llm import create_llm_request, get_agent


class RegisterValidator(BaseModel):
    username: str


app = FastAPI()
agent = get_agent()


@asynccontextmanager
async def streaming(websocket):
    # Send start of stream indicator
    await manager.send_to(
        websocket, {"sender": "LLM", "message": "", "stream_start": True}
    )
    yield
    # Send end of stream indicator
    await manager.send_to(
        websocket, {"sender": "LLM", "message": "", "stream_end": True}
    )


# manager
class SocketManager:
    def __init__(self):
        self.active_connections: List[(WebSocket, str)] = []

    async def connect(self, websocket: WebSocket, user: str):
        await websocket.accept()
        self.active_connections.append((websocket, user))

    def disconnect(self, websocket: WebSocket, user: str):
        self.active_connections.remove((websocket, user))

    async def broadcast(self, data):
        for connection in self.active_connections:
            await connection[0].send_json(data)

    async def send_to(self, websocket: WebSocket, data):
        await websocket.send_json(data)


@app.websocket("/api/chat")
async def chat(websocket: WebSocket):
    sender = websocket.cookies.get("X-Authorization")
    if sender:
        await manager.connect(websocket, sender)
        response = {"sender": sender, "message": "got connected"}
        await manager.broadcast(response)
        try:
            while True:
                data = await websocket.receive_json()
                await manager.broadcast(data)
                await asyncio.sleep(0)

                llm_request = create_llm_request(data["message"])

                async with streaming(websocket):
                    async for chunk, _metadata in agent.astream(
                        llm_request,
                        stream_mode="messages",
                    ):
                        if chunk.type != "AIMessageChunk":
                            continue
                        if chunk.content and chunk.content != "null":
                            stream_data = {
                                "sender": "LLM",
                                "message": chunk.content,
                                "stream_chunk": True,
                            }
                            await manager.send_to(websocket, stream_data)

        except WebSocketDisconnect:
            manager.disconnect(websocket, sender)
            response["message"] = "left"
            await manager.broadcast(response)
            response["message"] = "left"
            await manager.broadcast(response)


@app.get("/api/current_user")
def get_user(request: Request):
    return request.cookies.get("X-Authorization")


@app.post("/api/register")
def register_user(user: RegisterValidator, response: Response):
    response.set_cookie(key="X-Authorization", value=user.username, httponly=True)
    return {"status": "success", "username": user.username}


# locate templates
templates = Jinja2Templates(directory="templates")
manager = SocketManager()


@app.get("/")
def get_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/chat")
def get_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})
