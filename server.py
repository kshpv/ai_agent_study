import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from llm import create_llm_request, get_agent
from users import add_users, load_users


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
    sender_id = load_users()[sender]
    if sender:
        await manager.connect(websocket, sender)
        response = {"sender": sender, "message": "got connected"}
        await manager.broadcast(response)
        try:
            while True:
                data = await websocket.receive_json()
                await manager.send_to(websocket, data)

                llm_request = create_llm_request(data["message"])

                async with streaming(websocket):
                    async for chunk, _metadata in agent.astream(
                        llm_request,
                        config={"configurable": {"thread_id": sender_id}},
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
            config = {"configurable": {"thread_id": sender_id}}
            history = agent.get_state_history(config)

            conversation_data = {
                "thread_id": config["configurable"]["thread_id"],
                "exported_at": datetime.utcnow().isoformat(),
                "checkpoints": [],
            }

            for cp in history:
                checkpoint_data = {
                    "checkpoint_id": cp.metadata.get(
                        "checkpoint_id", "unknown"
                    ),  # From metadata!
                    "values": {
                        key: value  # Serialize values (messages are serializable)
                        for key, value in cp.values.items()
                    },
                    "metadata": cp.metadata,
                    "created_at": cp.created_at,
                }
                conversation_data["checkpoints"].append(checkpoint_data)

            with open("conversation_history.json", "w") as f:
                json.dump(conversation_data, f, indent=2, default=str)

            print("Saved successfully!")

            manager.disconnect(websocket, sender)


@app.get("/api/current_user")
def get_user(request: Request):
    return request.cookies.get("X-Authorization")


@app.post("/api/register")
def register_user(user: RegisterValidator, response: Response):
    new_user = user.username
    users = load_users()
    if new_user not in users:
        add_users(new_user)
    response.set_cookie(key="X-Authorization", value=new_user, httponly=True)
    return {"status": "success", "username": new_user}


# locate templates
templates = Jinja2Templates(directory="templates")
manager = SocketManager()


@app.get("/")
def get_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/chat")
def get_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})
    return templates.TemplateResponse("chat.html", {"request": request})
