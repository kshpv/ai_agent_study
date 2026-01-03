import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from chat import ChatSession, FileChatStore, get_messages_history, lc_to_stored
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
    # Extract session_id from query string
    query_string = websocket.url.query
    session_id = None
    if query_string:
        from urllib.parse import parse_qs

        params = parse_qs(query_string)
        if "session_id" in params:
            session_id = params["session_id"][0]

    sender = websocket.cookies.get("X-Authorization")
    if not sender:
        await websocket.close(code=1008, reason="Unauthorized")
        return

    sender_id = load_users().get(sender)
    if not sender_id:
        # Create new user if doesn't exist
        add_users(sender)
        sender_id = load_users()[sender]

    await manager.connect(websocket, sender)
    chat_store = FileChatStore()

    # Load existing session or create new one
    if session_id:
        try:
            session = chat_store.load(session_id)
            # Send loaded messages to frontend
            for stored_msg in session.messages:
                sender_name = "You" if stored_msg.role == "user" else "LLM"
                await manager.send_to(
                    websocket,
                    {
                        "sender": sender_name,
                        "message": stored_msg.content,
                        "loaded": True,
                    },
                )
        except FileNotFoundError:
            # Session not found, create new one with provided ID
            session = ChatSession.new(session_id, "New chat", metadata=None)
    else:
        # Create new session with unique ID
        session_id = str(uuid.uuid4())
        session = ChatSession.new(session_id, "New chat", metadata=None)

    # Ensure session has correct session_id
    session.session_id = session_id

    try:
        while True:
            data = await websocket.receive_json()
            await manager.send_to(websocket, data)

            user_text = data["message"]
            user_message = HumanMessage(content=user_text)
            stored_message = lc_to_stored(user_message)
            session.messages.append(stored_message)

            llm_request = create_llm_request(
                user_text, get_messages_history(session.messages)
            )
            full_message = []
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
                        full_message.append(chunk.content)
                        await manager.send_to(websocket, stream_data)

            response_text = "".join(full_message)
            response_message = AIMessage(content=response_text)
            stored_message = lc_to_stored(response_message)
            session.messages.append(stored_message)

    except WebSocketDisconnect:
        chat_store.save(session)
        manager.disconnect(websocket, sender)


@app.get("/api/sessions")
def get_sessions():
    """List all chat sessions from the sessions directory."""
    chat_store = FileChatStore()
    sessions_dir = Path(chat_store.root_dir)

    sessions = []
    if sessions_dir.exists():
        for session_file in sessions_dir.glob("*.json"):
            try:
                session_id = session_file.stem
                session = chat_store.load(session_id)
                sessions.append(
                    {
                        "id": session.session_id,
                        "name": session.title,
                        "created_at": session.created_at,
                        "updated_at": session.updated_at,
                    }
                )
            except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
                # Skip invalid session files
                continue

    # Sort by updated_at descending (most recent first)
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return {"sessions": sessions}


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
