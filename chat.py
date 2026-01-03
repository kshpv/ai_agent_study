from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

Role = Literal["user", "assistant", "system", "tool"]


def now_iso() -> str:
    # Use local timezone if you prefer; this is UTC for simplicity
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StoredMessage:
    role: Role
    content: str
    ts: str


@dataclass
class ChatSession:
    schema_version: int
    session_id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[StoredMessage]
    metadata: Dict[str, Any]

    @staticmethod
    def new(
        session_id: str,
        title: str = "New chat",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ChatSession":
        t = now_iso()
        return ChatSession(
            schema_version=1,
            session_id=session_id,
            title=title,
            created_at=t,
            updated_at=t,
            messages=[],
            metadata=metadata or {},
        )


def get_messages_history(messages: list[StoredMessage]) -> list[BaseMessage]:
    return [stored_to_lc(msg) for msg in messages]


def stored_to_lc(msg: StoredMessage) -> BaseMessage:
    if msg.role == "user":
        return HumanMessage(content=msg.content)
    if msg.role == "assistant":
        return AIMessage(content=msg.content)
    if msg.role == "system":
        return SystemMessage(content=msg.content)
    if msg.role == "tool":
        # ToolMessage typically needs a tool_call_id; if you don't track it, store in content only
        return ToolMessage(content=msg.content, tool_call_id="stored")
    raise ValueError(f"Unknown role: {msg.role}")


def lc_to_stored(msg: BaseMessage) -> StoredMessage:
    # Map message classes to stable roles
    if isinstance(msg, HumanMessage):
        role: Role = "user"
    elif isinstance(msg, AIMessage):
        role = "assistant"
    elif isinstance(msg, SystemMessage):
        role = "system"
    elif isinstance(msg, ToolMessage):
        role = "tool"
    else:
        # Fallback: treat unknown as assistant text
        role = "assistant"

    # LangChain messages can have non-string content (e.g. list of blocks). Handle safely:
    content = (
        msg.content
        if isinstance(msg.content, str)
        else json.dumps(msg.content, ensure_ascii=False)
    )
    return StoredMessage(role=role, content=content, ts=now_iso())


class FileChatStore:
    def __init__(self, root_dir: str = "sessions"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(exist_ok=True)

    def path(self, session_id: str) -> str:
        return self.root_dir / f"{session_id}.json"

    def save(self, session: ChatSession) -> None:
        session.updated_at = now_iso()
        data = asdict(session)
        # dataclasses -> dict but messages are dicts already
        tmp = self.path(session.session_id)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, session_id: str) -> ChatSession:
        with open(self.path(session_id), "r", encoding="utf-8") as f:
            raw = json.load(f)

        msgs = [StoredMessage(**m) for m in raw.get("messages", [])]
        return ChatSession(
            schema_version=raw.get("schema_version", 1),
            session_id=raw["session_id"],
            title=raw.get("title", "Chat"),
            created_at=raw.get("created_at", now_iso()),
            updated_at=raw.get("updated_at", raw.get("created_at", now_iso())),
            messages=msgs,
            metadata=raw.get("metadata", {}),
        )
