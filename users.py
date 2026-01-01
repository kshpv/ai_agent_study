import json
from pathlib import Path

USERS_FILENAME = Path("users.json")


def load_users():
    if not USERS_FILENAME.exists():
        with open(USERS_FILENAME, "w") as f:
            json.dump({}, f)
    with open(USERS_FILENAME, "r") as f:
        users = json.load(f)
    return users


def add_users(user: str):
    with open(USERS_FILENAME, "r") as f:
        users = json.load(f)
    users[user] = id(user)
    with open(USERS_FILENAME, "w") as f:
        json.dump(users, f)
