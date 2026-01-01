import asyncio

import requests
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# @tool
# def get_weather(city: str) -> str:
#     """Get the weather for a given city."""
#     return "the weather is good"


@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        current = data["current_condition"][0]
        location = data["nearest_area"][0]["areaName"][0]["value"]

        temp_c = current["temp_C"]
        condition = current["weatherDesc"][0]["value"]
        humidity = current["humidity"]
        wind_speed = current["windspeedKmph"]

        return f"Weather in {location}: {temp_c}°C, {condition}. Humidity: {humidity}%, Wind: {wind_speed} km/h"
    except Exception as e:
        return f"Sorry, couldn't fetch weather for {city}: {str(e)}"


def create_llm_request(message: str):
    return {"messages": [{"role": "user", "content": message}]}


async def get_request(agent):
    msg = input("Please enter your request.\n")
    request = {"messages": [{"role": "user", "content": msg}]}
    for chunk, metadata in agent.stream(
        request,
        stream_mode="messages",
    ):
        if chunk.type == "AIMessageChunk":
            if chunk.content and chunk.content != "null":
                print(chunk.content, end="", flush=True)
    print()
    return


async def main():
    llm = ChatOllama(model="qwen2.5", temperature=1)
    tools = [get_weather]
    system_prompt = "You are a weather agent. You are given a city and you need to return the weather for that city."

    agent = create_agent(
        llm,
        tools,
        system_prompt=system_prompt,
    )

    task1 = asyncio.create_task(get_request(agent))
    await task1


def get_agent():
    llm = ChatOllama(model="qwen2.5", temperature=1)
    tools = [get_weather]
    system_prompt = "You are a weather agent. You are given a city and you need to return the weather for that city."

    agent = create_agent(
        llm,
        tools,
        system_prompt=system_prompt,
    )
    return agent


if __name__ == "__main__":
    asyncio.run(main())
