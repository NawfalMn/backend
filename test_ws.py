import asyncio
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidMessage

async def test():
    try:
        headers = {"Bypass-Tunnel-Reminder": "true"}
        async with connect("wss://nasty-snails-lie.loca.lt/v1/audio2video/musetalk", additional_headers=headers) as ws:
            print("Connected!")
    except InvalidMessage as e:
        print("Invalid Message!")
    except Exception as e:
        print("Error:", repr(e))

asyncio.run(test())
