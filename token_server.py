import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from livekit import api
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI()

# Allow frontend to request tokens
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For dev only
    allow_methods=["*"],
    allow_headers=["*"],
)

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")
ROOM_NAME = "avatar-room"

@app.get("/getToken")
def get_token(identity: str = "react-user"):
    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name("React User")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=ROOM_NAME,
            )
        )
        .to_jwt()
    )
    return {"token": token, "url": LIVEKIT_URL}

if __name__ == "__main__":
    print(f"Token server running on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
