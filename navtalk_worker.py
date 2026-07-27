import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
import livekit.rtc as rtc
from livekit.api import AccessToken, VideoGrants
from livekit.agents.voice.avatar import AvatarRunner, AvatarOptions
from livekit.agents.voice.avatar._datastream_io import DataStreamAudioReceiver

from livekit_plugin_navtalk.navtalk_client import NavTalkClient
from livekit_plugin_navtalk.navtalk_avatar import NavTalkVideoGenerator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("navtalk-worker")

load_dotenv()

async def main():
    url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    navtalk_license = os.getenv("NAVTALK_LICENSE", "")
    navtalk_avatar_id = os.getenv("NAVTALK_AVATAR_ID", "")
    navtalk_ws_url = os.getenv("NAVTALK_WS_URL", "wss://transfer.navtalk.ai/wss/v2/realtime-chat")

    if not url or not api_key or not api_secret:
        log.error("LiveKit credentials are not set in environment.")
        return

    room_name = "voice_assistant_room"
    room = rtc.Room()
    identity = "navtalk-avatar"
    token = AccessToken(api_key, api_secret).with_identity(identity).with_name("NavTalk Avatar").with_grants(VideoGrants(room_join=True, room=room_name)).to_jwt()

    log.info(f"Connecting to room: {room_name}")
    await room.connect(url, token)
    log.info("Connected to LiveKit!")

    # 1. Connect to NavTalk
    log.info(f"Connecting to NavTalk with avatar_id: {navtalk_avatar_id}")
    navtalk_client = NavTalkClient(navtalk_license, navtalk_avatar_id, navtalk_ws_url)
    
    # 2. Receive Audio from Agent
    audio_recv = DataStreamAudioReceiver(room)

    # 3. Setup NavTalk Generator (WebRTC bridge)
    video_gen = NavTalkVideoGenerator(navtalk_client)
    
    runner = AvatarRunner(
        room=room,
        audio_recv=audio_recv,
        video_gen=video_gen,
        options=AvatarOptions(
            video_width=720, # adjust based on expected NavTalk video
            video_height=1280,
            video_fps=25, # NavTalk likely uses 25 FPS
            audio_sample_rate=48000,
            audio_channels=2
        ),
        _lazy_publish=False,
    )

    # We must connect NavTalk WebSocket first so signaling can happen
    await navtalk_client.connect()

    log.info("Starting Video Publisher...")
    await runner.start()

    log.info("Worker is fully running! Waiting for agent to send audio...")
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await navtalk_client.disconnect()
        await runner.aclose()
        await room.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
