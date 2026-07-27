import asyncio
import logging
import os
from typing import Annotated

from livekit import rtc, api
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    inference
)
from livekit.agents import AgentSession, Agent
from livekit.plugins import openai, cartesia, silero
from dotenv import load_dotenv

load_dotenv()

# Import the custom LiveTalking Avatar plugin we built
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#from livetalking_plugin import LiveTalkingAvatarSession

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("avatar-agent")

# The identity of our backend agent
AGENT_IDENTITY = "avatar-agent-backend"

from livekit.agents.voice.room_io.types import RoomOptions

async def entrypoint(ctx: JobContext):
    log.info(f"Connecting to room {ctx.room.name} as {AGENT_IDENTITY}")

    # 1. Connect to LiveKit Room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # 2. Setup standard Voice Pipeline Session using AgentSession
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="en"),
        llm=inference.LLM(
            model="google/gemma-4-31b-it",
        ),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
            language="en",
            sample_rate=24000
        )
    )

    # 3. Stream audio to our NavTalk worker
    from livekit.agents.voice.avatar import DataStreamAudioOutput
    session.output.audio = DataStreamAudioOutput(
        room=ctx.room,
        destination_identity="navtalk-avatar",
        sample_rate=24000,
        wait_remote_track=rtc.TrackKind.KIND_AUDIO,
    )

    # 5. Wait for a REAL user to join (ignore navtalk-avatar)
    user_participant = None
    for p in ctx.room.remote_participants.values():
        if p.identity != "navtalk-avatar" and not p.identity.startswith("agent"):
            user_participant = p
            break
            
    if not user_participant:
        log.info("Waiting for a user to join...")
        future = asyncio.Future()
        def on_participant_connected(p):
            if p.identity != "navtalk-avatar" and not p.identity.startswith("agent"):
                if not future.done():
                    future.set_result(p)
        ctx.room.on("participant_connected", on_participant_connected)
        user_participant = await future
        ctx.room.off("participant_connected", on_participant_connected)
        
    log.info(f"User participant joined: {user_participant.identity}")

    # 6. Start the Agent session with the user
    # Create the Agent identity and pass instructions
    agent = Agent(
        instructions="You are a helpful and concise voice assistant. You speak clearly and respond quickly.",
    )
    await session.start(
        room=ctx.room,
        agent=agent,
        room_options=RoomOptions(
            close_on_disconnect=False,
            participant_identity=user_participant.identity
        )
    )

    log.info("Agent session started successfully.")

    await asyncio.sleep(1)
    
    log.info("Triggering initial greeting...")
    await session.say("Hello! I am your AI avatar. How can I help you today?", allow_interruptions=True)
    
    log.info("Greeting triggered successfully.")

    # Keep alive
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        log.info("Agent shutting down")

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
