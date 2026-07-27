import asyncio
import logging
import os
import sys
from dotenv import load_dotenv

from livekit import rtc
from livekit.api import AccessToken, VideoGrants
from livekit.agents.voice.avatar import AvatarRunner, DataStreamAudioReceiver, AvatarOptions

# Load env variables
load_dotenv()
log = logging.getLogger("livetalking-worker")
logging.basicConfig(level=logging.INFO)

# Inject LiveTalking path so we can import its modules
# Update this path to where LiveTalking is located on your Vast.ai instance
LIVETALKING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../LiveTalking"))
if LIVETALKING_DIR not in sys.path:
    sys.path.insert(0, LIVETALKING_DIR)

from avatars.wav2lip_avatar import LipReal, load_model, load_avatar
from streamout.base_output import BaseOutput
import numpy as np
import cv2

class DummyOpt:
    def __init__(self, sessionid, avatar_id):
        self.sessionid = sessionid
        self.avatar_id = avatar_id
        self.fps = 25
        self.batch_size = 4
        self.transport = "livekit"
        self.tts = "none"
        self.customopt = None
        self.model = "wav2lip"
        self.l = 10
        self.m = 8
        self.r = 10
        self.modelres = 192

class LiveKitOutput(BaseOutput):
    def __init__(self, opt=None, parent=None, **kwargs):
        super().__init__(opt, parent)
        self.video_queue = asyncio.Queue(maxsize=100)

    def start(self) -> None: pass
    def stop(self) -> None: pass
    def get_buffer_size(self) -> int: return self.video_queue.qsize()
    
    def push_audio_frame(self, frame: np.ndarray, eventpoint=None) -> None: pass

    def push_video_frame(self, frame: np.ndarray) -> None:
        if frame is None: return
        height, width, _ = frame.shape
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        rtc_frame = rtc.VideoFrame(width, height, rtc.VideoBufferType.RGBA, rgba_frame.tobytes())
        try:
            self.video_queue.put_nowait(rtc_frame)
        except asyncio.QueueFull:
            pass

class LiveTalkingVideoGenerator:
    def __init__(self, output: LiveKitOutput, avatar):
        self.output = output
        self.avatar = avatar
        self._running = True
        self._q = asyncio.Queue()
        self._pull_task = asyncio.create_task(self._pull_video())

    async def _pull_video(self):
        try:
            while self._running:
                frame = await self.output.video_queue.get()
                log.info("Worker generated a video frame!")
                await self._q.put(frame)
        except asyncio.CancelledError:
            pass

    async def push_audio(self, frame: rtc.AudioFrame) -> None:
        log.info(f"Worker received an audio frame from Agent! size={len(frame.data)}")
        import numpy as np
        audio_data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32767.0
        self.avatar.put_audio_frame(audio_data)
        await self._q.put(frame)

    async def clear_buffer(self) -> None:
        pass

    async def __anext__(self):
        if not self._running: raise StopAsyncIteration
        try:
            return await self._q.get()
        except asyncio.CancelledError:
            self._running = False
            raise StopAsyncIteration

    def __aiter__(self): return self

async def main():
    url = "wss://synco-4yzh9wr6.livekit.cloud"
    api_key = "APIDeDfGGxAe6Ag"
    api_secret = "mtoRaHtl6OcOm26niGE5ewMBOXokoMVE8YlxDH1ne7p"
    
    if not all([url, api_key, api_secret]):
        log.error("Missing LiveKit credentials in environment variables.")
        return

    # You must connect to the EXACT SAME room that your agent connects to
    # Usually this is passed as a command line argument, but we'll hardcode or read from env
    room_name = "voice_assistant_room"
    
    room = rtc.Room()
    # Create an identity for the avatar that the agent expects to send audio to
    identity = "livetalking-avatar"
    token = AccessToken(api_key, api_secret).with_identity(identity).with_name("Digital Human").with_grants(VideoGrants(room_join=True, room=room_name)).to_jwt()
    
    log.info(f"Connecting to room: {room_name}")
    await room.connect(url, token)
    log.info("Connected!")

    # 1. Load LiveTalking Models
    avatar_id ="wav2lip256_avatar1"
    log.info(f"Loading AI models for {avatar_id}...")
    model_cache = load_model("models/wav2lip.pth")
    avatar_data = load_avatar(avatar_id)
    
    opt = DummyOpt("worker-session", avatar_id)
    output = LiveKitOutput(opt)
    
    avatar = LipReal(opt, model_cache, avatar_data)
    avatar.output = output
    log.info("AI models loaded successfully.")

    # 2. Listen for audio from the Main Agent
    log.info("Setting up DataStream receiver...")
    audio_recv = DataStreamAudioReceiver(room)

    # 3. Publish the lip-synced video back to the room
    video_gen = LiveTalkingVideoGenerator(output, avatar)
    runner = AvatarRunner(
        room=room,
        audio_recv=audio_recv,
        video_gen=video_gen,
        options=AvatarOptions(
            video_width=256,
            video_height=256,
            video_fps=25,
            audio_sample_rate=16000,
            audio_channels=1
        ),
        _lazy_publish=False,
    )
    
    log.info("Starting Video Publisher...")
    await runner.start()
    
    log.info("Worker is fully running! Waiting for agent to send audio...")
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        feed_task.cancel()
        await runner.aclose()
        await room.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
