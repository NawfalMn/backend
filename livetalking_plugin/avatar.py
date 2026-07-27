from __future__ import annotations

import logging
import sys
import os
from pathlib import Path
import asyncio
from typing import AsyncIterable

from livekit import rtc
from livekit.agents import AgentSession
from livekit.agents.voice.avatar import (
    AvatarRunner,
    QueueAudioOutput,
    AvatarSession as BaseAvatarSession,
    AvatarOptions
)

# Inject LiveTalking path so we can import its modules
LIVETALKING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../LiveTalking"))
if LIVETALKING_DIR not in sys.path:
    sys.path.insert(0, LIVETALKING_DIR)

# Now we can import LiveTalking classes
from LiveTalking.avatars.musetalk_avatar import MuseReal
from LiveTalking.utils.logger import logger as lt_logger
from .livekit_output import LiveKitOutput

log = logging.getLogger(__name__)

# Dummy opt class for LiveTalking initialization
class DummyOpt:
    def __init__(self, sessionid, avatar_id):
        self.sessionid = sessionid
        self.avatar_id = avatar_id
        self.fps = 25
        self.batch_size = 4
        self.transport = "livekit" # We'll override this
        self.tts = "none"
        self.customopt = None
        self.model = "musetalk"

class LiveTalkingVideoGenerator:
    """Consumes the LiveKitOutput queue and yields rtc.VideoFrame"""
    def __init__(self, output: LiveKitOutput, avatar=None):
        self.output = output
        self.avatar = avatar
        self._running = True

    async def clear_buffer(self) -> None:
        """Called when the agent gets interrupted by the user."""
        # Empty the internal video queue
        while not self.output.video_queue.empty():
            try:
                self.output.video_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
                
        # Flush the avatar's internal buffers
        if self.avatar and hasattr(self.avatar, 'flush_talk'):
            self.avatar.flush_talk()
            
        log.info("LiveTalkingVideoGenerator: buffer cleared (interruption)")

    async def __anext__(self) -> rtc.VideoFrame:
        if not self._running:
            raise StopAsyncIteration
        try:
            frame = await self.output.video_queue.get()
            return frame
        except asyncio.CancelledError:
            self._running = False
            raise StopAsyncIteration

    def __aiter__(self):
        return self


class LiveTalkingAvatarSession(BaseAvatarSession):
    def __init__(
        self,
        avatar_id: str,
        session_id: str = "livekit-session"
    ) -> None:
        super().__init__()
        self._avatar_id = avatar_id
        self._session_id = session_id
        
        self._output: LiveKitOutput | None = None
        self._runner: AvatarRunner | None = None
        self._audio_output: QueueAudioOutput | None = None
        self._avatar = None
        
        # Load the model directly here or assume it's pre-loaded?
        # LiveTalking musetalk_avatar requires the models loaded.
        # Let's import load_model and load_avatar
        from LiveTalking.avatars.musetalk_avatar import load_model, load_avatar
        
        log.info("Loading MuseTalk models...")
        self._model_cache = load_model()
        log.info(f"Loading avatar data for {avatar_id}...")
        self._avatar_data = load_avatar(avatar_id)


    @property
    def avatar_identity(self) -> str:
        return "livetalking-avatar"

    @property
    def provider(self) -> str:
        return "livetalking"

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
    ) -> None:
        await super().start(agent_session, room)
        
        # 1. Setup LiveTalking Opt & Output
        opt = DummyOpt(self._session_id, self._avatar_id)
        self._output = LiveKitOutput(opt)
        
        # 2. Initialize Avatar
        self._avatar = MuseReal(opt, self._model_cache, self._avatar_data)
        self._avatar.output = self._output  # Override output transport
        
        # Start LiveTalking internal threads/async loops if any
        # Actually in LiveTalking, reading audio into self.asr starts processing
        
        # 3. Audio queue to feed into LiveTalking avatar.put_audio_frame
        self._audio_output = QueueAudioOutput(sample_rate=16000)
        agent_session.output.replace_audio_tail(self._audio_output)
        
        # 4. Background task to read audio from livekit and feed to LiveTalking
        self._audio_feed_task = asyncio.create_task(self._feed_audio_loop())

        # 5. Start AvatarRunner for LiveKit
        self._video_gen = LiveTalkingVideoGenerator(self._output, avatar=self._avatar)
        self._runner = AvatarRunner(
            room=room,
            audio_recv=self._audio_output,
            video_gen=self._video_gen,
            options=AvatarOptions(
                video_width=256, # MuseTalk default
                video_height=256,
                video_fps=25,
                audio_sample_rate=16000,
                audio_channels=1
            ),
            _queue_size_ms=5000,
            _lazy_publish=False,
        )
        await self._runner.start()
        log.info(f"LiveTalkingAvatarSession started for avatar {self._avatar_id}")

    async def _feed_audio_loop(self):
        """Consume audio from LiveKit and pass to LiveTalking"""
        try:
            async for audio_chunk in self._audio_output:
                # audio_chunk is rtc.AudioFrame
                import numpy as np
                # Convert rtc.AudioFrame to numpy float32 [-1, 1] as LiveTalking expects
                audio_data = np.frombuffer(audio_chunk.data, dtype=np.int16).astype(np.float32) / 32767.0
                if self._avatar:
                    self._avatar.put_audio_frame(audio_data)
        except asyncio.CancelledError:
            pass

    async def aclose(self) -> None:
        if self._runner is not None:
            await self._runner.aclose()
        if self._audio_feed_task:
            self._audio_feed_task.cancel()
        await super().aclose()
