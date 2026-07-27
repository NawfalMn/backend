from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Union

import numpy as np
from livekit import rtc
from livekit.agents.voice.avatar import VideoGenerator, AudioSegmentEnd

from .ws_client import OmniRTWsClient

log = logging.getLogger(__name__)

# Format VideoFrame LiveKit : BGR24 ou RGBA
# OmniRT retourne BGR numpy → on convertit en RGBA pour LiveKit
LIVEKIT_FRAME_TYPE = rtc.VideoBufferType.RGBA


class OmniRTVideoGenerator(VideoGenerator):
    """
    VideoGenerator qui drive OmniRT comme backend de synthèse lip-sync.
    
    Cycle de vie (géré par AvatarRunner) :
      1. push_audio(AudioFrame) × N   — audio du TTS
      2. push_audio(AudioSegmentEnd)  — fin de segment
      3. __aiter__() yield VideoFrame × N, puis AudioSegmentEnd
      4. clear_buffer()               — si interruption
    """

    def __init__(self, ws_client: OmniRTWsClient) -> None:
        self._ws = ws_client
        # Queue interne : frames générées en attente d'être yield par __aiter__
        self._frame_queue: asyncio.Queue[
            Union[rtc.VideoFrame, rtc.AudioFrame, AudioSegmentEnd]
        ] = asyncio.Queue(maxsize=50)
        self._cleared = asyncio.Event()
        self._pending_audio: list[np.ndarray] = []  # buffer PCM entre segments

    async def push_audio(
        self, frame: rtc.AudioFrame | AudioSegmentEnd
    ) -> None:
        """Reçoit un AudioFrame TTS ou un marqueur de fin de segment."""
        if isinstance(frame, AudioSegmentEnd):
            # Fin de segment → flush le buffer restant + signal EOF
            if self._pending_audio:
                await self._flush_buffer()
            await self._frame_queue.put(AudioSegmentEnd())
            log.debug("OmniRTVideoGenerator: segment end flushed")
            return

        # Accumule PCM int16 16kHz
        pcm = np.frombuffer(frame.data, dtype=np.int16)
        self._pending_audio.append(pcm)

        # Envoie les chunks complets à OmniRT dès qu'on a assez de samples
        chunk_size = self._ws.chunk_samples
        accumulated = np.concatenate(self._pending_audio)

        while len(accumulated) >= chunk_size:
            chunk = accumulated[:chunk_size]
            accumulated = accumulated[chunk_size:]

            if self._cleared.is_set():
                break

            bgr_frames = await self._ws.generate(chunk)
            for bgr in bgr_frames:
                vf = self._bgr_to_livekit_frame(bgr)
                await self._frame_queue.put(vf)

        self._pending_audio = [accumulated] if len(accumulated) > 0 else []

    async def _flush_buffer(self) -> None:
        """Envoie le dernier chunk partiel (padded) à OmniRT."""
        accumulated = np.concatenate(self._pending_audio)
        self._pending_audio = []
        if len(accumulated) == 0:
            return
        chunk_size = self._ws.chunk_samples
        if len(accumulated) < chunk_size:
            accumulated = np.pad(accumulated, (0, chunk_size - len(accumulated)))
        bgr_frames = await self._ws.generate(accumulated[:chunk_size])
        for bgr in bgr_frames:
            await self._frame_queue.put(self._bgr_to_livekit_frame(bgr))

    def clear_buffer(self) -> None:
        """Appelé par AvatarRunner lors d'une interruption utilisateur."""
        self._cleared.set()
        self._pending_audio.clear()
        # Vide la queue de frames
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._cleared.clear()
        log.info("OmniRTVideoGenerator: buffer cleared (interruption)")

    def __aiter__(
        self,
    ) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        return self._frame_iterator()

    async def _frame_iterator(
        self,
    ) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        while True:
            item = await self._frame_queue.get()
            yield item
            if isinstance(item, AudioSegmentEnd):
                break

    @staticmethod
    def _bgr_to_livekit_frame(bgr: np.ndarray) -> rtc.VideoFrame:
        """Convertit un numpy BGR (H,W,3) en rtc.VideoFrame RGBA."""
        rgba = np.dstack([
            bgr[:, :, 2],  # R
            bgr[:, :, 1],  # G
            bgr[:, :, 0],  # B
            np.full(bgr.shape[:2], 255, dtype=np.uint8),  # A
        ])
        h, w = bgr.shape[:2]
        return rtc.VideoFrame(
            width=w,
            height=h,
            type=LIVEKIT_FRAME_TYPE,
            data=rgba.tobytes(),
        )
