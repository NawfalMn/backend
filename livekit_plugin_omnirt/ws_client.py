from __future__ import annotations

import asyncio
import base64
import json
import logging
import struct
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from websockets.asyncio.client import connect as ws_connect

log = logging.getLogger(__name__)

MAGIC_AUDIO = b"AUDI"
MAGIC_VIDEO = b"VIDX"


class OmniRTWsClient:
    def __init__(
        self,
        omnirt_endpoint: str,     # "http://localhost:9000" ou "ws://..."
        model: str = "musetalk",  # "musetalk" | "quicktalk" | "wav2lip"
        api_key: str | None = None,
    ) -> None:
        ep = omnirt_endpoint.replace("http://", "ws://").replace("https://", "wss://")
        self.ws_url = f"{ep.rstrip('/')}/v1/audio2video/{model}"
        self._api_key = api_key
        self._ws = None

        # Renseigné après init_session
        self.fps: int = 25
        self.height: int = 512
        self.width: int = 512
        self.sample_rate: int = 16000
        self.chunk_samples: int = 0    # samples par chunk audio
        self.slice_len: int = 0        # frames vidéo par chunk

    async def connect(self) -> None:
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        self._ws = await ws_connect(
            self.ws_url,
            max_size=50 * 1024 * 1024,
            open_timeout=30,
            ping_interval=20,
            ping_timeout=180,
            additional_headers=headers or None,
        )
        log.info("OmniRT connected: %s", self.ws_url)

    async def init_session(
        self,
        ref_image: bytes | str | Path,
        seed: int = 9999,
        prompt: str = "A person is talking.",
        **extra: Any,
    ) -> dict:
        if isinstance(ref_image, (str, Path)):
            ref_image = Path(ref_image).read_bytes()

        payload: dict[str, Any] = {
            "type": "init",
            "ref_image": base64.b64encode(ref_image).decode(),
            "seed": seed,
            "prompt": prompt,
        }
        payload.update(extra)  # video_config, reference_mode, etc.

        await self._ws.send(json.dumps(payload))
        resp = json.loads(await self._ws.recv())

        if resp.get("type") == "error":
            raise RuntimeError(f"OmniRT init failed: {resp.get('message')}")

        self.fps = int(resp.get("fps", 25))
        self.height = int(resp.get("height", 512))
        self.width = int(resp.get("width", 512))
        self.sample_rate = int(resp.get("sample_rate", 16000))
        self.slice_len = int(resp.get("slice_len", 0))
        self.chunk_samples = int(
            resp.get("chunk_samples")
            or (self.slice_len * self.sample_rate // max(1, self.fps))
        )
        log.info(
            "OmniRT init OK: %dx%d @%dfps chunk_samples=%d",
            self.width, self.height, self.fps, self.chunk_samples,
        )
        return resp

    async def generate(self, audio_pcm: np.ndarray) -> list[np.ndarray]:
        """
        Envoie audio int16 PCM → reçoit frames BGR numpy (H,W,3).
        audio_pcm doit contenir exactement chunk_samples samples.
        """
        pcm = np.asarray(audio_pcm, dtype=np.int16)
        await self._ws.send(MAGIC_AUDIO + pcm.tobytes())

        resp = await self._ws.recv()
        if isinstance(resp, str):
            msg = json.loads(resp)
            raise RuntimeError(f"OmniRT error: {msg.get('message')}")
        if len(resp) < 8 or resp[:4] != MAGIC_VIDEO:
            raise RuntimeError(f"Bad magic: {resp[:4]!r}")

        n = struct.unpack("<I", resp[4:8])[0]
        offset, bgr_frames = 8, []
        for _ in range(n):
            jlen = struct.unpack("<I", resp[offset:offset + 4])[0]
            offset += 4
            jpeg_bytes = resp[offset:offset + jlen]
            offset += jlen
            buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is not None:
                bgr_frames.append(bgr)
        return bgr_frames

    async def close(self) -> None:
        if self._ws is None:
            return
        try:
            await self._ws.send(json.dumps({"type": "close"}))
            await asyncio.wait_for(self._ws.recv(), timeout=3.0)
        except Exception:
            pass
        try:
            await self._ws.close()
        except Exception:
            pass
        self._ws = None
