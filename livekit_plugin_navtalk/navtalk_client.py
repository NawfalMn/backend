import asyncio
import base64
import json
import logging
from urllib.parse import urlparse
import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger("navtalk.client")

class NavTalkClient:
    def __init__(self, license_key: str, avatar_id: str, ws_url: str):
        self.license_key = license_key
        self.avatar_id = avatar_id
        self.ws_url = ws_url
        self.ws = None
        self.reader_task = None
        
        # Callbacks for WebRTC signaling
        self.on_offer = None
        self.on_ice_candidate = None
        self.on_connected = None
        
        # Ensure only one connection at a time
        self.lock = asyncio.Lock()

    async def connect(self):
        async with self.lock:
            if self.ws is not None:
                return
            
            separator = "&" if "?" in self.ws_url else "?"
            url = f"{self.ws_url}{separator}license={self.license_key}&avatarId={self.avatar_id}&audioBack=true"
            
            logger.info(f"Connecting to NavTalk WS: {urlparse(self.ws_url).netloc}")
            self.ws = await websockets.connect(url, ping_interval=20, ping_timeout=20)
            self.reader_task = asyncio.create_task(self._read_loop())
            logger.info("Connected to NavTalk successfully.")

    async def disconnect(self):
        async with self.lock:
            if self.ws:
                await self.ws.close()
                self.ws = None
            if self.reader_task:
                self.reader_task.cancel()
                try:
                    await self.reader_task
                except asyncio.CancelledError:
                    pass
                self.reader_task = None

    async def _read_loop(self):
        try:
            async for raw in self.ws:
                if isinstance(raw, str):
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON received from NavTalk.")
                        continue
                    await self._handle_event(event)
        except ConnectionClosed:
            logger.warning("NavTalk WebSocket closed.")
        except Exception as e:
            logger.error(f"NavTalk reader error: {e}")
        finally:
            self.ws = None

    async def _handle_event(self, event: dict):
        event_type = event.get("type", "")
        data = event.get("data", {})
        
        if event_type == "conversation.connected.success":
            logger.info("NavTalk conversation connected.")
            if self.on_connected:
                asyncio.create_task(self.on_connected(data))
        elif event_type == "webrtc.signaling.offer":
            sdp = data.get("sdp")
            if isinstance(sdp, dict):
                sdp = sdp.get("sdp")
            if sdp and self.on_offer:
                asyncio.create_task(self.on_offer(sdp))
        elif event_type == "webrtc.signaling.iceCandidate":
            candidate = data.get("candidate")
            if candidate and self.on_ice_candidate:
                asyncio.create_task(self.on_ice_candidate(candidate))
        elif "error" in event_type or event_type in ["conversation.connected.fail", "conversation.connected.close"]:
            logger.error(f"NavTalk error event: {event}")

    async def send_answer(self, sdp: str):
        if not self.ws:
            return
        payload = {
            "type": "webrtc.signaling.answer",
            "data": {"sdp": {"type": "answer", "sdp": sdp}}
        }
        await self.ws.send(json.dumps(payload))

    async def send_ice_candidate(self, candidate: dict):
        if not self.ws:
            return
        payload = {
            "type": "webrtc.signaling.iceCandidate",
            "data": {"candidate": candidate}
        }
        await self.ws.send(json.dumps(payload))

    async def send_audio(self, pcm_bytes: bytes):
        if not self.ws:
            return
        
        encoded = base64.b64encode(pcm_bytes).decode("ascii")
        payload = {
            "type": "input_audio_buffer.append",
            "audio": encoded
        }
        await self.ws.send(json.dumps(payload))

    async def clear_buffer(self):
        if not self.ws:
            return
        
        payload = {
            "type": "input_audio_buffer.clear"
        }
        await self.ws.send(json.dumps(payload))
