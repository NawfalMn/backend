import asyncio
import logging
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, RTCConfiguration, RTCIceServer
import livekit.rtc as rtc

logger = logging.getLogger("navtalk.avatar")

class NavTalkVideoGenerator:
    def __init__(self, client):
        self._client = client
        self._client.on_offer = self._on_offer
        self._client.on_ice_candidate = self._on_ice_candidate
        
        self.pc = RTCPeerConnection(
            configuration=RTCConfiguration(
                iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
            )
        )
        self._q = asyncio.Queue()
        self._running = True
        
        self._sent_audio_duration = 0.0
        self._received_audio_duration = 0.0
        self._pending_end_frame = None
        
        self._video_task = None
        self._audio_task = None
        
        @self.pc.on("track")
        def on_track(track):
            logger.info(f"NavTalk WebRTC track received: {track.kind}")
            if track.kind == "video":
                self._video_task = asyncio.create_task(self._process_video_track(track))
            elif track.kind == "audio":
                self._audio_task = asyncio.create_task(self._process_audio_track(track))

    async def _on_offer(self, sdp):
        logger.info("Received offer from NavTalk.")
        if isinstance(sdp, dict):
            sdp_str = sdp.get("sdp")
        else:
            sdp_str = sdp
        desc = RTCSessionDescription(sdp=sdp_str, type="offer")
        await self.pc.setRemoteDescription(desc)
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        await self._client.send_answer(self.pc.localDescription.sdp)
        logger.info("Sent answer to NavTalk.")

    async def _on_ice_candidate(self, candidate_dict: dict):
        # aiortc handles ICE candidates via setRemoteDescription usually,
        # but if trickle ICE is used, we need to add them manually.
        # candidate_dict format usually matches standard WebRTC candidate
        sdpMid = candidate_dict.get("sdpMid")
        sdpMLineIndex = candidate_dict.get("sdpMLineIndex")
        candidate_str = candidate_dict.get("candidate")
        
        if candidate_str:
            candidate = RTCIceCandidate(
                component=1, # typically 1 for RTP
                foundation=candidate_str.split()[0],
                ip=candidate_str.split()[4],
                port=int(candidate_str.split()[5]),
                priority=int(candidate_str.split()[3]),
                protocol=candidate_str.split()[2],
                type=candidate_str.split()[7],
                sdpMid=sdpMid,
                sdpMLineIndex=sdpMLineIndex
            )
            # Aiortc addIceCandidate support is basic, but we try:
            try:
                await self.pc.addIceCandidate(candidate)
            except Exception as e:
                logger.warning(f"Failed to add ICE candidate: {e}")

    async def _process_video_track(self, track):
        first_frame = True
        while self._running:
            try:
                frame = await track.recv() # PyAV VideoFrame
                if first_frame:
                    logger.info("Received first VIDEO frame from NavTalk!")
                    first_frame = False
                # Convert to RGBA for LiveKit
                img = frame.to_ndarray(format="rgba")
                height, width, _ = img.shape
                rtc_frame = rtc.VideoFrame(width, height, rtc.VideoBufferType.RGBA, img.tobytes())
                await self._q.put(rtc_frame)
            except Exception as e:
                logger.error(f"Video track processing error: {e}")
                break

    async def _process_audio_track(self, track):
        first_frame = True
        while self._running:
            try:
                frame = await track.recv() # PyAV AudioFrame
                if first_frame:
                    logger.info(f"Received first AUDIO frame from NavTalk! rate={frame.sample_rate}, channels={len(frame.layout.channels)}")
                    first_frame = False
                
                duration = frame.samples / frame.sample_rate
                self._received_audio_duration += duration

                data = bytes(frame.planes[0])
                rtc_frame = rtc.AudioFrame(
                    data=data,
                    sample_rate=frame.sample_rate,
                    num_channels=len(frame.layout.channels),
                    samples_per_channel=frame.samples
                )
                await self._q.put(rtc_frame)
                
                # Yield the pending AudioSegmentEnd if we've received enough audio back
                if self._pending_end_frame is not None and self._received_audio_duration >= self._sent_audio_duration - 0.5:
                    await self._q.put(self._pending_end_frame)
                    self._pending_end_frame = None
                    self._sent_audio_duration = 0.0
                    self._received_audio_duration = 0.0
                    
            except Exception as e:
                logger.error(f"Audio track processing error: {e}")
                break

    async def push_audio(self, frame) -> None:
        """Called by LiveKit Agent SDK to send audio to the avatar."""
        if not self._running:
            return
        
        if not isinstance(frame, rtc.AudioFrame):
            # It's likely an AudioSegmentEnd or similar control frame
            self._pending_end_frame = frame
            return
            
        duration = frame.samples_per_channel / frame.sample_rate
        self._sent_audio_duration += duration
        
        # We forward the PCM bytes to NavTalk WS.
        # Ensure we only send if connected.
        data_bytes = bytes(frame.data)
        await self._client.send_audio(data_bytes)

    async def clear_buffer(self) -> None:
        """Called when the agent gets interrupted by the user."""
        self._sent_audio_duration = 0.0
        self._received_audio_duration = 0.0
        self._pending_end_frame = None
        
        # Clear the internal frame queue
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except asyncio.QueueEmpty:
                break
                
        # Tell the NavTalk backend to clear its audio buffer
        await self._client.clear_buffer()
        logger.info("NavTalk Video Generator buffer cleared.")

    async def __anext__(self):
        if not self._running:
            raise StopAsyncIteration
        try:
            return await self._q.get()
        except asyncio.CancelledError:
            self._running = False
            raise StopAsyncIteration

    def __aiter__(self):
        return self

    async def aclose(self):
        self._running = False
        await self.pc.close()
        if self._video_task:
            self._video_task.cancel()
        if self._audio_task:
            self._audio_task.cancel()
