from typing import TYPE_CHECKING, Optional
import numpy as np
import cv2
import asyncio
from livekit import rtc

from LiveTalking.streamout.base_output import BaseOutput

if TYPE_CHECKING:
    from LiveTalking.avatars.base_avatar import BaseAvatar


class LiveKitOutput(BaseOutput):
    """
    Adapter bridging LiveTalking's BaseOutput to LiveKit.
    It takes frames (OpenCV BGR arrays) from the Avatar engine, converts them to rtc.VideoFrame,
    and pushes them into an asyncio.Queue that the LiveKit VideoGenerator consumes.
    """

    def __init__(self, opt=None, parent: Optional['BaseAvatar'] = None, **kwargs):
        super().__init__(opt, parent)
        self.video_queue = asyncio.Queue(maxsize=100)

    def start(self) -> None:
        pass

    def push_video_frame(self, frame: np.ndarray) -> None:
        """
        frame: OpenCV BGR numpy array
        We convert it to ARGB and push to the queue.
        """
        # Convert BGR to RGBA/ARGB as LiveKit expects
        # rtc.VideoFrame allows RGBA. Wait, LiveKit python rtc.VideoFrame
        # usually prefers ARGB or I420. Let's use ARGB.
        # OpenCV converts BGR to BGRA, then we can swap channels if needed,
        # but cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) gives RGBA.
        # rtc.VideoBufferType.RGBA exists!
        if frame is None:
            return
            
        height, width, _ = frame.shape
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        
        rtc_frame = rtc.VideoFrame(
            width, 
            height, 
            rtc.VideoBufferType.RGBA, 
            rgba_frame.tobytes()
        )
        
        try:
            self.video_queue.put_nowait(rtc_frame)
        except asyncio.QueueFull:
            pass # Drop frame if generator is too slow

    def push_audio_frame(self, frame: np.ndarray, eventpoint=None) -> None:
        """
        LiveTalking avatar engines also emit audio chunks.
        Since LiveKit's TTS manages audio in LiveKit avatars, we may just ignore this,
        or push it if we need lip-sync to rely on this output audio instead of LiveKit TTS direct output.
        """
        pass

    def get_buffer_size(self) -> int:
        return self.video_queue.qsize()

    def stop(self) -> None:
        pass
