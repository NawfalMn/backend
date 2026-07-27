from __future__ import annotations

import logging
from pathlib import Path

from livekit import api, rtc
from livekit.agents import AgentSession
from livekit.agents.voice.avatar import (
    AvatarRunner,
    QueueAudioOutput,
    ATTRIBUTE_PUBLISH_ON_BEHALF,
)

from .ws_client import OmniRTWsClient
from .video_generator import OmniRTVideoGenerator

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000          # OmniRT MuseTalk travaille en 16kHz
AVATAR_IDENTITY = "omnirt-avatar-worker"


class OmniRTAvatarSession:
    """
    Plugin Avatar LiveKit pour OmniRT (MuseTalk / QuickTalk / Wav2Lip).
    
    Suit le même pattern que les autres providers :
      - QueueAudioOutput  : bus audio in-process (pas DataStream — tout est local)
      - AvatarRunner      : orchestre audio→frames + AVSynchronizer
      - OmniRTVideoGenerator : génère les frames via OmniRT WS
    """

    def __init__(
        self,
        omnirt_endpoint: str,           # "ws://localhost:9000"
        model: str = "musetalk",        # "musetalk" | "quicktalk" | "wav2lip"
        ref_image: bytes | str | Path | None = None,  # portrait PNG/JPEG
        api_key: str | None = None,     # OMNIRT_API_KEY optionnel
        avatar_identity: str = AVATAR_IDENTITY,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self._endpoint = omnirt_endpoint
        self._model = model
        self._ref_image = ref_image
        self._api_key = api_key
        self._avatar_identity = avatar_identity
        self._sample_rate = sample_rate

        self._ws_client: OmniRTWsClient | None = None
        self._video_gen: OmniRTVideoGenerator | None = None
        self._runner: AvatarRunner | None = None
        self._audio_output: QueueAudioOutput | None = None

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        livekit_api_key: str | None = None,
        livekit_api_secret: str | None = None,
    ) -> None:
        """Démarre la session avatar. Appelle avant session.start()."""

        # 1. Connexion à OmniRT et initialisation de la session avatar
        self._ws_client = OmniRTWsClient(
            omnirt_endpoint=self._endpoint,
            model=self._model,
            api_key=self._api_key,
        )
        await self._ws_client.connect()
        await self._ws_client.init_session(ref_image=self._ref_image)

        # 2. Crée le VideoGenerator wrappant OmniRT
        self._video_gen = OmniRTVideoGenerator(self._ws_client)

        # 3. QueueAudioOutput : bus audio in-process
        #    (alternative locale à DataStreamAudioOutput qui est pour le cloud)
        self._audio_output = QueueAudioOutput(sample_rate=self._sample_rate)

        # 4. Remplace la sortie audio de l'agent par notre queue
        agent_session.output.replace_audio_tail(self._audio_output)

        # 5. Lance l'AvatarRunner qui orchestre tout
        #    AvatarRunner: audio_recv → VideoGenerator → AVSynchronizer → publish_track
        self._runner = AvatarRunner(
            room=room,
            audio_recv=self._audio_output,       # lit depuis la queue
            video_gen=self._video_gen,            # notre générateur OmniRT
            publish_on_behalf_of=room.local_participant.identity,
        )
        await self._runner.start()
        log.info(
            "OmniRTAvatarSession started: model=%s endpoint=%s",
            self._model, self._endpoint,
        )

    async def wait_for_join(self, timeout: float = 30.0) -> None:
        """Attend que la video track soit publiée (optionnel mais recommandé)."""
        if self._runner is not None:
            await self._runner.wait_for_first_frame(timeout=timeout)

    async def aclose(self) -> None:
        if self._runner is not None:
            await self._runner.aclose()
        if self._ws_client is not None:
            await self._ws_client.close()
