"""Settings loaded from environment / .env. The API key never leaves the server."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _bool(name: str, default: bool = False) -> bool:
    v = _env(name)
    if not v:
        return default
    return v.lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: _env("NEXUS_API_KEY"), repr=False)
    base_url: str = field(default_factory=lambda: _env("NEXUS_BASE_URL", "http://localhost/v1"))
    llm_model: str = field(default_factory=lambda: _env("NEXUS_LLM_MODEL", "model-a"))
    llm_model_b: str = field(default_factory=lambda: _env("NEXUS_LLM_MODEL_B", "model-b"))
    vision_model: str = field(default_factory=lambda: _env("NEXUS_VISION_MODEL", "vision"))
    stt_model: str = field(default_factory=lambda: _env("NEXUS_STT_MODEL", "stt"))
    tts_model: str = field(default_factory=lambda: _env("NEXUS_TTS_MODEL", "tts"))
    tts_voice: str = field(default_factory=lambda: _env("NEXUS_TTS_VOICE", "alloy"))
    json_mode: bool = field(default_factory=lambda: _bool("NEXUS_JSON_MODE", True))
    timeout: float = field(default_factory=lambda: float(_env("NEXUS_TIMEOUT", "60")))
    mock_flag: bool = field(default_factory=lambda: _bool("NEXUS_MOCK", False))

    data_dir: Path = field(default_factory=lambda: Path(_env("FORWARDCHECK_DATA", str(ROOT / "data"))))
    source_dir: Path = field(default_factory=lambda: Path(_env("FORWARDCHECK_SOURCES", str(ROOT / "source_pack"))))
    db_path: str = field(default_factory=lambda: _env("FORWARDCHECK_DB", ""))

    max_upload_mb: int = 10
    min_image_chars: int = 20
    max_claims: int = 8
    reply_max_words: int = 90      # ~2.3 words/s spoken -> under 45 s
    reply_retry_words: int = 60
    max_audio_seconds: float = 45.0

    def __post_init__(self) -> None:
        if not self.db_path:
            self.db_path = str(self.data_dir / "forwardcheck.db")

    @property
    def mock(self) -> bool:
        return self.mock_flag or not self.api_key

    def resolve_model(self, choice: str | None) -> str:
        """Map 'A'/'B' (used by the UI and eval) to the configured Nexus LLM name."""
        if choice in (None, "", "A"):
            return self.llm_model
        if choice == "B":
            return self.llm_model_b
        raise ValueError("model must be 'A' or 'B'")
