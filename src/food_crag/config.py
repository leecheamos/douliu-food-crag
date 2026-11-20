"""Environment configuration. Never expose credentials in status responses."""
import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8000
    gemini_api_key: str = field(default="", repr=False)
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_timeout: float = 20.0
    embedding_model: str = ""
    line_channel_secret: str = field(default="", repr=False)
    line_channel_access_token: str = field(default="", repr=False)
    chat_api_enabled: bool = True  # /api/chat is a local testing entry; disable when public.

    @classmethod
    def from_env(cls, env_file=None) -> "Settings":
        values = {}
        if env_file is not None:
            for number, line in enumerate(Path(env_file).read_text(encoding="utf-8-sig").splitlines(), 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    raise ValueError(f"Invalid env file line {number}; expected KEY=value")
                key, value = line.split("=", 1)
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                values[key.strip()] = value
        values.update(os.environ)
        port = int(values.get("APP_PORT", "8000"))
        if not 1 <= port <= 65535:
            raise ValueError("APP_PORT must be between 1 and 65535")
        chat_api = values.get("CHAT_API_ENABLED", "true").strip().lower()
        if chat_api not in {"true", "false", "1", "0", ""}:
            raise ValueError("CHAT_API_ENABLED must be true or false")
        timeout = float(values.get("GEMINI_TIMEOUT", "20") or "20")
        if not 0 < timeout <= 60:
            raise ValueError("GEMINI_TIMEOUT must be greater than 0 and at most 60")
        return cls(
            host=values.get("APP_HOST", "127.0.0.1"), port=port,
            gemini_api_key=values.get("GEMINI_API_KEY", "").strip(),
            gemini_model=values.get("GEMINI_MODEL", "").strip() or "gemini-3.5-flash-lite",
            gemini_timeout=timeout,
            embedding_model=values.get("EMBEDDING_MODEL", "").strip(),
            line_channel_secret=values.get("LINE_CHANNEL_SECRET", ""),
            line_channel_access_token=values.get("LINE_CHANNEL_ACCESS_TOKEN", ""),
            chat_api_enabled=chat_api in {"true", "1", ""},
        )
