from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    agent_workdir: str = "../workspace"
    ollama_host: str = "http://localhost:11434"
    model_name: str = "qwen2.5-coder:7b"
    frontend_origin: str = "http://localhost:5173"

    max_tool_iterations: int = 16
    shell_timeout_seconds: int = 120
    max_tool_output_chars: int = 8000
    max_read_file_chars: int = 60000

    @property
    def workdir_path(self) -> Path:
        path = Path(self.agent_workdir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
