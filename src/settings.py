from typing import Literal

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(extra="ignore")

    LOG_LEVEL: Literal["DEBUG", "INFO", "ERROR"] = "INFO"
    HOST: str
    PORT: str
    OPENAI_API_KEY: str
    GEMINI_API_KEY: str | None
    OPENROUTER_API_KEY: str | None
    OPENROUTER_BASE_URL: str | None
    LANGSMITH_API_KEY: str | None


settings = Settings(_env_file=".env")
