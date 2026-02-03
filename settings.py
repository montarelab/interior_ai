from typing import Literal

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(extra="ignore")

    LOG_LEVEL: Literal["DEBUG", "INFO", "ERROR"] = "INFO"
    OPENAI_API_KEY: str
    OPENROUTER_API_KEY: str
    OPENROUTER_BASE_URL: str
    LANGSMITH_API_KEY: str


settings = Settings(_env_file=".env")
