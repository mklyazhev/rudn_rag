from pydantic import SecretStr
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    telegram_bot_token: SecretStr
    openai_api_key: SecretStr

    embeddings: str
    llm: str

    documents: Path
    logfile: Path

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


config = Settings()
