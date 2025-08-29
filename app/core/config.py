import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

load_dotenv()

class Settings(BaseSettings):
    OPENAI_API_KEY: str = Field(...,env = "OPENAI_API_KEY")

    class Config:
        env_file = ".env"

settings = Settings()
