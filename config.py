from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    LINUX_NPY_DIR: str
    LINUX_JSON_DIR: str
    BASE_LLM_URL: str
    CLIP_MODEL: str

    model_config = SettingsConfigDict(
        env_file=".env",          # Tên file chứa biến môi trường
        env_file_encoding="utf-8",
        extra="ignore"            # Bỏ qua nếu trong .env có thừa biến nào đó không khai báo ở trên
    )

config = Settings()