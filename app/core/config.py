import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    # 1. Cấu hình chung cho Server FastAPI
    PROJECT_NAME: str = "AI AR Campus Explorer - HCMUS"
    API_V1_STR: str = "/api/v1"
    
    # 2. Cấu hình bảo mật CORS (Dành cho Web Frontend kết nối vào)
    # Cho phép các địa chỉ IP/Domain này gọi API tới Backend
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",    # React mặc định
        "http://localhost:5173",    # Vite / Vue mặc định
        "http://127.0.0.1:5500",    # Live Server của VS Code (Dành cho HTML/JS thuần)
    ]

    # 3. Cấu hình kết nối Cơ sở dữ liệu (SQLite cho Campus)
    DATABASE_URL: str = "sqlite:///./campus_explorer.db"

    # 4. Chìa khóa bí mật gọi tới AI (Đọc tự động từ file .env)
    GEMINI_API_KEY: str = ""
    # Nếu nhóm AI dùng OpenAI/ChatGPT thì bỏ dấu thăng dòng dưới ra:
    # OPENAI_API_KEY: str = ""

    # Cấu hình để Pydantic tự động quét và nạp dữ liệu từ file .env bên ngoài vào
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore" # Bỏ qua nếu trong file .env có dư biến
    )

# Khởi tạo một đối tượng settings duy nhất để toàn bộ dự án gọi dùng chung
settings = Settings()