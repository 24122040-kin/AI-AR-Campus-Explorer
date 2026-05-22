import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# Tải cấu hình từ tệp .env
load_dotenv()

# Lấy đường dẫn cơ sở dữ liệu từ tệp .env
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

# Khởi tạo 'engine' (động cơ cốt lõi giao tiếp với database)
# connect_args={"check_same_thread": False} là cấu hình bắt buộc khi dùng SQLite với FastAPI
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Tạo một 'nhà máy' sản xuất các phiên (session) làm việc với database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base là một lớp nền tảng, khi tạo các bảng dữ liệu ta sẽ kế thừa từ lớp Base này
Base = declarative_base()

# Hàm hỗ trợ để FastAPI lấy kết nối database mỗi khi có yêu cầu (request) tới
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()