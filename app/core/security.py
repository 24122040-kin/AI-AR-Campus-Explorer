import os
import jwt
import bcrypt
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Tải cấu hình
load_dotenv()

# Lấy chìa khóa bí mật từ file .env
SECRET_KEY = os.getenv("SECRET_KEY", "chuoi-du-phong")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # Thẻ này sẽ hết hạn sau 7 ngày

# Hàm nhận vào mật khẩu gốc và trả về chuỗi đã được mã hoá
def get_password_hash(password: str) -> str:
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(pwd_bytes, salt)
    # Trả về dạng chuỗi (str) để lưu vào database
    return hashed_password.decode('utf-8')

# Hàm kiểm tra mật khẩu nhập vào có khớp với chuỗi mã hoá trong DB hay không
def verify_password(plain_password: str, hashed_password: str) -> bool:
    password_byte_enc = plain_password.encode('utf-8')
    hashed_password_byte_enc = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_byte_enc, hashed_password_byte_enc)

# Hàm tạo "Thẻ ra vào" (Token)
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt