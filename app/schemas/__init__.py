from pydantic import BaseModel, EmailStr, Field

# Khuôn mẫu dữ liệu khi người dùng gửi yêu cầu ĐĂNG KÝ
class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str = Field(..., max_length=70) # Giới hạn độ dài mật khẩu dưới 72 ký tự để tương thích với bcrypt

# Khuôn mẫu dữ liệu TRẢ VỀ sau khi tạo thành công (tuyệt đối KHÔNG trả về mật khẩu)
class UserResponse(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    is_active: bool

    # Cấu hình này giúp Pydantic hiểu được dữ liệu dạng Object của SQLAlchemy
    class Config:
        from_attributes = True

# Khuôn mẫu khi người dùng gửi yêu cầu ĐĂNG NHẬP
class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(..., max_length=70)

# Khuôn mẫu trả về cho ứng dụng Mobile/Frontend chứa Token
class Token(BaseModel):
    access_token: str
    token_type: str

# Khuôn mẫu khi Admin gửi yêu cầu THÊM địa điểm mới
class LocationCreate(BaseModel):
    name: str
    description: str
    latitude: float
    longitude: float
    is_ar_active: bool = True

# Khuôn mẫu TRẢ VỀ thông tin địa điểm
class LocationResponse(BaseModel):
    id: int
    name: str
    description: str
    latitude: float
    longitude: float
    is_ar_active: bool

    class Config:
        from_attributes = True