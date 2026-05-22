from sqlalchemy import Column, Integer, String, Boolean, Float
from app.db.session import Base

class User(Base):
    """
    Model này đại diện cho bảng 'users' trong cơ sở dữ liệu
    Nó sẽ lưu thông tin cơ bản của sinh viên sử dụng ứng dụng AR
    """
    __tablename__ = "users"

    # ID tự tăng, là khóa chính (Primary Key)
    id = Column(Integer, primary_key=True, index=True)
    
    # Tên đầy đủ của sinh viên
    full_name = Column(String, index=True)
    
    # Email sinh viên (ví dụ: @student.hcmus.edu.vn), dùng để đăng nhập
    email = Column(String, unique=True, index=True, nullable=False)
    
    # Mật khẩu đã được mã hóa (hashed_password)
    hashed_password = Column(String, nullable=False)
    
    # Trạng thái tài khoản
    is_active = Column(Boolean, default=True)

    # Có thể thêm các trường khác như MSSV, Khoa... tại đây

class Location(Base):
    """
    Model này đại diện cho bảng 'locations'
    Lưu trữ thông tin các tòa nhà, phòng lab hoặc điểm AR trong trường HCMUS
    """
    __tablename__ = "locations"

    id = Column(Integer, primary_key=True, index=True)
    
    # Tên địa điểm (VD: "Tòa nhà C", "Căn tin")
    name = Column(String, index=True, nullable=False)
    
    # Mô tả ngắn gọn
    description = Column(String)
    
    # Tọa độ GPS để ứng dụng Mobile biết vị trí hiển thị AR
    latitude = Column(Float, nullable=False)  # Vĩ độ
    longitude = Column(Float, nullable=False) # Kinh độ
    
    # Điểm này có đang được bật các tính năng AR hay không
    is_ar_active = Column(Boolean, default=True)