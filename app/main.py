import jwt
import asyncio
from typing import List
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.db.session import engine, Base, get_db
from app.models import User, Location
from app.schemas import UserCreate, UserResponse, UserLogin, Token, LocationCreate, LocationResponse
from app.core.security import get_password_hash, verify_password, create_access_token
from app.core.security import SECRET_KEY, ALGORITHM

# 1. NHẬP KHẨU TÍNH NĂNG ĐỒNG BỘ HÓA FIREBASE CỦA BẠN VÀO ĐÂY
from app.services.firebase_sync import init_firebase

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="AI AR Campus Explorer API",
    description="API Gateway cho dự án AR Campus Explorer của trường HCMUS",
    version="1.0.0"
)

# 2. KHỞI ĐỘNG FIREBASE NGAY KHI APP VỪA CHẠY
init_firebase()

# 3. CẤU HÌNH CORS (ĐÃ LỌC BỎ ĐOẠN TRÙNG LẶP Ở GIỮA FILE)
# Giúp Web Frontend (React/Vue/HTML) có thể gọi API tới server thoải mái không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Cho phép mọi trang web gọi vào (khi deploy thật sẽ chỉnh lại)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Công cụ yêu cầu người dùng phải cung cấp Token (Bearer Token)
security = HTTPBearer()

# Hàm kiểm tra Token và lấy thông tin người dùng
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials
    try:
        # Giải mã thẻ ra vào
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Thẻ ra vào không hợp lệ!")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Thẻ đã hết hạn, vui lòng đăng nhập lại!")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Thẻ ra vào không hợp lệ hoặc bị làm giả!")
    
    # Tìm sinh viên trong database bằng email giải mã được
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy sinh viên này!")
    
    return user

# Endpoint kiểm tra trạng thái hoạt động của Server
@app.get("/")
async def root():
    return {"message": "Chào mừng đến với API của AI AR Campus Explorer!"}

# --- CÁC HÀM GIẢ LẬP AI ---

async def ai_process_voice_chat(user_message: str):
    """Giả lập hàm RAG của Bảo Kin: Xử lý câu hỏi và trả về câu trả lời"""
    await asyncio.sleep(1) # Giả lập thời gian AI suy nghĩ mất 1 giây
    return {"status": "success", "reply": "Thư viện nằm ở Tòa nhà C."}

async def ai_process_ar_navigation(lat: float, lon: float):
    """Giả lập hàm CV của Trung Hiếu và GNN của Chấn Khoa: Phân tích tọa độ"""
    await asyncio.sleep(0.5) # Giả lập thời gian xử lý
    return {
        "status": "success", 
        "ar_data": {
            "arrow_direction": "right", 
            "distance_to_turn": "15m",
            "crowd_warning": "Không có kẹt xe"
        }
    }

# --- TỔNG ĐÀI ĐIỀU PHỐI WEBSOCKET CHÍNH (ĐÃ CHUYỂN LOG SANG CHO WEB) ---

@app.websocket("/ws/ar-stream")
async def websocket_endpoint(websocket: WebSocket):
    # Chấp nhận cuộc gọi từ Web Frontend
    await websocket.accept()
    print("🌐 Web Frontend đã kết nối WebSocket thành công!")
    
    try:
        while True:
            # Nhận dữ liệu từ Web dưới dạng chuẩn JSON
            data = await websocket.receive_json()
            action = data.get("action")
            
            # ĐIỀU PHỐI CÔNG VIỆC
            if action == "chat":
                message = data.get("message", "")
                print(f"Nhận yêu cầu CHAT từ Web: {message}")
                
                # Gọi bộ phận AI Chat xử lý
                ai_response = await ai_process_voice_chat(message)
                
                # Trả kết quả ngược lại cho Web
                await websocket.send_json({"type": "chat_response", "data": ai_response})
                
            elif action == "navigate":
                lat = data.get("lat", 0.0)
                lon = data.get("lon", 0.0)
                print(f"Nhận yêu cầu NAVIGATE từ Web tại tọa độ: {lat}, {lon}")
                
                # Gọi bộ phận Computer Vision & GNN xử lý
                ai_response = await ai_process_ar_navigation(lat, lon)
                
                # Trả dữ liệu vẽ ngược lại cho Web
                await websocket.send_json({"type": "ar_response", "data": ai_response})
                
            else:
                await websocket.send_json({"error": "Hành động (action) không hợp lệ!"})
                
    except WebSocketDisconnect:
        print("🌐 Web Frontend đã ngắt kết nối WebSocket.")

# --- API QUẢN LÝ NGƯỜI DÙNG (USERS) ---

@app.post("/users/", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email này đã được đăng ký!")
    
    new_user = User(
        full_name=user.full_name,
        email=user.email,
        hashed_password=get_password_hash(user.password)
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/login", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Email hoặc mật khẩu không chính xác!")
        
    access_token = create_access_token(data={"sub": db_user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserResponse)
def read_user_profile(current_user: User = Depends(get_current_user)):
    return current_user

@app.get("/users/", response_model=List[UserResponse])
def read_all_users(db: Session = Depends(get_db), current_admin: User = Depends(get_current_user)):
    users = db.query(User).all()
    return users

# --- API QUẢN LÝ ĐỊA ĐIỂM (LOCATIONS) ---

@app.post("/locations/", response_model=LocationResponse)
def create_location(location: LocationCreate, db: Session = Depends(get_db), current_admin: User = Depends(get_current_user)):
    new_location = Location(
        name=location.name,
        description=location.description,
        latitude=location.latitude,
        longitude=location.longitude,
        is_ar_active=location.is_ar_active
    )
    db.add(new_location)
    db.commit()
    db.refresh(new_location)
    return new_location

@app.get("/locations/", response_model=List[LocationResponse])
def read_all_locations(db: Session = Depends(get_db)):
    locations = db.query(Location).all()
    return locations

@app.delete("/locations/{location_id}")
def delete_location(location_id: int, db: Session = Depends(get_db), current_admin: User = Depends(get_current_user)):
    location = db.query(Location).filter(Location.id == location_id).first()
    
    if not location:
        raise HTTPException(status_code=404, detail="Không tìm thấy địa điểm này!")
    
    db.delete(location)
    db.commit()
    return {"message": "Đã xóa địa điểm thành công!"}

# Lệnh tự động tạo bảng dữ liệu
Base.metadata.create_all(bind=engine)