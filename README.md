# AI AR Campus Explorer - HCMUS

## 📁 Cấu Trúc Mã Nguồn

Dự án được chia thành 3 phân hệ độc lập: **Backend** (Xử lý logic & AI), **Frontend Admin** (Quản lý hệ thống), và **Web App** (Dành cho sinh viên/người dùng cuối).

```text
AI-AR-Campus-Explorer/
│
├── app/                           # THƯ MỤC BACKEND (Python / FastAPI)
│   ├── ai/                        # Nơi chứa logic Trí tuệ nhân tạo (Tích hợp sau)
│   │   ├── rag.py                 # (Dự kiến) Xử lý Chatbot RAG
│   │   └── cv_gnn.py              # (Dự kiến) Xử lý CV và Đồ thị tìm đường
│   ├── api/                       # Định nghĩa các Endpoints (REST & WebSocket)
│   ├── core/                      # Lõi cấu hình và bảo mật
│   │   ├── config.py              # Load biến môi trường từ file .env
│   │   └── security.py            # Xử lý mã hóa mật khẩu & JWT Token
│   ├── db/                        # Quản lý Database
│   │   └── session.py             # Kết nối SQLite & SQLAlchemy
│   ├── models/                    # Các Model Database (User, Location...)
│   ├── schemas/                   # Pydantic Schemas kiểm duyệt dữ liệu In/Out
│   ├── services/                  # Tích hợp dịch vụ bên thứ 3
│   │   └── firebase_sync.py       # Khởi tạo & Đồng bộ dữ liệu lên Firebase Cloud
│   └── main.py                    # File gốc khởi chạy Server, cấu hình CORS & Websocket
│
├── frontend/                      # THƯ MỤC WEB ADMIN (Vite / React / Vue)
│   └── ...                        # Chứa giao diện quản lý User, Location dành cho Admin
│
├── web_app/                       # THƯ MỤC WEB APP NGƯỜI DÙNG (Giao diện Chatbot)
│   ├── index.html                 # Giao diện chính (Tailwind CSS) - Login & Chat UI
│   └── script.js                  # Logic kết nối WebSocket, xử lý Chatbot realtime
│
├── .env                           # [QUAN TRỌNG] Chứa GEMINI_API_KEY (Không push lên Git)
├── firebase-service-account.json  # [QUAN TRỌNG] Khóa riêng tư Firebase (Không push lên Git)
├── campus_explorer.db             # [TỰ ĐỘNG TẠO] CSDL SQLite lưu trữ thông tin nội bộ
├── requirements.txt               # Danh sách thư viện Python (pip install -r requirements.txt)
└── README.md                      # Tài liệu mô tả dự án
```

## ⚙️ Hướng Dẫn Cài Đặt & Khởi Chạy

1. Thiết lập Backend (FastAPI)
- Cài các thư viện cần thiết (terminal):
    pip install -r requirements.txt
- Thêm API key (file .env):
    GEMINI_API_KEY=your_gemini_api_key_here
    DATABASE_URL=sqlite:///./campus_explorer.db
- Chạy backend (terminal):
    uvicorn app.main:app --reload (muốn dừng chạy thì bấm Ctrl C)

2. Chạy Web Admin
- Di chuyển đến thư mục frontend (terminal):
    cd frontend
- Chạy frontend (terminal):
    npm install (nếu chưa cài npm)
    npm run dev (muốn dừng chạy thì bấm Ctrl C)

3. Chạy Web App
- Mở thư mục web_app
- Mở file index.html
