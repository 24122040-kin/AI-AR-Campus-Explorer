# AI AR Campus Explorer

source/
├── backend/
│   ├── ai/
│   │   ├── information_chatbot/   # Chừa chỗ cho code RAG
│   │   ├── local_map/             # Chừa chỗ cho code CV, DUSt3R
│   │   ├── recommendation_system/ # Chừa chỗ cho code GNN
│   │   └── face_guard/            # Nơi chứa code nhận diện khuôn mặt
│   ├── core/
│   │   ├── config.py              # Cấu hình Firebase
│   │   ├── security.py            # Xử lý JWT Token và mã hóa mật khẩu
│   │   └── serviceAccountKey.json # Chứa khoá bí mật để giao tiếp với Firebase DB
│   ├── models/
│   │   └── schemas.py             # Khai báo cấu trúc dữ liệu đầu vào/ra
│   ├── routers/
│   │   ├── users.py               # Quản lý tài khoản và xác thực
│   │   ├── locations.py           # Truy xuất dữ liệu tòa nhà, phòng học
│   │   └── ws.py                  # Kênh kết nối WebSocket thời gian thực
│   ├── main.py                    # File gốc chạy ứng dụng FastAPI
│   └── requirements.txt           # Danh sách các thư viện cần tải
└── frontend/                      # Giao diện cho người dùng (HTML/CSS, JS)
