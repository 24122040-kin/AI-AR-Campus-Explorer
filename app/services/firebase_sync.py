import os
import firebase_admin
from firebase_admin import credentials, firestore

def init_firebase():
    """Khởi tạo kết nối tới Firebase khi Server FastAPI vừa chạy"""
    # Kiểm tra xem đã kết nối chưa để tránh lỗi khởi tạo nhiều lần
    if not firebase_admin._apps:
        # Đường dẫn tới file chìa khóa JSON của Firebase (Sẽ tạo ở bước sau)
        cred_path = "firebase-service-account.json"
        
        if os.path.exists(cred_path):
            try:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                print("✅ Đã kết nối thành công tới Firebase Cloud!")
            except Exception as e:
                print(f"❌ Lỗi kết nối Firebase: {e}")
        else:
            print(f"⚠️ Cảnh báo: Không tìm thấy file chìa khóa '{cred_path}'. Tạm bỏ qua Firebase.")

def sync_data_to_firestore(collection_name: str, document_id: str, data: dict):
    """
    Hàm dùng để đẩy dữ liệu từ Backend (SQLite) lên Firebase Firestore.
    Ví dụ: sync_data_to_firestore("users", "user_123", {"name": "Khánh", "role": "admin"})
    """
    if not firebase_admin._apps:
        return {"status": "error", "message": "Firebase chưa được kết nối"}
    
    db = firestore.client()
    try:
        # Lưu hoặc cập nhật dữ liệu (merge=True giúp không xóa đè dữ liệu cũ)
        db.collection(collection_name).document(document_id).set(data, merge=True)
        return {"status": "success", "message": f"Đã đồng bộ {document_id} lên mây!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}