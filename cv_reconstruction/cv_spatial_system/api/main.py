from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import cv2
import io
import base64
from PIL import Image
from ultralytics import YOLO
from pillow_heif import register_heif_opener

register_heif_opener() # Hỗ trợ định dạng HEIC cho PIL

# Import from our modules (using try-except for robust importing during local runs)
try:
    from cv_spatial_system.spatial_computing.dust3r_vps import VisualPositioningSystem
    from cv_spatial_system.spatial_computing.geometry import PoseEstimator6DoF
except ModuleNotFoundError:
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from spatial_computing.dust3r_vps import VisualPositioningSystem
    from spatial_computing.geometry import PoseEstimator6DoF

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CV Spatial Computing API", version="1.0.0")

# Enable CORS for Web App testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize mock components
vps = VisualPositioningSystem(point_cloud_map=np.zeros((100, 3)))
pose_estimator = PoseEstimator6DoF()
# Khởi tạo mô hình nhận diện vật thể YOLOv8
print("Đang tải YOLOv8 model...")
yolo_model = YOLO("yolov8n.pt")

@app.post("/api_identify_location")
async def api_identify_location(image: UploadFile = File(...)):
    """
    Exposes a backend API that processes incoming images from devices 
    and returns the corresponding Location ID.
    """
    contents = await image.read()
    
    # Thử đọc bằng PIL (đã cấu hình hỗ trợ HEIC)
    try:
        pil_img = Image.open(io.BytesIO(contents))
        pil_img = pil_img.convert("RGB") # Chuyển về RGB chuẩn
        # OpenCV dùng không gian màu BGR
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        # Fallback về phương pháp chuẩn của OpenCV nếu PIL lỗi
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"error": "Invalid image format."}
        
    # Chạy YOLOv8 để nhận diện vật thể
    results = yolo_model(frame)
    annotated_frame = results[0].plot() # Tự động vẽ Bounding Box lên ảnh
    
    # Mã hoá ảnh sang Base64 để gửi về frontend
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Process through VPS
    t_vec, r_mat = vps.visual_localization_vps(frame)
    
    # In a real scenario, map the t_vec to a semantic Location ID
    location_id = f"LOC_{np.sum(t_vec):.2f}"
    
    return {
        "status": "success",
        "location_id": location_id,
        "translation": t_vec.tolist(),
        "rotation": r_mat.tolist(),
        "image_base64": image_base64
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
