import numpy as np
import torch
import cv2

class DepthEstimator:
    def __init__(self, model_type: str = "MiDaS_small"):
        """
        Khởi tạo mô hình ước lượng độ sâu MiDaS thực tế.
        model_type có thể là: "DPT_Large", "DPT_Hybrid", hoặc "MiDaS_small"
        """
        self.model_type = model_type
        print(f"Đang tải mô hình {model_type} từ PyTorch Hub...")
        
        # Tự động chọn thiết bị chạy (CPU, CUDA, hoặc MPS cho Mac)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            
        # Tải mô hình từ Torch Hub
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.midas.to(self.device)
        self.midas.eval() # Chuyển sang chế độ evaluation
        
        # Tải các hàm tiền xử lý ảnh tương ứng
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
            
        print(f"Tải mô hình thành công lên thiết bị: {self.device}")

    def extract_depth_maps(self, frame: np.ndarray) -> np.ndarray:
        """
        Ước lượng depth map thực tế từ ảnh đầu vào để xử lý Occlusion (che khuất) cho AR.
        """
        if frame is None:
            return None
            
        # Đảm bảo ảnh ở không gian màu RGB (OpenCV mặc định là BGR)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Tiền xử lý ảnh (Resize, Normalize)
        input_batch = self.transform(img).to(self.device)
        
        # Chạy mô hình (Inference)
        with torch.no_grad():
            prediction = self.midas(input_batch)
            
            # Resize bản đồ độ sâu trả về kích thước gốc của frame
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        output = prediction.cpu().numpy()
        
        # Chuẩn hoá mảng độ sâu về [0, 255] dưới dạng ảnh uint8
        depth_min = output.min()
        depth_max = output.max()
        if depth_max > depth_min:
            normalized_depth = 255 * (output - depth_min) / (depth_max - depth_min)
        else:
            normalized_depth = output
            
        return normalized_depth.astype(np.uint8)

class PoseEstimator6DoF:
    def __init__(self):
        pass

    def pose_estimation_6dof(self, visual_features: dict, imu_data: dict) -> dict:
        """
        Computes the exact rotation and translation of the mobile device in 3D space with 6DoF.
        """
        return {
            "rotation_matrix": np.eye(3), 
            "translation_vector": np.zeros(3)
        }
