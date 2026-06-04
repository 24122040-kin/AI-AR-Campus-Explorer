# engine/train_models.py
"""
Script huấn luyện các mô hình PyTorch cho Campus Explorer.
Huấn luyện 2 mô hình:
1. IntentClassifier (Phân loại ý định tìm kiếm/hỏi đường của sinh viên)
2. CrowdPredictor (Dự báo mức độ đông đúc của các địa điểm)
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from engine.nlp_processor import normalize_text, IntentClassifier
from engine.recommender import CrowdPredictor

# Thiết lập seed để kết quả ổn định
torch.manual_seed(42)
np.random.seed(42)

# Đường dẫn
ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(ENGINE_DIR, "model_metadata.json")
INTENT_MODEL_PATH = os.path.join(ENGINE_DIR, "intent_model.pth")
CROWD_MODEL_PATH = os.path.join(ENGINE_DIR, "crowd_model.pth")

# ---------------------------------------------------------------------------
# 1. DỮ LIỆU HUẤN LUYỆN Ý ĐỊNH (INTENT CLASSIFIER DATA)
# ---------------------------------------------------------------------------
INTENT_TRAINING_DATA = [
    # 0: route_search
    ("chỉ đường đi", 0),
    ("đường tới tòa b", 0),
    ("lộ trình đến thư viện", 0),
    ("đi tới căn tin thế nào", 0),
    ("chỉ tôi đường đi nhanh nhất tới nhà xe", 0),
    ("hướng dẫn đường đi tránh mưa tới tòa d", 0),
    ("đường đi xe lăn tới tòa c", 0),
    ("muốn đi đến tòa e", 0),
    ("làm sao đi từ tòa a sang tòa g", 0),
    ("chỉ đường đến cổng trường", 0),
    ("đường tắt đi học", 0),
    ("lộ trình ngắn nhất", 0),
    ("tìm đường đi tránh nắng", 0),
    ("đi thang máy tòa b", 0),
    ("đường đi không có thang bộ", 0),
    ("hướng dẫn di chuyển đến atm", 0),
    ("chỉ đường sang tòa f", 0),

    # 1: search_empty_lab
    ("tìm phòng học trống", 1),
    ("phòng máy tính nào rảnh", 1),
    ("có phòng tự học nào trống không", 1),
    ("tìm phòng lab còn chỗ", 1),
    ("phòng học nào không có lớp", 1),
    ("tự học ở đâu bây giờ", 1),
    ("lab cntt còn chỗ trống không", 1),
    ("phòng tự học tòa c có mở cửa không", 1),
    ("kiếm phòng máy tính rảnh", 1),
    ("phòng nào yên tĩnh để học bài", 1),
    ("tìm chỗ tự học", 1),
    ("phòng lab b301 rảnh không", 1),

    # 2: search_food_low_crowd
    ("căn tin có đông không", 2),
    ("đói bụng muốn ăn cơm", 2),
    ("chỗ nào ăn uống vắng người", 2),
    ("canteen tòa d có đông không", 2),
    ("kiếm quán nước uống tránh nóng", 2),
    ("muốn ăn trưa chỗ nào mát mẻ", 2),
    ("căn tin tòa b có đồ ăn không", 2),
    ("muốn uống cà phê vắng người", 2),
    ("tìm chỗ ăn trưa không quá đông", 2),
    ("canteen có bánh mì không", 2),
    ("chỗ nào ăn vặt trong trường", 2),
    ("ăn trưa ở đâu ngon rẻ", 2),

    # 3: event_recommend
    ("hôm nay có sự kiện gì không", 3),
    ("clb nào đang sinh hoạt", 3),
    ("có hội thảo hay workshop nào", 3),
    ("sắp tới có seminar gì thế", 3),
    ("sự kiện hot clb cntt", 3),
    ("lịch sinh hoạt clb robot", 3),
    ("hội thảo nghiên cứu khoa học tổ chức ở đâu", 3),
    ("có hoạt động ngoại khóa nào hôm nay", 3),
    ("chương trình ca nhạc tối nay", 3),
    ("workshop học máy tòa d", 3),
    ("sự kiện tiếp theo trong tuần", 3),

    # 4: general_chat
    ("hello", 4),
    ("chào bạn", 4),
    ("bạn là ai", 4),
    ("app này dùng làm gì", 4),
    ("chúc một ngày tốt lành", 4),
    ("cảm ơn nhé", 4),
    ("tạm biệt", 4),
    ("hi assistant", 4),
    ("giúp tôi với", 4),
    ("bạn biết làm gì", 4),
    ("ok cảm ơn", 4),
    ("bye bye", 4)
]


class IntentDataset(Dataset):
    def __init__(self, data, vocab):
        self.x_data = []
        self.y_data = []
        
        for text, label in data:
            norm_text = normalize_text(text)
            bow = self.text_to_bow(norm_text, vocab)
            self.x_data.append(bow)
            self.y_data.append(label)
            
        self.x_data = torch.tensor(np.array(self.x_data), dtype=torch.float32)
        self.y_data = torch.tensor(self.y_data, dtype=torch.long)
        self.n_samples = len(data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

    @staticmethod
    def text_to_bow(text: str, vocab: list) -> np.ndarray:
        words = text.split()
        vector = np.zeros(len(vocab), dtype=np.float32)
        for w in words:
            if w in vocab:
                vector[vocab.index(w)] += 1.0
        return vector


# ---------------------------------------------------------------------------
# 2. DỰ BÁO ĐỘ ĐÔNG ĐÚC (CROWD PREDICTOR DATA GENERATION)
# ---------------------------------------------------------------------------
def generate_crowd_data(nodes, weather_types, input_dim, num_samples=1000):
    """
    Sinh dữ liệu mô phỏng mức độ đông đúc theo các quy luật thực tế:
    - Căn tin đông vào giờ trưa (11h30 - 13h) và sáng (7h - 9h).
    - Thư viện / Tòa học đông vào giờ học (8h - 11h, 14h - 16h30).
    - Nhà thể dục đông vào chiều tối (16h30 - 18h30).
    - Mùa thi cử (tháng 1, 5, 6, 12) đông đúc hơn tại thư viện và phòng học.
    - Thời tiết mưa làm các khu vực trong nhà đông hơn dã ngoại.
    """
    x_data = []
    y_data = []
    
    for _ in range(num_samples):
        # Chọn ngẫu nhiên node
        node = np.random.choice(nodes)
        node_idx = nodes.index(node)
        node_vec = np.zeros(len(nodes), dtype=np.float32)
        node_vec[node_idx] = 1.0
        
        # Chọn ngẫu nhiên các đặc trưng thời gian
        hour = np.random.uniform(6.0, 22.0)
        dow = np.random.randint(0, 7) # 0: thứ 2, 6: chủ nhật
        month = np.random.randint(1, 13)
        is_exam = 1.0 if month in (1, 5, 6, 12) else 0.0
        
        # Chọn ngẫu nhiên thời tiết
        weather = np.random.choice(weather_types)
        weather_idx = weather_types.index(weather)
        weather_vec = np.zeros(len(weather_types), dtype=np.float32)
        weather_vec[weather_idx] = 1.0
        
        # Tạo vector đặc trưng thô
        raw_features = np.concatenate([
            node_vec,
            [hour / 24.0, dow / 6.0, (month - 1) / 11.0, is_exam],
            weather_vec
        ])
        
        # Slicing/padding khớp với input_dim của mô hình
        if len(raw_features) != input_dim:
            if len(raw_features) > input_dim:
                features = raw_features[:input_dim]
            else:
                features = np.pad(raw_features, (0, input_dim - len(raw_features)))
        else:
            features = raw_features
            
        # Xác định nhãn đông đúc mô phỏng
        crowd_level = 0.2 # Mức nền
        
        # Quy luật căn tin
        if node == "Căn tin":
            if 11.5 <= hour <= 13.0:
                crowd_level = np.random.uniform(0.85, 0.98)
            elif 7.0 <= hour <= 9.0:
                crowd_level = np.random.uniform(0.5, 0.7)
            else:
                crowd_level = np.random.uniform(0.3, 0.5)
                
        # Quy luật thư viện & phòng học
        elif node in ("Thư viện Tòa D", "Tòa B", "Tòa C", "Tòa D"):
            if (8.0 <= hour <= 11.0) or (14.0 <= hour <= 16.5):
                base_c = 0.75 if is_exam else 0.55
                crowd_level = np.random.uniform(base_c, base_c + 0.2)
            else:
                crowd_level = np.random.uniform(0.2, 0.4)
                
        # Quy luật nhà thể thao/gym
        elif node == "Nhà thể dục":
            if 16.5 <= hour <= 19.0:
                crowd_level = np.random.uniform(0.75, 0.9)
            elif dow >= 5 and 8.0 <= hour <= 11.0: # Cuối tuần sáng
                crowd_level = np.random.uniform(0.6, 0.8)
            else:
                crowd_level = np.random.uniform(0.15, 0.35)
                
        # Quy luật thời tiết mưa đối với các khu vực trong nhà
        if weather == "rainy" and node in ("Căn tin", "Thư viện Tòa D", "Tòa B", "Tòa C"):
            crowd_level = min(1.0, crowd_level + np.random.uniform(0.05, 0.15))
            
        # Nhiễu nhẹ
        crowd_level = float(np.clip(crowd_level + np.random.normal(0, 0.02), 0.0, 1.0))
        
        x_data.append(features)
        y_data.append(crowd_level)
        
    return torch.tensor(np.array(x_data), dtype=torch.float32), torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)


class CrowdDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
        
    def __len__(self):
        return len(self.x)


# ---------------------------------------------------------------------------
# 3. TIẾN TRÌNH HUẤN LUYỆN CHÍNH
# ---------------------------------------------------------------------------
def train_and_save():
    # ── ĐỌC METADATA CŨ ĐỂ ĐỒNG BỘ ──────────────────────────────────────────
    print("📖 Đang nạp metadata hiện tại từ model_metadata.json...")
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Đảm bảo các cấu trúc khóa tồn tại
    if "intent" not in metadata:
        metadata["intent"] = {}
    if "crowd" not in metadata:
        metadata["crowd"] = {}

    # ── XỬ LÝ VOCABULARY VÀ LABELS CHO INTENT ───────────────────────────────
    # Trích xuất từ vựng từ tập dữ liệu huấn luyện mới và tích hợp từ vựng cũ
    existing_vocab = metadata["intent"].get("vocab", [])
    new_vocab_set = set(existing_vocab)
    
    for text, _ in INTENT_TRAINING_DATA:
        norm_text = normalize_text(text)
        for word in norm_text.split():
            if word and len(word) >= 1:
                new_vocab_set.add(word)
                
    vocab = sorted(list(new_vocab_set))
    vocab_size = len(vocab)
    
    labels_dict = {
        "0": "route_search",
        "1": "search_empty_lab",
        "2": "search_food_low_crowd",
        "3": "event_recommend",
        "4": "general_chat"
    }
    
    print(f"📌 Intent Vocabulary size: {vocab_size} từ (Đã đồng bộ và mở rộng)")
    
    # ── XỬ LÝ CONFIG CHO CROWD PREDICTOR ────────────────────────────────────
    nodes = metadata["crowd"].get("nodes", [
        "Tòa A", "Tòa B", "Tòa C", "Tòa D", "Tòa E", "Tòa F", "Tòa G",
        "Nhà thể dục", "Nhà xe", "ATM", "Nhà điều hành", "Cổng trường", "Căn tin"
    ])
    weather_types = metadata["crowd"].get("weather_types", ["normal", "sunny", "rainy"])
    
    # input_dim gốc là 19, ta giữ nguyên 19 để tương thích 100% với recommender.py
    input_dim = metadata["crowd"].get("input_dim", 19)
    print(f"📌 Crowd Predictor input dimension: {input_dim}")

    # ── HUẤN LUYỆN INTENT CLASSIFIER ────────────────────────────────────────
    print("\n🧠 Bắt đầu huấn luyện Intent Classifier...")
    intent_dataset = IntentDataset(INTENT_TRAINING_DATA, vocab)
    intent_loader = DataLoader(intent_dataset, batch_size=8, shuffle=True)
    
    intent_model = IntentClassifier(vocab_size, len(labels_dict))
    intent_model.train()
    
    criterion_intent = nn.CrossEntropyLoss()
    optimizer_intent = optim.Adam(intent_model.parameters(), lr=0.005, weight_decay=1e-4)
    
    epochs_intent = 80
    for epoch in range(epochs_intent):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for x_batch, y_batch in intent_loader:
            optimizer_intent.zero_grad()
            outputs = intent_model(x_batch)
            loss = criterion_intent(outputs, y_batch)
            loss.backward()
            optimizer_intent.step()
            
            epoch_loss += loss.item() * x_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
        if (epoch + 1) % 20 == 0:
            acc = (correct / total) * 100
            avg_loss = epoch_loss / len(intent_loader.dataset)
            print(f"Epoch [{epoch+1}/{epochs_intent}], Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")

    # ── HUẤN LUYỆN CROWD PREDICTOR ──────────────────────────────────────────
    print("\n🧠 Bắt đầu huấn luyện Crowd Predictor...")
    x_crowd, y_crowd = generate_crowd_data(nodes, weather_types, input_dim, num_samples=1500)
    crowd_dataset = CrowdDataset(x_crowd, y_crowd)
    crowd_loader = DataLoader(crowd_dataset, batch_size=32, shuffle=True)
    
    crowd_model = CrowdPredictor(input_dim)
    crowd_model.train()
    
    criterion_crowd = nn.MSELoss()
    optimizer_crowd = optim.Adam(crowd_model.parameters(), lr=0.002)
    
    epochs_crowd = 120
    for epoch in range(epochs_crowd):
        epoch_loss = 0.0
        for x_batch, y_batch in crowd_loader:
            optimizer_crowd.zero_grad()
            outputs = crowd_model(x_batch)
            loss = criterion_crowd(outputs, y_batch)
            loss.backward()
            optimizer_crowd.step()
            epoch_loss += loss.item() * x_batch.size(0)
            
        if (epoch + 1) % 30 == 0:
            avg_loss = epoch_loss / len(crowd_loader.dataset)
            print(f"Epoch [{epoch+1}/{epochs_crowd}], Loss (MSE): {avg_loss:.6f}")

    # ── LƯU TRỮ MÔ HÌNH VÀ CẬP NHẬT METADATA ────────────────────────────────
    print("\n💾 Đang lưu trữ mô hình và cập nhật metadata...")
    
    # Lưu trọng số mô hình PyTorch
    torch.save(intent_model.state_dict(), INTENT_MODEL_PATH)
    torch.save(crowd_model.state_dict(), CROWD_MODEL_PATH)
    print(f"💾 Đã lưu Intent weights -> {INTENT_MODEL_PATH}")
    print(f"💾 Đã lưu Crowd weights -> {CROWD_MODEL_PATH}")

    # Cập nhật metadata
    metadata["intent"]["vocab"] = vocab
    metadata["intent"]["vocab_size"] = vocab_size
    metadata["intent"]["labels"] = labels_dict
    
    metadata["crowd"]["nodes"] = nodes
    metadata["crowd"]["weather_types"] = weather_types
    metadata["crowd"]["input_dim"] = input_dim
    
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"💾 Đã cập nhật metadata -> {METADATA_PATH}")
    
    print("\n✅ Quá trình huấn luyện hoàn tất thành công!")

if __name__ == "__main__":
    train_and_save()
