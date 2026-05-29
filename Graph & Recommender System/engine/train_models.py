# engine/train_models.py
"""
Script huấn luyện các mô hình AI cho hệ thống Campus Navigator.

Các mô hình được train:
  1. IntentClassifier   — Phân loại ý định người dùng (NLP, BoW + MLP)
  2. CrowdPredictor     — Dự báo mật độ đám đông theo giờ/ngày/thời tiết tại cấp Tòa nhà

NCF/NFC đã bị loại bỏ hoàn toàn.
"""

import os
import json
import re
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------------------------
# Tiện ích chuẩn hóa văn bản tiếng Việt
# ---------------------------------------------------------------------------
_ACCENTED = (
    "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ"
    "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂễỆĐÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ"
)
_PLAIN = (
    "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyy"
    "AAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIIIOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYY"
)
_TRANS_TABLE = str.maketrans(_ACCENTED, _PLAIN)

def remove_accents(text: str) -> str:
    return text.translate(_TRANS_TABLE)

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = remove_accents(text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

# ===========================================================================
# PHẦN 1: INTENT CLASSIFIER
# ===========================================================================
INTENT_LABELS = {
    0: "route_search",
    1: "search_empty_lab",
    2: "search_food_low_crowd",
    3: "event_recommend",
    4: "general_chat",
}
INTENT_MAP = {v: k for k, v in INTENT_LABELS.items()}


def generate_intent_data():
    """
    Sinh ~800+ mẫu huấn luyện cho Intent Classifier.
    """
    raw = []

    # ---- 0: route_search ----
    verbs = [
        "chi duong", "tim duong", "dan duong", "lo trinh", "duong di",
        "huong dan di", "lam sao de di", "chi to di", "di den", "muon di",
        "can di", "loi nao den", "duong nao toi", "chi minh duong",
        "ban co the chi", "giup minh tim duong", "toi muon biet duong",
    ]
    dests = [
        "thu vien", "can tin", "nha xe", "phong may", "lab", "toa d",
        "toa b", "toa a", "toa c", "toa e", "toa f", "nha the duc",
        "phong tu hoc", "phong hoc", "phong nghi", "atm", "nha dieu hanh",
        "cong truong", "van phong khoa", "phong thi nghiem",
    ]
    mods = [
        "nhanh nhat", "co mai che", "cho xe lan", "co thang may",
        "tranh mua", "khong leo thang bo", "tranh nang", "gan nhat",
        "di xe lan", "an toan nhat", "it nguoi nhat", "co bong mat",
        "", "", "",  # nhiều mẫu không có modifier
    ]
    for _ in range(120):
        v = random.choice(verbs)
        d = random.choice(dests)
        m = random.choice(mods)
        raw.append((f"{v} den {d} {m}".strip(), "route_search"))
        raw.append((f"duong toi {d} {m}".strip(), "route_search"))
        if m:
            raw.append((f"{d} {m}", "route_search"))

    specific_routes = [
        "di tu toa a sang toa c bang duong nao",
        "huong dan di thang may len tang 3",
        "duong di khong leo cau thang toi phong lab",
        "lo trinh co thang may den van phong khoa",
        "troi mua to roi di den thu vien kieu gi",
        "tim duong di co mai che den nha xe",
        "troi nang lam co duong co mai che khong",
        "xe lan di loi nao vao toa d",
        "huong dan duong cho xe lan",
        "tim duong nhanh nhat ve nha xe",
        "muon di thang may len tang 2 toa b",
        "loi di nao tranh nang den can tin",
        "duong nao it nguoi nhat den thu vien",
        "chi minh duong di toa g xem su kien",
        "lam sao de di tu cong truong vao toa d",
        "co duong nao di khong bi mua khong",
        "muon di tu nha xe vao phong hoc 101",
        "duong di ngan nhat tu toa b sang toa d",
        "huong dan di tu can tin len thu vien",
        "tim lo trinh tu atm den phong may b301",
    ]
    for q in specific_routes:
        raw.append((q, "route_search"))

    # ---- 1: search_empty_lab ----
    lab_templates = [
        "phong may nao trong luc nay",
        "lab nao con may trong khong",
        "tu hoc o dau bay gio",
        "tim phong may tinh thuc hanh dang ranh",
        "co phong thuc hanh tin hoc nao dang trong khong",
        "kiem tra phong may c tang 2",
        "phong may thuc hanh nao khong co lop hoc",
        "kiem tra phong lab trong",
        "tim phong may tinh ranh",
        "co cho nao de tu hoc yen tinh khong",
        "cho ngoi tu hoc ranh thoi diem nay",
        "muon tim phong lab cntt con trong may",
        "cho nao co ban ghe de hoc bai",
        "tim phong tu hoc",
        "phong nao co may tinh trong de lam bai tap",
        "can phong may de code bai tap",
        "phong thuc hanh nao dang ranh may tinh",
        "lab may tinh nao khong co lop",
        "tim cho ngoi hoc co may tinh",
        "phong may nao co the vao hoc tu do",
        "can tim phong co may tinh de lam do an",
        "phong lab nao dang trong co the vao khong",
        "muon tim cho co may tinh de on thi",
        "phong may nao it nguoi nhat hien tai",
        "lab nao co the vao tu hoc khong can dat truoc",
    ]
    for q in lab_templates:
        raw.append((q, "search_empty_lab"))
        raw.append((f"cho minh {q}", "search_empty_lab"))
        raw.append((f"xem {q}", "search_empty_lab"))
        raw.append((f"giup toi {q}", "search_empty_lab"))

    # ---- 2: search_food_low_crowd ----
    food_templates = [
        "tim can tin vang",
        "cho nao an trua khong dong",
        "muon an uong cho vang nguoi",
        "can tin co dong khong",
        "quan an nao it nguoi nhat bay gio",
        "tim quan ca phe vang nguoi",
        "cho an uong nao thua nguoi luc nay",
        "an trua o dau vang nhat",
        "muon uong nuoc cho nao vang nguoi",
        "canteen bay gio co dong khong",
        "cho nao an com vang ve",
        "tim cho uong ca phe it nguoi",
        "doi bung muon an o cho vang",
        "can tin nao it hang cho ngoi",
        "muon an sang o dau khong dong",
        "cho nao co do an ma khong phai xep hang",
        "tim quan an nhanh it nguoi",
        "can tin nao mo som nhat sang nay",
        "muon an trua som tranh dong",
        "cho nao co nuoc uong ma vang nguoi",
        "tim quan an co cho ngoi thoai mai",
        "can tin nao co mon an ngon ma khong dong",
        "muon tim cho an uong yen tinh",
        "cho nao ban do an ma khong qua dong",
        "tim quan an nhanh gan day",
    ]
    for q in food_templates:
        raw.append((q, "search_food_low_crowd"))
        raw.append((f"cho minh {q}", "search_food_low_crowd"))
        raw.append((f"chi can {q}", "search_food_low_crowd"))
        raw.append((f"ban oi {q}", "search_food_low_crowd"))

    # ---- 3: event_recommend ----
    event_templates = [
        "hom nay co hoi thao gi khong",
        "su kien clb nao dang dien ra",
        "co seminar gi moi khong",
        "goi y hoi thao khoa hoc",
        "co hoat dong clb nao hom nay",
        "lich su kien o san toa g",
        "seminar o thu vien may gio",
        "su kien nao phu hop voi so thich cua minh",
        "co chuong trinh gi hot khong",
        "tim su kien de tham gia",
        "goi y chuong trinh clb",
        "co hoi thao nao ve cntt hay robot khong",
        "hom nay co gi vui khong",
        "su kien the thao nao dang dien ra",
        "co hoat dong ngoai khoa nao khong",
        "tim su kien phu hop voi sinh vien cntt",
        "co workshop nao hay khong",
        "lich hoat dong clb hom nay",
        "co su kien am nhac nao khong",
        "tim hoat dong giai tri tren campus",
    ]
    for q in event_templates:
        raw.append((q, "event_recommend"))
        raw.append((f"xem {q}", "event_recommend"))
        raw.append((f"cho minh biet {q}", "event_recommend"))

    # ---- 4: general_chat ----
    chat_templates = [
        "xin chao", "tro ly campus", "ban la ai", "hello", "hi",
        "cuu toi", "huong dan su dung", "gioi thieu ve app",
        "chuc nang cua app la gi", "tro ly ao campus navigatior",
        "ban giup gi duoc cho toi", "test thu xem", "ok cam on ban",
        "thank you", "bye", "tam biet", "chao buoi sang",
        "ban co the lam gi", "app nay dung de lam gi",
        "toi can giup do", "ho tro toi voi", "ban oi",
        "campus navigator la gi", "gioi thieu he thong",
        "co the giup toi khong", "toi moi vao truong",
        "huong dan cho nguoi moi", "app nay co nhung tinh nang gi",
    ]
    for q in chat_templates:
        raw.append((q, "general_chat"))

    random.shuffle(raw)
    return raw


class IntentClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_vocab(sentences):
    vocab = set()
    for s in sentences:
        for w in s.split():
            vocab.add(w)
    return sorted(vocab)


def text_to_bow(text: str, vocab: list) -> np.ndarray:
    words = text.split()
    vec = np.zeros(len(vocab), dtype=np.float32)
    for w in words:
        if w in vocab:
            vec[vocab.index(w)] += 1.0
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec

# ===========================================================================
# PHẦN 2: CROWD PREDICTOR (Nâng cấp cấp Tòa nhà)
# ===========================================================================
NODES = [
    "Tòa A", "Tòa B", "Tòa C", "Tòa D", "Tòa E", "Tòa F", "Tòa G", 
    "Nhà thể dục", "Nhà xe", "ATM", "Nhà điều hành", "Cổng trường"
]
NODE_MAP = {n: i for i, n in enumerate(NODES)}
WEATHER_TYPES = ["normal", "sunny", "rainy"]
WEATHER_MAP = {w: i for i, w in enumerate(WEATHER_TYPES)}

# INPUT_DIM = len(NODES) + hour_norm + dow_norm + month_norm + is_exam + weather_onehot
# = 12 + 1 + 1 + 1 + 1 + 3 = 19
CROWD_INPUT_DIM = len(NODES) + 6


def _crowd_rule(node: str, hour: float, dow: int, weather: str) -> float:
    """Luật sinh crowd level có nhiễu Gaussian cho các tòa nhà và tiện ích."""
    base = 0.15
    noise = lambda s=0.06: random.gauss(0, s)

    if node == "Tòa D": # Căn tin & Thư viện
        if 11.3 <= hour < 13.2: 
            base = 0.85 # Giờ trưa rất đông ăn uống
        elif (8.5 <= hour < 11.5) or (13.5 <= hour < 17.0): 
            base = 0.70 # Giờ học đông thư viện
        elif 17.0 <= hour < 20.0: 
            base = 0.45
        else: 
            base = 0.15
        if weather == "rainy": 
            base = min(1.0, base + 0.10)

    elif node in ("Tòa B", "Tòa C"): # Tự học & Phòng máy
        if (8.0 <= hour < 11.5) or (13.5 <= hour < 17.0):
            base = 0.75 if random.random() > 0.3 else 0.20
        else: 
            base = 0.10
        if dow >= 5: 
            base = max(0.05, base - 0.35)

    elif node == "Nhà xe":
        if 7.0 <= hour < 8.5:   
            base = 0.85
        elif 11.2 <= hour < 12.5: 
            base = 0.70
        elif 16.5 <= hour < 18.2: 
            base = 0.90
        else: 
            base = 0.12

    elif node == "Nhà thể dục":
        if 16.5 <= hour < 19.5: 
            base = 0.80
        elif dow >= 5 and 7.5 <= hour < 11.0: 
            base = 0.60
        else: 
            base = 0.15
        if weather == "rainy": 
            base = max(0.05, base - 0.20)

    elif node in ("Tòa E", "Tòa F"): # Lý thuyết & Phòng nghỉ
        if 11.5 <= hour < 13.5: 
            base = 0.65 # Giờ nghỉ trưa đông
        elif (7.5 <= hour < 11.5) or (13.5 <= hour < 17.0):
            base = 0.50
        else: 
            base = 0.10

    elif node == "ATM":
        if (8.0 <= hour < 9.5) or (11.5 <= hour < 13.0): 
            base = 0.55
        else: 
            base = 0.15

    elif node == "Cổng trường":
        if (7.0 <= hour < 8.5) or (16.5 <= hour < 18.0): 
            base = 0.75
        else: 
            base = 0.25

    elif node == "Nhà điều hành":
        if (8.0 <= hour < 11.5) or (13.5 <= hour < 16.5): 
            base = 0.50
        else: 
            base = 0.05
        if dow >= 5: 
            base = 0.02

    elif node == "Tòa G":
        if dow >= 5 and 9.0 <= hour < 17.0: 
            base = 0.55
        elif 16.0 <= hour < 19.0: 
            base = 0.40
        else: 
            base = 0.15

    return max(0.0, min(1.0, base + noise()))


def generate_crowd_data(num_samples: int = 4000):
    """
    Sinh dữ liệu crowd với đặc trưng mở rộng cấp Tòa nhà.
    """
    X, y = [], []
    for _ in range(num_samples):
        node    = random.choice(NODES)
        hour    = random.uniform(6.0, 22.0)
        dow     = random.randint(0, 6)
        month   = random.randint(1, 12)
        is_exam = 1.0 if month in (1, 5, 6, 12) else 0.0
        weather = random.choice(WEATHER_TYPES)

        crowd = _crowd_rule(node, hour, dow, weather)
        if is_exam and (node in ("Tòa D", "Tòa C", "Tòa B")):
            crowd = min(1.0, crowd + random.uniform(0.05, 0.15))

        node_vec    = np.zeros(len(NODES), dtype=np.float32)
        node_vec[NODE_MAP[node]] = 1.0
        weather_vec = np.zeros(3, dtype=np.float32)
        weather_vec[WEATHER_MAP[weather]] = 1.0

        feat = np.concatenate([
            node_vec,
            [hour / 24.0, dow / 6.0, (month - 1) / 11.0, is_exam],
            weather_vec,
        ])
        X.append(feat)
        y.append([crowd])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class CrowdPredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

# ===========================================================================
# PHẦN 3: HÀM TRAIN TỔNG HỢP (Loại bỏ NCF)
# ===========================================================================

def train_all(actual_profile: dict = None):
    print("=" * 60)
    print("🚀 [AI Training] Bắt đầu huấn luyện Intent Classifier & Crowd Predictor...")
    print("=" * 60)
    engine_dir = os.path.dirname(os.path.abspath(__file__))
    stats = {}

    # -----------------------------------------------------------------------
    # 1. INTENT CLASSIFIER
    # -----------------------------------------------------------------------
    print("\n📌 [1/2] Huấn luyện Intent Classifier...")
    raw_intent = generate_intent_data()
    random.shuffle(raw_intent)

    norm_texts = [normalize_text(t) for t, _ in raw_intent]
    vocab = build_vocab(norm_texts)

    X_intent = np.array([text_to_bow(normalize_text(t), vocab) for t, _ in raw_intent])
    y_intent = np.array([INTENT_MAP[lbl] for _, lbl in raw_intent])

    X_t = torch.tensor(X_intent, dtype=torch.float32)
    y_t = torch.tensor(y_intent, dtype=torch.long)

    print(f"  Dữ liệu: {len(X_t)} mẫu | Từ vựng: {len(vocab)} từ")

    intent_model = IntentClassifier(vocab_size=len(vocab), num_classes=len(INTENT_LABELS))
    criterion    = nn.CrossEntropyLoss()
    optimizer    = optim.Adam(intent_model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler    = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-5)

    intent_model.train()
    for epoch in range(150):
        optimizer.zero_grad()
        loss = criterion(intent_model(X_t), y_t)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/150 | Loss: {loss.item():.4f}")

    intent_model.eval()
    with torch.no_grad():
        preds = torch.argmax(intent_model(X_t), dim=1)
        acc   = (preds == y_t).float().mean().item()
    print(f"  ✅ Accuracy: {acc*100:.1f}% | Loss cuối: {loss.item():.4f}")
    stats["intent_accuracy"] = round(acc, 4)
    stats["intent_samples"]  = len(X_t)
    stats["vocab_size"]      = len(vocab)

    # -----------------------------------------------------------------------
    # 2. CROWD PREDICTOR
    # -----------------------------------------------------------------------
    print("\n📌 [2/2] Huấn luyện Crowd Predictor...")
    X_crowd, y_crowd = generate_crowd_data(num_samples=4000)
    X_ct = torch.tensor(X_crowd, dtype=torch.float32)
    y_ct = torch.tensor(y_crowd, dtype=torch.float32)

    print(f"  Dữ liệu: {len(X_ct)} mẫu | Input dim: {X_crowd.shape[1]}")

    crowd_model  = CrowdPredictor(input_dim=X_crowd.shape[1])
    criterion_c  = nn.MSELoss()
    optimizer_c  = optim.Adam(crowd_model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler_c  = CosineAnnealingLR(optimizer_c, T_max=200, eta_min=1e-5)

    crowd_model.train()
    for epoch in range(200):
        optimizer_c.zero_grad()
        loss_c = criterion_c(crowd_model(X_ct), y_ct)
        loss_c.backward()
        optimizer_c.step()
        scheduler_c.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/200 | MSE: {loss_c.item():.5f}")

    crowd_model.eval()
    with torch.no_grad():
        preds_c = crowd_model(X_ct)
        mae = torch.mean(torch.abs(preds_c - y_ct)).item()
    print(f"  ✅ MAE: {mae:.4f} | MSE cuối: {loss_c.item():.5f}")
    stats["crowd_mae"]     = round(mae, 4)
    stats["crowd_samples"] = len(X_ct)
    stats["crowd_input_dim"] = int(X_crowd.shape[1])

    # -----------------------------------------------------------------------
    # LƯU TẤT CẢ MODEL VÀ METADATA (Loại bỏ NCF)
    # -----------------------------------------------------------------------
    print("\n💾 Đang lưu trọng số mô hình...")

    torch.save(intent_model.state_dict(), os.path.join(engine_dir, "intent_model.pth"))
    torch.save(crowd_model.state_dict(),  os.path.join(engine_dir, "crowd_model.pth"))

    # Xóa file ncf_model.pth cũ nếu tồn tại
    ncf_file = os.path.join(engine_dir, "ncf_model.pth")
    if os.path.exists(ncf_file):
        try:
            os.remove(ncf_file)
        except Exception:
            pass

    metadata = {
        "intent": {
            "vocab": vocab,
            "labels": INTENT_LABELS,
            "vocab_size": len(vocab),
        },
        "crowd": {
            "nodes": NODES,
            "weather_types": WEATHER_TYPES,
            "input_dim": int(X_crowd.shape[1]),
        },
    }

    with open(os.path.join(engine_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("✅ [AI Training] Hoàn thành! Tóm tắt kết quả:")
    print(f"   Intent Classifier : {stats['intent_accuracy']*100:.1f}% accuracy ({stats['intent_samples']} mẫu)")
    print(f"   Crowd Predictor   : MAE={stats['crowd_mae']:.4f} ({stats['crowd_samples']} mẫu)")
    print("=" * 60)
    return stats


if __name__ == "__main__":
    train_all()
