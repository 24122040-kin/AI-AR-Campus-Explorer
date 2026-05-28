# engine/train_models.py
"""
Script huấn luyện toàn bộ mô hình AI cho hệ thống Campus Navigator.

Các mô hình được train:
  1. IntentClassifier   — Phân loại ý định người dùng (NLP, BoW + MLP)
  2. CrowdPredictor     — Dự báo mật độ đám đông theo giờ/ngày/thời tiết
  3. NCFRecommender     — Neural Collaborative Filtering: học từ lịch sử
                          ghé thăm của người dùng để cá nhân hóa đề xuất

Cải tiến so với phiên bản cũ:
  - NCFRecommender hoàn toàn mới: embedding user × item → MLP → score
  - IntentClassifier: tăng từ 300 → 800+ mẫu, thêm nhiều biến thể ngôn ngữ
  - CrowdPredictor: tăng từ 1200 → 4000 mẫu, thêm đặc trưng tuần/tháng
  - Tất cả model train nhiều epoch hơn với learning rate scheduling
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
    Mỗi nhãn có nhiều template + biến thể ngôn ngữ tự nhiên đa dạng.
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
    """MLP 3 lớp với Dropout — tốt hơn mô hình 2 lớp cũ."""
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
    # L1 normalize
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec

# ===========================================================================
# PHẦN 2: CROWD PREDICTOR (nâng cấp — 4000 mẫu, thêm đặc trưng)
# ===========================================================================
NODES = [
    "Tòa A_Tầng 1_Sảnh", "Tòa A_Tầng 2_Phòng thí nghiệm A201",
    "Tòa A_Tầng 3_Phòng thí nghiệm A301",
    "Tòa B_Tầng 1_Sảnh", "Tòa B_Tầng 2_Tự học B201",
    "Tòa B_Tầng 3_Phòng máy B301",
    "Tòa C_Tầng 1_Sảnh", "Tòa C_Tầng 2_Lab máy tính 202",
    "Tòa C_Tầng 3_Văn phòng khoa",
    "Tòa D_Tầng 1_Căn tin", "Tòa D_Tầng 2_Thư viện",
    "Tòa D_Tầng 3_Quầy giáo trình",
    "Tòa E_Tầng 1_Phòng học 101", "Tòa E_Tầng 2_Phòng nghỉ trưa",
    "Tòa F_Tầng 1_Phòng nghỉ 102", "Tòa F_Tầng 2_Phòng tự học F201",
    "Tòa G", "Nhà thể dục", "Nhà xe", "ATM", "Nhà điều hành", "Cổng trường",
]
NODE_MAP = {n: i for i, n in enumerate(NODES)}
WEATHER_TYPES = ["normal", "sunny", "rainy"]
WEATHER_MAP = {w: i for i, w in enumerate(WEATHER_TYPES)}

# INPUT_DIM = len(NODES) + hour_norm + dow_norm + month_norm + is_holiday + weather_onehot
# = 22 + 1 + 1 + 1 + 1 + 3 = 29
CROWD_INPUT_DIM = len(NODES) + 6


def _crowd_rule(node: str, hour: float, dow: int, weather: str) -> float:
    """Luật sinh crowd level có nhiễu Gaussian."""
    base = 0.15
    noise = lambda s=0.06: random.gauss(0, s)

    if "Căn tin" in node:
        if 7.0 <= hour < 9.0:   base = 0.55
        elif 11.3 <= hour < 13.2: base = 0.90
        elif 16.5 <= hour < 17.5: base = 0.50
        else: base = 0.15
        if weather == "rainy": base = min(1.0, base + 0.12)

    elif "Thư viện" in node or "Tự học" in node:
        if (8.5 <= hour < 11.5) or (13.5 <= hour < 17.0): base = 0.75
        elif 17.0 <= hour < 20.0: base = 0.55
        else: base = 0.20
        if dow >= 5: base = max(0.05, base - 0.35)  # cuối tuần vắng
        if weather == "rainy": base = min(1.0, base + 0.10)

    elif "Nhà xe" in node:
        if 7.0 <= hour < 8.5:   base = 0.80
        elif 11.2 <= hour < 12.5: base = 0.70
        elif 16.5 <= hour < 18.2: base = 0.92
        else: base = 0.12

    elif "Nhà thể dục" in node:
        if 16.5 <= hour < 19.5: base = 0.85
        elif dow >= 5 and 7.5 <= hour < 11.0: base = 0.65
        else: base = 0.15
        if weather == "rainy": base = max(0.05, base - 0.25)

    elif "Lab máy tính" in node or "Phòng thí nghiệm" in node or "Phòng máy" in node:
        if (8.0 <= hour < 11.5) or (13.5 <= hour < 17.0):
            base = 0.85 if random.random() > 0.3 else 0.15  # có/không có lớp
        else: base = 0.08

    elif "Phòng học" in node:
        if (7.5 <= hour < 11.5) or (13.0 <= hour < 17.5):
            base = 0.80 if random.random() > 0.25 else 0.10
        else: base = 0.05

    elif "Phòng nghỉ" in node:
        if 11.5 <= hour < 13.5: base = 0.70
        elif 13.5 <= hour < 14.5: base = 0.55
        else: base = 0.15
        if weather == "rainy": base = min(1.0, base + 0.15)

    elif "ATM" in node:
        if (8.0 <= hour < 9.5) or (11.5 <= hour < 13.0): base = 0.60
        else: base = 0.20

    elif "Cổng trường" in node:
        if (7.0 <= hour < 8.5) or (16.5 <= hour < 18.0): base = 0.75
        else: base = 0.30

    elif "Văn phòng khoa" in node or "Nhà điều hành" in node:
        if 8.0 <= hour < 11.5 or 13.5 <= hour < 16.5: base = 0.55
        else: base = 0.05
        if dow >= 5: base = 0.02

    elif "Tòa G" in node:
        if dow >= 5 and 9.0 <= hour < 17.0: base = 0.60
        elif 16.0 <= hour < 19.0: base = 0.45
        else: base = 0.20

    return max(0.0, min(1.0, base + noise()))


def generate_crowd_data(num_samples: int = 4000):
    """
    Sinh dữ liệu crowd với đặc trưng mở rộng:
      node_onehot(22) + hour_norm + dow_norm + month_norm + is_holiday + weather_onehot(3)
    """
    X, y = [], []
    for _ in range(num_samples):
        node    = random.choice(NODES)
        hour    = random.uniform(6.0, 22.0)
        dow     = random.randint(0, 6)
        month   = random.randint(1, 12)
        # Tháng thi (1, 5, 6, 12) → đông hơn ở thư viện/lab
        is_exam = 1.0 if month in (1, 5, 6, 12) else 0.0
        weather = random.choice(WEATHER_TYPES)

        crowd = _crowd_rule(node, hour, dow, weather)
        # Tháng thi: thư viện/lab đông hơn
        if is_exam and ("Thư viện" in node or "Lab" in node or "Tự học" in node):
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
    """MLP sâu hơn với BatchNorm — ổn định hơn khi train nhiều epoch."""
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
# PHẦN 3: NEURAL COLLABORATIVE FILTERING (NCF) — MỚI HOÀN TOÀN
# ===========================================================================
"""
NCF học từ ma trận tương tác user × item (địa điểm campus).
Kiến trúc: Embedding(user) ⊕ Embedding(item) → MLP → score [0,1]

Dữ liệu huấn luyện được sinh từ các hồ sơ người dùng mẫu đa dạng:
  - Sinh viên CNTT: thích lab, thư viện, phòng máy
  - Sinh viên thể thao: thích nhà thể dục, sân G
  - Sinh viên ăn uống: thích căn tin, ATM
  - Giảng viên: thích văn phòng khoa, thư viện
  - Khách: thích cổng trường, nhà xe, ATM
"""

# Danh sách user profiles mẫu để sinh dữ liệu
USER_PROFILES = [
    # (user_id, role, interests, preferred_nodes)
    ("u_cntt_1",    "student", ["cntt", "code"],
     ["Tòa C_Tầng 2_Lab máy tính 202", "Tòa B_Tầng 3_Phòng máy B301",
      "Tòa D_Tầng 2_Thư viện", "Tòa B_Tầng 2_Tự học B201"]),
    ("u_cntt_2",    "student", ["cntt", "thuat_toan"],
     ["Tòa C_Tầng 2_Lab máy tính 202", "Tòa A_Tầng 2_Phòng thí nghiệm A201",
      "Tòa D_Tầng 2_Thư viện", "Tòa F_Tầng 2_Phòng tự học F201"]),
    ("u_robot_1",   "student", ["robot", "iot"],
     ["Tòa A_Tầng 2_Phòng thí nghiệm A201", "Tòa A_Tầng 3_Phòng thí nghiệm A301",
      "Tòa B_Tầng 3_Phòng máy B301"]),
    ("u_sport_1",   "student", ["the_thao", "gym"],
     ["Nhà thể dục", "Tòa G", "Tòa D_Tầng 1_Căn tin"]),
    ("u_sport_2",   "student", ["bong_da", "cau_long"],
     ["Nhà thể dục", "Tòa G", "Tòa E_Tầng 1_Phòng học 101"]),
    ("u_food_1",    "student", ["an_uong"],
     ["Tòa D_Tầng 1_Căn tin", "ATM", "Tòa D_Tầng 3_Quầy giáo trình"]),
    ("u_quiet_1",   "student", ["hoc_tap", "yen_tinh"],
     ["Tòa D_Tầng 2_Thư viện", "Tòa B_Tầng 2_Tự học B201",
      "Tòa F_Tầng 2_Phòng tự học F201", "Tòa E_Tầng 2_Phòng nghỉ trưa"]),
    ("u_quiet_2",   "student", ["doc_sach", "hoc_tap"],
     ["Tòa D_Tầng 2_Thư viện", "Tòa F_Tầng 2_Phòng tự học F201",
      "Tòa B_Tầng 2_Tự học B201"]),
    ("u_group_1",   "student", ["hoc_nhom"],
     ["Tòa D_Tầng 1_Căn tin", "Tòa G", "Tòa B_Tầng 1_Sảnh",
      "Tòa E_Tầng 1_Phòng học 101"]),
    ("u_english_1", "student", ["english", "ielts"],
     ["Tòa D_Tầng 2_Thư viện", "Tòa B_Tầng 2_Tự học B201",
      "Tòa F_Tầng 2_Phòng tự học F201"]),
    ("u_lecturer_1","lecturer", ["giang_day"],
     ["Tòa C_Tầng 3_Văn phòng khoa", "Nhà điều hành",
      "Tòa D_Tầng 2_Thư viện", "Tòa A_Tầng 2_Phòng thí nghiệm A201"]),
    ("u_lecturer_2","lecturer", ["nghien_cuu"],
     ["Tòa D_Tầng 2_Thư viện", "Tòa C_Tầng 3_Văn phòng khoa",
      "Tòa A_Tầng 3_Phòng thí nghiệm A301"]),
    ("u_visitor_1", "visitor",  ["tham_quan"],
     ["Cổng trường", "Nhà xe", "ATM", "Tòa D_Tầng 1_Căn tin"]),
    ("u_rest_1",    "student",  ["nghi_ngoi"],
     ["Tòa E_Tầng 2_Phòng nghỉ trưa", "Tòa F_Tầng 1_Phòng nghỉ 102",
      "Tòa D_Tầng 1_Căn tin"]),
    ("u_music_1",   "student",  ["am_nhac", "giai_tri"],
     ["Tòa G", "Tòa D_Tầng 1_Căn tin", "Nhà thể dục"]),
]

# Tất cả user IDs và item IDs
ALL_USER_IDS = [p[0] for p in USER_PROFILES] + ["current_user"]
ALL_ITEM_IDS = NODES  # địa điểm campus = items


def generate_ncf_data(num_samples: int = 6000, actual_profile: dict = None):
    """
    Sinh dữ liệu tương tác user-item cho NCF.

    Positive samples: user ghé thăm node trong preferred_nodes → label=1
    Negative samples: user không ghé node không phù hợp → label=0
    Tỉ lệ positive:negative = 1:3 (imbalanced như thực tế)
    """
    user_to_idx = {uid: i for i, uid in enumerate(ALL_USER_IDS)}
    item_to_idx = {nid: i for i, nid in enumerate(ALL_ITEM_IDS)}

    interactions = []  # (user_idx, item_idx, label, context_features)

    for profile in USER_PROFILES:
        uid, role, interests, preferred = profile
        u_idx = user_to_idx[uid]
        preferred_set = set(preferred)

        # Positive samples — ghé thăm nhiều lần với ngữ cảnh khác nhau
        for node in preferred:
            i_idx = item_to_idx[node]
            for _ in range(12):  # 12 lần ghé mỗi node yêu thích
                hour = random.uniform(7.0, 20.0)
                dow  = random.randint(0, 6)
                interactions.append((u_idx, i_idx, 1.0, hour, dow))

        # Negative samples — không ghé các node không phù hợp
        non_preferred = [n for n in ALL_ITEM_IDS if n not in preferred_set]
        neg_count = len(preferred) * 36  # 3× negative
        for node in random.choices(non_preferred, k=neg_count):
            i_idx = item_to_idx[node]
            hour = random.uniform(7.0, 20.0)
            dow  = random.randint(0, 6)
            interactions.append((u_idx, i_idx, 0.0, hour, dow))

    # --- Thêm dữ liệu cho current_user dựa trên hồ sơ thực tế ---
    current_preferred = []
    if actual_profile:
        visited_history = actual_profile.get("visited_history", {})
        for node, count in visited_history.items():
            if node in ALL_ITEM_IDS and count >= 1:
                current_preferred.append(node)
        
        # Giải quyết cold-start: So khớp sở thích của current_user với các profile mẫu
        role = actual_profile.get("role", "student")
        interests = set(i.lower().replace(" ", "_") for i in actual_profile.get("interests", []))
        best_profile = None
        best_overlap = -1
        for p in USER_PROFILES:
            if p[1] != role:
                continue
            p_interests = set(i.lower().replace(" ", "_") for i in p[2])
            overlap = len(interests & p_interests)
            if overlap > best_overlap:
                best_overlap = overlap
                best_profile = p
        
        if best_profile is None:
            for p in USER_PROFILES:
                if p[1] == role:
                    best_profile = p
                    break
        if best_profile is None:
            best_profile = USER_PROFILES[0]
            
        # Gộp các địa điểm ưa thích từ profile mẫu tốt nhất
        for node in best_profile[3]:
            if node not in current_preferred:
                current_preferred.append(node)
    else:
        # Mặc định sử dụng các địa điểm của sinh viên CNTT
        current_preferred = list(USER_PROFILES[0][3])
        
    u_idx = user_to_idx["current_user"]
    current_preferred_set = set(current_preferred)
    
    # Sinh positive samples cho current_user
    for node in current_preferred:
        i_idx = item_to_idx[node]
        # Nếu có trong lịch sử ghé thăm của actual_profile, ta có thể tăng thêm số lượng mẫu dựa theo số lần ghé
        visit_mult = 1
        if actual_profile:
            visit_mult = max(1, min(4, actual_profile.get("visited_history", {}).get(node, 0)))
        
        for _ in range(12 * visit_mult):
            hour = random.uniform(7.0, 20.0)
            dow  = random.randint(0, 6)
            interactions.append((u_idx, i_idx, 1.0, hour, dow))
            
    # Sinh negative samples cho current_user
    non_preferred = [n for n in ALL_ITEM_IDS if n not in current_preferred_set]
    neg_count = len(current_preferred) * 36
    for node in random.choices(non_preferred, k=neg_count):
        i_idx = item_to_idx[node]
        hour = random.uniform(7.0, 20.0)
        dow  = random.randint(0, 6)
        interactions.append((u_idx, i_idx, 0.0, hour, dow))

    # Thêm noise: đôi khi user ghé node không phải sở thích (khám phá)
    for _ in range(num_samples // 10):
        profile = random.choice(USER_PROFILES + [("current_user", "student", [], current_preferred)])
        uid = profile[0]
        u_idx = user_to_idx[uid]
        node = random.choice(ALL_ITEM_IDS)
        i_idx = item_to_idx[node]
        hour = random.uniform(7.0, 20.0)
        dow  = random.randint(0, 6)
        # Label thấp (0.3) — ghé thăm ngẫu nhiên, không phải yêu thích
        interactions.append((u_idx, i_idx, 0.3, hour, dow))

    random.shuffle(interactions)
    return interactions, user_to_idx, item_to_idx


class NCFRecommender(nn.Module):
    """
    Neural Collaborative Filtering.
    Kiến trúc: GMF (element-wise product) + MLP → concat → output layer

    user_embed ⊗ item_embed  (GMF path)
    [user_embed ⊕ item_embed] → MLP  (MLP path)
    concat(GMF, MLP) → Linear(1) → Sigmoid
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 32,
        mlp_layers: list = None,
        context_dim: int = 2,  # hour_norm + dow_norm
    ):
        super().__init__()
        mlp_layers = mlp_layers or [128, 64, 32]

        # GMF embeddings
        self.user_embed_gmf  = nn.Embedding(num_users, embed_dim)
        self.item_embed_gmf  = nn.Embedding(num_items, embed_dim)

        # MLP embeddings
        self.user_embed_mlp  = nn.Embedding(num_users, embed_dim)
        self.item_embed_mlp  = nn.Embedding(num_items, embed_dim)

        # MLP path: input = 2*embed_dim + context_dim
        mlp_input = embed_dim * 2 + context_dim
        layers = []
        for out_dim in mlp_layers:
            layers += [nn.Linear(mlp_input, out_dim), nn.ReLU(), nn.Dropout(0.2)]
            mlp_input = out_dim
        self.mlp = nn.Sequential(*layers)

        # Output: GMF(embed_dim) + MLP(last_layer) → 1
        self.output = nn.Linear(embed_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

        # Khởi tạo trọng số
        nn.init.normal_(self.user_embed_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embed_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embed_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embed_mlp.weight, std=0.01)

    def forward(self, user_ids, item_ids, context):
        # GMF path
        u_gmf = self.user_embed_gmf(user_ids)
        i_gmf = self.item_embed_gmf(item_ids)
        gmf_out = u_gmf * i_gmf  # element-wise product

        # MLP path
        u_mlp = self.user_embed_mlp(user_ids)
        i_mlp = self.item_embed_mlp(item_ids)
        mlp_in = torch.cat([u_mlp, i_mlp, context], dim=1)
        mlp_out = self.mlp(mlp_in)

        # Concat và output
        combined = torch.cat([gmf_out, mlp_out], dim=1)
        return self.sigmoid(self.output(combined)).squeeze(1)

# ===========================================================================
# PHẦN 4: HÀM TRAIN TỔNG HỢP
# ===========================================================================

def train_all(actual_profile: dict = None):
    """
    Huấn luyện toàn bộ 3 mô hình và lưu vào engine/.
    Trả về dict thống kê kết quả.
    """
    print("=" * 60)
    print("🚀 [AI Training] Bắt đầu huấn luyện toàn bộ mô hình...")
    print("=" * 60)
    engine_dir = os.path.dirname(os.path.abspath(__file__))
    stats = {}

    # -----------------------------------------------------------------------
    # 1. INTENT CLASSIFIER
    # -----------------------------------------------------------------------
    print("\n📌 [1/3] Huấn luyện Intent Classifier...")
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
    print("\n📌 [2/3] Huấn luyện Crowd Predictor...")
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
    # 3. NCF RECOMMENDER
    # -----------------------------------------------------------------------
    print("\n📌 [3/3] Huấn luyện NCF Recommender...")
    interactions, user_to_idx, item_to_idx = generate_ncf_data(num_samples=6000, actual_profile=actual_profile)

    num_users = len(ALL_USER_IDS)
    num_items = len(ALL_ITEM_IDS)
    print(f"  Users: {num_users} | Items: {num_items} | Interactions: {len(interactions)}")

    # Chuẩn bị tensors
    u_ids    = torch.tensor([x[0] for x in interactions], dtype=torch.long)
    i_ids    = torch.tensor([x[1] for x in interactions], dtype=torch.long)
    labels   = torch.tensor([x[2] for x in interactions], dtype=torch.float32)
    contexts = torch.tensor(
        [[x[3] / 24.0, x[4] / 6.0] for x in interactions],
        dtype=torch.float32,
    )

    ncf_model   = NCFRecommender(
        num_users=num_users,
        num_items=num_items,
        embed_dim=32,
        mlp_layers=[128, 64, 32],
        context_dim=2,
    )
    criterion_n = nn.BCELoss()
    optimizer_n = optim.Adam(ncf_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler_n = CosineAnnealingLR(optimizer_n, T_max=100, eta_min=1e-5)

    # Mini-batch training
    batch_size = 256
    n = len(interactions)
    ncf_model.train()
    for epoch in range(100):
        # Shuffle
        perm = torch.randperm(n)
        u_ids    = u_ids[perm]
        i_ids    = i_ids[perm]
        labels   = labels[perm]
        contexts = contexts[perm]

        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            bu, bi, bl, bc = u_ids[start:end], i_ids[start:end], labels[start:end], contexts[start:end]
            optimizer_n.zero_grad()
            preds_n = ncf_model(bu, bi, bc)
            loss_n  = criterion_n(preds_n, bl)
            loss_n.backward()
            optimizer_n.step()
            epoch_loss += loss_n.item() * (end - start)

        scheduler_n.step()
        if (epoch + 1) % 25 == 0:
            avg_loss = epoch_loss / n
            print(f"  Epoch {epoch+1}/100 | BCE Loss: {avg_loss:.4f}")

    # Đánh giá
    ncf_model.eval()
    with torch.no_grad():
        all_preds = ncf_model(u_ids, i_ids, contexts)
        binary_preds = (all_preds >= 0.5).float()
        binary_labels = (labels >= 0.5).float()
        ncf_acc = (binary_preds == binary_labels).float().mean().item()
    print(f"  ✅ NCF Accuracy: {ncf_acc*100:.1f}%")
    stats["ncf_accuracy"]    = round(ncf_acc, 4)
    stats["ncf_interactions"] = len(interactions)

    # -----------------------------------------------------------------------
    # LƯU TẤT CẢ MODEL VÀ METADATA
    # -----------------------------------------------------------------------
    print("\n💾 Đang lưu trọng số mô hình...")

    torch.save(intent_model.state_dict(), os.path.join(engine_dir, "intent_model.pth"))
    torch.save(crowd_model.state_dict(),  os.path.join(engine_dir, "crowd_model.pth"))
    torch.save(ncf_model.state_dict(),    os.path.join(engine_dir, "ncf_model.pth"))

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
        "ncf": {
            "user_ids": ALL_USER_IDS,
            "user_to_idx": user_to_idx,
            "item_to_idx": item_to_idx,
            "num_users": num_users,
            "num_items": num_items,
            "embed_dim": 32,
            "mlp_layers": [128, 64, 32],
            "context_dim": 2,
            # Lưu profile để inference
            "user_profiles": [
                {"user_id": p[0], "role": p[1], "interests": p[2], "preferred_nodes": p[3]}
                for p in USER_PROFILES
            ],
        },
    }

    with open(os.path.join(engine_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("✅ [AI Training] Hoàn thành! Tóm tắt kết quả:")
    print(f"   Intent Classifier : {stats['intent_accuracy']*100:.1f}% accuracy ({stats['intent_samples']} mẫu)")
    print(f"   Crowd Predictor   : MAE={stats['crowd_mae']:.4f} ({stats['crowd_samples']} mẫu)")
    print(f"   NCF Recommender   : {stats['ncf_accuracy']*100:.1f}% accuracy ({stats['ncf_interactions']} interactions)")
    print("=" * 60)
    return stats


if __name__ == "__main__":
    train_all()
