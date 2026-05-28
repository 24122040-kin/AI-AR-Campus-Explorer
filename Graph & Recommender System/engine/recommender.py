# engine/recommender.py
import math
from collections import Counter
from datetime import time
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from engine.building_catalog import (
    build_function_reason,
    enrich_suggestion,
    get_building_profile,
    list_all_building_guides,
    match_services_to_query,
)
from engine.optimizer import is_node_open, pathfinding_optimizer
from engine.nlp_processor import normalize_text
from engine.utils import haversine, parse_time
from engine.context_features import (
    get_time_of_day_band,
    indoor_boost,
    is_weekend,
    open_status_detail,
    time_category_boost,
    effective_crowd_level,
)
from engine.campus_knowledge import report_live_crowd, get_live_crowd

_MAX_RAW_SCORE = 60.0
_MAX_PROACTIVE = 6
_DETOUR_RADIUS_M = 120.0
_NEARBY_RADIUS_M = 350.0


def _normalize_score(raw: float) -> float:
    """Chuẩn hóa điểm thô về thang [0, 100]."""
    normalized = (raw / _MAX_RAW_SCORE) * 100
    return round(max(0.0, min(100.0, normalized)), 1)


# =====================================================================
# LỚP AI: TF-IDF + Cosine Similarity (không cần API ngoài)
# =====================================================================
class CampusSemanticAI:
    """Chỉ mục ngữ nghĩa cho toàn bộ node campus — dùng TF-IDF thuần NumPy."""

    def __init__(self, G: nx.Graph):
        self._nodes: List[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._vocab: List[str] = []
        self._idf: Optional[np.ndarray] = None
        self._build(G)

    @staticmethod
    def _node_document(node: str, data: dict) -> str:
        aliases = " ".join(data.get("aliases", []))
        features = data.get("features", {})
        tags: List[str] = []
        if features.get("has_ac"):
            tags += ["mat me", "may lanh", "dieu hoa", "lanh"]
        if features.get("has_tables"):
            tags += ["ban ghe", "hoc bai", "tu hoc", "ngoi hoc"]
        noise = features.get("noise_level", 0.5)
        if noise <= 0.3:
            tags += ["yen tinh", "on a", "tap trung", "doc sach"]
        elif noise >= 0.7:
            tags += ["on ao", "dong vui"]
        node_type = data.get("type", "")
        if node_type == "facility":
            tags += ["tien ich", "phuc vu"]
        for svc in data.get("services", []):
            tags.append(normalize_text(svc.get("name", "")))
            tags.extend(normalize_text(k) for k in svc.get("keywords", []))
        if data.get("tagline"):
            tags.append(normalize_text(data["tagline"]))
        from engine.campus_knowledge import REVIEW_SIGNALS
        for sig in REVIEW_SIGNALS.get(node, []):
            tags.extend(normalize_text(k) for k in sig.get("keywords", []))
        return normalize_text(f"{node} {aliases} {' '.join(tags)}")

    def _build(self, G: nx.Graph) -> None:
        docs: List[Counter] = []
        vocab: set = set()
        self._nodes = list(G.nodes())

        for node in self._nodes:
            tokens = self._node_document(node, G.nodes[node]).split()
            counter = Counter(t for t in tokens if t)
            docs.append(counter)
            vocab.update(counter.keys())

        self._vocab = sorted(vocab)
        if not self._vocab:
            self._matrix = np.zeros((len(self._nodes), 0))
            return

        n_docs = len(docs)
        df = np.zeros(len(self._vocab), dtype=float)
        for counter in docs:
            for idx, term in enumerate(self._vocab):
                if term in counter:
                    df[idx] += 1.0

        self._idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0
        matrix = np.zeros((n_docs, len(self._vocab)), dtype=float)

        for row, counter in enumerate(docs):
            total = sum(counter.values()) or 1
            for idx, term in enumerate(self._vocab):
                if term in counter:
                    tf = counter[term] / total
                    matrix[row, idx] = tf * self._idf[idx]

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms

    def score_query(self, query: str) -> Dict[str, float]:
        if not query or self._matrix is None or self._matrix.size == 0:
            return {}

        tokens = normalize_text(query).split()
        if not tokens:
            return {}

        counter = Counter(tokens)
        total = sum(counter.values()) or 1
        q_vec = np.zeros(len(self._vocab), dtype=float)
        for idx, term in enumerate(self._vocab):
            if term in counter:
                tf = counter[term] / total
                q_vec[idx] = tf * self._idf[idx]

        norm = np.linalg.norm(q_vec)
        if norm == 0:
            return {}
        q_vec /= norm

        sims = self._matrix @ q_vec
        return {self._nodes[i]: float(sims[i]) for i in range(len(self._nodes))}


def _nearest_node(G: nx.Graph, lat: float, lon: float) -> Tuple[str, float]:
    nearest = min(G.nodes(), key=lambda n: haversine(lat, lon, *G.nodes[n]["gps"]))
    dist = haversine(lat, lon, *G.nodes[nearest]["gps"])
    return nearest, dist


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_lam = math.radians(lon2 - lon1)
    x = math.sin(d_lam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lam)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _angle_diff(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def _geo_alignment_score(
    user_lat: float,
    user_lon: float,
    node_lat: float,
    node_lon: float,
    dest_lat: float,
    dest_lon: float,
) -> float:
    """Điểm cao hơn khi node nằm hướng về phía đích (phù hợp ghé dọc đường)."""
    to_node = _bearing_deg(user_lat, user_lon, node_lat, node_lon)
    to_dest = _bearing_deg(user_lat, user_lon, dest_lat, dest_lon)
    diff = _angle_diff(to_node, to_dest)
    if diff <= 35:
        return 18.0
    if diff <= 70:
        return 8.0
    return -5.0


def _extract_rule_needs(query: str) -> dict:
    q = normalize_text(query)
    return {
        "ac": any(w in q for w in ["mat", "may lanh", "nong", "dieu hoa"]),
        "quiet": any(w in q for w in ["yen tinh", "on a", "hoc bai", "doc sach", "tap trung"]),
        "tables": any(w in q for w in ["hoc", "ngoi", "lam bai", "ban ghe", "tu hoc"]),
        "food": any(w in q for w in ["doi", "an", "uong", "cafe", "ca phe", "com", "nuoc", "canteen"]),
        "sport": any(w in q for w in ["the thao", "tap", "van dong", "gym", "cau long", "bong ban", "the duc"]),
        "rest": any(w in q for w in ["ngu", "nghi ngoi", "met", "nga lung", "buon ngu"]),
    }


def _rule_based_score(G: nx.Graph, node: str, needs: dict, weather: str) -> float:
    if weather == "sunny" and not needs["ac"]:
        needs = {**needs, "ac": True}

    if not any(needs.values()):
        return 0.0

    data = G.nodes[node]
    features = data.get("features", {})
    aliases = " ".join(data.get("aliases", []))
    score = 0.0

    if needs["ac"]:
        score += 15 if features.get("has_ac") else -15
    if needs["tables"]:
        score += 10 if features.get("has_tables") else -10
    if needs["quiet"]:
        noise = features.get("noise_level", 1.0)
        if noise <= 0.3:
            score += 15
        elif noise >= 0.7:
            score -= 20
        else:
            score += (1.0 - noise) * 10
    if needs["rest"]:
        if "nghi" in aliases or "ngu" in aliases:
            score += 25
        if features.get("noise_level", 1.0) <= 0.3:
            score += 5
        if features.get("has_ac"):
            score += 5
    if needs["food"]:
        if "can tin" in aliases or "an" in aliases or "doi bung" in aliases:
            score += 30
        else:
            score -= 30
    if needs["sport"]:
        if "the thao" in aliases or "the duc" in aliases or "gym" in aliases:
            score += 30
        else:
            score -= 30

    score += (min(features.get("capacity", 0), 1000) / 1000) * 5
    return score


def _interest_score(node: str, data: dict, interests: List[str]) -> float:
    if not interests:
        return 0.0
    interests_str = " ".join(interests).lower()
    aliases = " ".join(data.get("aliases", [])).lower()
    score = 0.0

    if any(k in interests_str for k in ["c++", "codeforces", "sql", "code", "thuat toan"]):
        if any(k in aliases for k in ["may tinh", "lab", "thuc hanh", "nha c"]):
            score += 50
    if any(k in interests_str for k in ["genshin", "tft", "game", "esport"]):
        if any(k in aliases for k in ["phong nghi", "canteen"]):
            score += 20
    if any(k in interests_str for k in ["football", "bayern", "chelsea", "the thao"]):
        if any(k in aliases for k in ["the duc", "the thao", "gym"]):
            score += 45
    return score if score > 0 else 0.0


def _build_reason(
    G: nx.Graph,
    node: str,
    semantic: float,
    on_route: bool,
    dist_m: float,
    crowd: float,
    destination: Optional[str],
) -> str:
    parts: List[str] = []
    if destination and on_route:
        parts.append(f"Nằm trên hướng đi tới {destination}")
    elif dist_m < 80:
        parts.append(f"Rất gần vị trí bạn (~{int(dist_m)}m)")
    elif dist_m < 300:
        parts.append(f"Gần vị trí hiện tại (~{int(dist_m)}m)")

    if semantic >= 0.35:
        parts.append("khớp nhu cầu bạn mô tả")
    if crowd >= 0.85:
        parts.append("dự báo đông, nên ghé sớm")
    elif crowd <= 0.25:
        parts.append("dự báo vắng, thoải mái")

    ctx = ""
    if parts:
        ctx = parts[0].capitalize() + (", " + ", ".join(parts[1:]) if len(parts) > 1 else "")

    profile = get_building_profile(G, node)
    func = profile.get("function_summary", "")
    if func:
        if ctx:
            return f"{ctx} — Tại đây: {func}."
        return f"Ghé {node}: {func}."

    if not ctx:
        return f"Gợi ý AI: ghé {node} trước khi tiếp tục di chuyển."
    return ctx + "."


# =====================================================================
# TASK 1: SEMANTIC & INTENT RECOMMENDER (nhiều kết quả + AI)
# =====================================================================
def recommend_locations(
    G: nx.Graph,
    query: str,
    current_time: str = None,
    weather: str = "normal",
    limit: int = 5,
) -> List[dict]:
    """Trả về danh sách địa điểm xếp hạng theo AI ngữ nghĩa + luật nhu cầu."""
    query_norm = normalize_text(query)
    if not query_norm:
        return []

    semantic_ai = CampusSemanticAI(G)
    semantic_scores = semantic_ai.score_query(query)
    needs = _extract_rule_needs(query)

    ranked: List[dict] = []
    time_band = get_time_of_day_band(current_time) if current_time else "afternoon"
    weekend = is_weekend()

    for node, data in G.nodes(data=True):
        if not is_node_open(G, node, current_time):
            continue

        sem = semantic_scores.get(node, 0.0) * 40.0
        rule = _rule_based_score(G, node, dict(needs), weather)
        raw = sem + rule + time_category_boost(node, G, time_band, weekend)
        raw += indoor_boost(G, node, weather)
        if raw <= 0 and sem < 8:
            continue

        open_info = open_status_detail(G, node, current_time) if current_time else {}
        ranked.append({
            "node": node,
            "score": _normalize_score(max(raw, sem)),
            "raw_score": round(raw, 2),
            "semantic_score": round(sem, 2),
            "method": "AI Semantic + Intent",
            "reason": _build_reason(G, node, sem / 40.0, False, 9999, 0.0, None),
            "open_now": open_info.get("open_now", True),
            "closing_soon": open_info.get("closing_soon", False),
            "close_warning": open_info.get("warning"),
        })

    ranked.sort(key=lambda x: (x["raw_score"], x["semantic_score"]), reverse=True)
    return [enrich_suggestion(G, r, query) for r in ranked[:limit]]


def recommend_location(
    G: nx.Graph,
    query: str,
    current_time: str = None,
    weather: str = "normal",
) -> Tuple[Optional[str], float]:
    results = recommend_locations(G, query, current_time, weather, limit=1)
    if not results:
        return None, 0
    top = results[0]
    return top["node"], top["score"]


# =====================================================================
# TASK 2: GỢI Ý THEO TỌA ĐỘ + ĐIỂM ĐÍCH (AI tổng hợp)
# =====================================================================
def _personalization_boost(
    G: nx.Graph,
    node: str,
    role: str,
    study_style: str,
    active_interests: List[str]
) -> Tuple[float, List[str]]:
    """Tính điểm cộng cá nhân hóa dựa trên role, study_style và interests."""
    boost = 0.0
    reasons = []

    data = G.nodes.get(node, {})
    aliases = " ".join(data.get("aliases", [])).lower()
    node_type = data.get("type", "")
    features = data.get("features", {})
    noise = features.get("noise_level", 0.5)
    capacity = features.get("capacity", 0)

    # 1. Boost theo Role
    if role == "student":
        if node_type == "building":
            boost += 5.0
        if any(w in aliases for w in ["thu vien", "tu hoc", "can tin", "gym", "phong hoc"]):
            boost += 12.0
            reasons.append("Sinh viên")
    elif role == "lecturer":
        if node_type == "admin":
            boost += 20.0
            reasons.append("Hành chính/Giảng viên")
        if "van phong khoa" in aliases or "vp khoa" in aliases or "giao vu" in aliases:
            boost += 15.0
            reasons.append("VP Khoa")
        if "thu vien" in aliases:
            boost += 8.0
    elif role == "visitor":
        if node_type == "facility":
            boost += 10.0
        if any(w in aliases for w in ["cong truong", "nha xe", "atm", "can tin"]):
            boost += 15.0
            reasons.append("Khách tham quan")

    # 2. Boost theo Study Style
    if study_style == "silent":
        if noise <= 0.3:
            boost += 12.0
            reasons.append("Không gian yên tĩnh")
        elif noise >= 0.7:
            boost -= 15.0
        if any(w in aliases for w in ["thu vien", "tu hoc"]):
            boost += 10.0
    elif study_style == "group":
        if 0.4 <= noise <= 0.6 or capacity >= 100:
            boost += 10.0
            reasons.append("Học nhóm/Thảo luận")
        if any(w in aliases for w in ["sanh", "can tin", "phong nghi", "nha g"]):
            boost += 8.0

    # 3. Boost theo Interests
    if active_interests:
        interests_str = " ".join(active_interests).lower()
        if any(k in interests_str for k in ["c++", "codeforces", "sql", "code", "thuat toan", "lap trinh", "cntt"]):
            if any(k in aliases for k in ["may tinh", "lab", "thuc hanh", "nha c"]):
                boost += 25.0
                reasons.append("Sở thích CNTT")
        if any(k in interests_str for k in ["robot", "iot", "arduino", "dien tu"]):
            if any(k in aliases for k in ["phong thi nghiem", "nha a", "lab"]):
                boost += 20.0
                reasons.append("Sở thích Robotics")
        if any(k in interests_str for k in ["football", "the thao", "bong da", "cau long", "gym"]):
            if any(k in aliases for k in ["the duc", "the thao", "gym"]):
                boost += 20.0
                reasons.append("Đam mê Thể thao")
        if any(k in interests_str for k in ["english", "ielts", "ngoai ngu"]):
            if any(k in aliases for k in ["thu vien", "tu hoc"]):
                boost += 15.0
                reasons.append("Học Ngoại ngữ")

    return boost, reasons


def get_smart_recommendations(
    G: nx.Graph,
    current_lat: float,
    current_lon: float,
    destination: Optional[str] = None,
    query: Optional[str] = None,
    weather: str = "normal",
    current_time_str: Optional[str] = None,
    user_interests: Optional[List[str]] = None,
    limit: int = _MAX_PROACTIVE,
    user_profile: Optional[dict] = None,
) -> List[dict]:
    """
    Gợi ý thông minh dựa trên GPS, điểm đến, câu hỏi tự nhiên, sở thích và lịch sử di chuyển.
    """
    nearest_node, dist_nearest = _nearest_node(G, current_lat, current_lon)
    curr_t = parse_time(current_time_str) if current_time_str else None
    if not curr_t:
        return []

    if user_profile is None:
        user_profile = {}

    role = user_profile.get("role", "student")
    study_style = user_profile.get("study_style", "silent")
    profile_interests = user_profile.get("interests", [])
    visited_history = user_profile.get("visited_history", {})
    active_interests = list(set((user_interests or []) + (profile_interests or [])))

    semantic_ai = CampusSemanticAI(G) if query else None
    semantic_scores = semantic_ai.score_query(query) if semantic_ai and query else {}
    needs = _extract_rule_needs(query) if query else {}

    path_nodes: set = set()
    dest_gps = None

    if destination and destination in G.nodes and destination != nearest_node:
        path, _ = pathfinding_optimizer(G, nearest_node, destination, weather, current_time_str)
        if path:
            path_nodes = set(path)
            dest_gps = G.nodes[destination]["gps"]

    # --- Tính NCF scores một lần cho toàn bộ session ---
    ncf_scores: Dict[str, float] = {}
    if _NCF_MODEL is not None and user_profile:
        try:
            ncf_results = ncf_recommend(G, user_profile, current_time_str, limit=len(G.nodes))
            ncf_scores = {r["node"]: r["ncf_score"] for r in ncf_results}
        except Exception:
            pass

    candidates: Dict[str, dict] = {}

    def _add(node: str, raw: float, reason: str, source: str, priority: int = 5) -> None:
        if node == nearest_node and dist_nearest < 30 and source != "nearby_context" and source != "personal_history":
            return
        if not is_node_open(G, node, current_time_str):
            return

        # Tính điểm cộng cá nhân hóa
        p_boost, p_reasons = _personalization_boost(G, node, role, study_style, active_interests)
        raw += p_boost

        # Điểm cộng NCF — học từ lịch sử tương tác
        ncf_s = ncf_scores.get(node, 0.0)
        ncf_boost = 0.0
        if ncf_s > 0.5:
            ncf_boost = (ncf_s - 0.5) * 40.0  # max +20 điểm khi ncf_score=1.0
            raw += ncf_boost

        # Điểm cộng lịch sử ghé thăm
        visit_count = visited_history.get(node, 0)
        history_boost = 0.0
        if visit_count > 0:
            history_boost = min(15.0, visit_count * 3.0)
            raw += history_boost

        entry = candidates.get(node)
        if entry and entry["raw_score"] >= raw:
            return

        dist_m = haversine(current_lat, current_lon, *G.nodes[node]["gps"])
        crowd = effective_crowd_level(
            G, node, predict_crowd_level(G, node, current_time_str)
        )
        open_info = open_status_detail(G, node, current_time_str)

        # Điều chỉnh lý do nếu có các thẻ cá nhân hóa
        enhanced_reason = reason
        if p_reasons:
            p_desc = " & ".join(p_reasons[:2])
            enhanced_reason = f"{reason.rstrip('.')} ({p_desc})."

        # Tách nhỏ điểm số thành phần để phục vụ Explainable AI (XAI)
        persona_raw = p_boost + ncf_boost + history_boost
        if source == "personal_history":
            persona_raw += 25.0
        elif source == "ai_context":
            persona_raw += (_interest_score(node, G.nodes[node], active_interests or []) * 0.4)

        proximity_raw = 0.0
        if source == "nearby_context":
            proximity_raw = 28.0
        elif source in ("route_context", "route_neighbor"):
            proximity_raw = raw - p_boost - ncf_boost - history_boost
            if source == "route_context":
                crowd_val = predict_crowd_level(G, node, current_time_str)
                crowd_adj_val = -8 if crowd_val >= 0.85 else 4
                proximity_raw -= crowd_adj_val
        elif source == "ai_context":
            prox_bonus = 12 if dist_m < 50 else (6 if dist_m < 150 else 0)
            align_val = 0.0
            if dest_gps:
                align_val = _geo_alignment_score(current_lat, current_lon, G.nodes[node]["gps"][0], G.nodes[node]["gps"][1], dest_gps[0], dest_gps[1])
            sem_val = (semantic_scores.get(node, 0.0) if semantic_scores else 0.0) * 35.0
            rule_val = _rule_based_score(G, node, dict(needs), weather) if query else 0.0
            proximity_raw = prox_bonus + align_val + sem_val + rule_val

        context_raw = 0.0
        if source in ("time_morning", "time_context"):
            context_raw = raw - p_boost - ncf_boost - history_boost
        elif source == "ai_context":
            context_raw = time_category_boost(node, G, time_band, weekend) + indoor_boost(G, node, weather)

        crowd_raw = 0.0
        if source == "route_context":
            crowd_val = predict_crowd_level(G, node, current_time_str)
            crowd_raw = -8 if crowd_val >= 0.85 else 4
        elif source == "ai_context":
            crowd_val = predict_crowd_level(G, node, current_time_str)
            crowd_raw = 5 if crowd_val <= 0.35 else (-6 if crowd_val >= 0.85 else 0)
        else:
            crowd_val = predict_crowd_level(G, node, current_time_str)
            crowd_raw = 5 if crowd_val <= 0.35 else (-6 if crowd_val >= 0.85 else 0)

        score_breakdown = {
            "personalization": round(max(0.0, min(100.0, (persona_raw / 30.0) * 100)), 1),
            "proximity": round(max(0.0, min(100.0, (proximity_raw / 25.0) * 100)), 1),
            "context": round(max(0.0, min(100.0, (context_raw / 20.0) * 100)), 1),
            "crowd": round(max(0.0, min(100.0, ((crowd_raw + 8.0) / 13.0) * 100)), 1),
        }

        candidates[node] = {
            "node": node,
            "raw_score": raw,
            "score": _normalize_score(raw),
            "reason": enhanced_reason,
            "priority": priority,
            "distance_m": round(dist_m, 1),
            "crowd_level": round(crowd, 2),
            "on_route": node in path_nodes and node != destination,
            "source": source,
            "gps": G.nodes[node]["gps"],
            "closing_soon": open_info.get("closing_soon", False),
            "close_warning": open_info.get("warning"),
            "time_band": time_band,
            "personal_tags": p_reasons,
            "ncf_score": round(ncf_scores.get(node, 0.0), 4),
            "score_breakdown": score_breakdown,
        }

    # --- Gợi ý theo thời gian (Time of Day + Open Now) ---
    def in_time_range(start_str: str, end_str: str) -> bool:
        return parse_time(start_str) <= curr_t <= parse_time(end_str)

    time_band = get_time_of_day_band(current_time_str)
    weekend = is_weekend()

    # Sáng sớm 6h–9h: ưu tiên ăn sáng / căn tin
    if time_band == "early_morning":
        for fn, d in G.nodes(data=True):
            if "can tin" not in " ".join(d.get("aliases", [])).lower():
                continue
            dist_m = haversine(current_lat, current_lon, *d["gps"])
            if dist_m < _NEARBY_RADIUS_M:
                _add(
                    fn, 36,
                    build_function_reason(G, fn, query, "Buổi sáng — ghé ăn sáng/cà phê campus"),
                    "time_morning", 1,
                )

    if in_time_range("16:30", "18:30") and "Nhà xe" in G.nodes:
        _add(
            "Nhà xe", 42,
            build_function_reason(G, "Nhà xe", query, "Sắp hết giờ chiều — ra lấy xe về"),
            "time_context", 1,
        )

    if in_time_range("11:00", "13:00") or in_time_range("17:00", "18:30"):
        for fn, d in G.nodes(data=True):
            has_food = any(
                s.get("category") == "an_uong"
                for s in d.get("services", [])
            ) or "can tin" in " ".join(d.get("aliases", [])).lower()
            if not has_food:
                continue
            dist_m = haversine(current_lat, current_lon, *d["gps"])
            if dist_m < _NEARBY_RADIUS_M:
                _add(
                    fn, 38,
                    build_function_reason(G, fn, query, "Đã đến giờ ăn — ghé căn tin"),
                    "time_context", 2,
                )

    # --- Gợi ý từ Lịch sử ghé thăm (Personalized History) ---
    for node, count in visited_history.items():
        if node not in G.nodes:
            continue
        dist_m = haversine(current_lat, current_lon, *G.nodes[node]["gps"])
        if count >= 1 and dist_m < _NEARBY_RADIUS_M * 2.0:
            h_boost = min(15.0, count * 3.0)
            raw = 25.0 + h_boost
            reason = f"Bạn thường ghé thăm địa điểm này (đã đi {count} lần)"
            _add(node, raw, reason, "personal_history", 2)

    # --- Gợi ý dọc lộ trình tới đích ---
    if path_nodes and dest_gps:
        dest_lat, dest_lon = dest_gps
        for node in path_nodes:
            if node == destination or node == nearest_node:
                continue
            dist_m = haversine(current_lat, current_lon, *G.nodes[node]["gps"])
            align = _geo_alignment_score(
                current_lat, current_lon,
                G.nodes[node]["gps"][0], G.nodes[node]["gps"][1],
                dest_lat, dest_lon,
            )
            detour_bonus = 25 if dist_m < _DETOUR_RADIUS_M else 12
            crowd = predict_crowd_level(G, node, current_time_str)
            crowd_adj = -8 if crowd >= 0.85 else 4
            raw = 30 + detour_bonus + align + crowd_adj
            reason = _build_reason(G, node, 0, True, dist_m, crowd, destination)
            _add(node, raw, reason, "route_context", 3)

        # Điểm lân cận lộ trình (1 hop từ path)
        for u, v in G.edges():
            for a, b in ((u, v), (v, u)):
                if a not in path_nodes or b in path_nodes:
                    continue
                dist_m = haversine(current_lat, current_lon, *G.nodes[b]["gps"])
                if dist_m > _NEARBY_RADIUS_M:
                    continue
                raw = 22 + _geo_alignment_score(
                    current_lat, current_lon,
                    G.nodes[b]["gps"][0], G.nodes[b]["gps"][1],
                    dest_lat, dest_lon,
                )
                _add(
                    b, raw,
                    build_function_reason(
                        G, b, query,
                        f"Ngay cạnh lộ trình tới {destination} — tiện ghé",
                    ),
                    "route_neighbor", 4,
                )

    # --- Gợi ý theo câu hỏi / sở thích + vị trí ---
    for node, data in G.nodes(data=True):
        dist_m = haversine(current_lat, current_lon, *data["gps"])
        if dist_m > _NEARBY_RADIUS_M * 1.5:
            continue

        raw = 0.0
        sem = semantic_scores.get(node, 0.0) if semantic_scores else 0.0
        raw += sem * 35.0
        if query:
            raw += _rule_based_score(G, node, dict(needs), weather)
        raw += _interest_score(node, data, active_interests or []) * 0.4
        raw += time_category_boost(node, G, time_band, weekend)
        raw += indoor_boost(G, node, weather)

        if dist_m < 50:
            raw += 12
        elif dist_m < 150:
            raw += 6

        if dest_gps:
            raw += _geo_alignment_score(
                current_lat, current_lon,
                data["gps"][0], data["gps"][1],
                dest_gps[0], dest_gps[1],
            )

        crowd = predict_crowd_level(G, node, current_time_str)
        if crowd <= 0.35:
            raw += 5
        elif crowd >= 0.85:
            raw -= 6

        if raw < 12:
            continue

        reason = _build_reason(
            G, node, sem,
            node in path_nodes,
            dist_m, crowd, destination,
        )
        _add(node, raw, reason, "ai_context", 5)

    # --- Đang đứng sát một điểm thú vị ---
    if dist_nearest < 50:
        profile = get_building_profile(G, nearest_node)
        if profile.get("function_summary"):
            _add(
                nearest_node, 28,
                build_function_reason(
                    G, nearest_node, query,
                    f"Bạn đang rất gần — có muốn ghé",
                ),
                "nearby_context", 2,
            )

    results = sorted(
        candidates.values(),
        key=lambda x: (x["priority"], -x["raw_score"]),
    )

    seen = set()
    final: List[dict] = []
    for item in results:
        if item["node"] in seen:
            continue
        seen.add(item["node"])
        entry = enrich_suggestion(G, {
            "node": item["node"],
            "score": item["score"],
            "reason": item["reason"],
            "priority": item["priority"],
            "distance_m": item["distance_m"],
            "crowd_level": item["crowd_level"],
            "on_route": item["on_route"],
            "source": item["source"],
            "gps": item["gps"],
            "score_breakdown": item.get("score_breakdown", {
                "personalization": 50.0,
                "proximity": 50.0,
                "context": 50.0,
                "crowd": 50.0
            }),
        }, query)

        # Thêm thông tin cá nhân hóa bổ sung
        visit_count = visited_history.get(item["node"], 0)
        entry["visit_count"] = visit_count
        entry["is_favorite"] = visit_count >= 3
        entry["personal_tags"] = item.get("personal_tags", [])
        entry["ncf_score"] = item.get("ncf_score", 0.0)

        final.append(entry)
        if len(final) >= limit:
            break

    return final


def get_proactive_recommendations(
    G: nx.Graph,
    current_lat: float,
    current_lon: float,
    current_time_str: str,
    destination: Optional[str] = None,
    query: Optional[str] = None,
    weather: str = "normal",
    user_interests: Optional[List[str]] = None,
    limit: int = 5,
    user_profile: Optional[dict] = None,
) -> list:
    """
    Gợi ý chủ động — tương thích API cũ, mở rộng theo đích đến & AI.
    """
    smart = get_smart_recommendations(
        G, current_lat, current_lon,
        destination=destination,
        query=query,
        weather=weather,
        current_time_str=current_time_str,
        user_interests=user_interests,
        limit=limit,
        user_profile=user_profile,
    )
    return [
        {
            "node": s["node"],
            "reason": s["reason"],
            "priority": s["priority"],
            "score": s["score"],
            "distance_m": s["distance_m"],
            "on_route": s["on_route"],
            "tagline": s.get("tagline", ""),
            "function_summary": s.get("function_summary", ""),
            "services": s.get("services", []),
            "matched_services": s.get("matched_services", []),
            "visit_count": s.get("visit_count", 0),
            "is_favorite": s.get("is_favorite", False),
            "personal_tags": s.get("personal_tags", []),
            "score_breakdown": s.get("score_breakdown", {
                "personalization": 50.0,
                "proximity": 50.0,
                "context": 50.0,
                "crowd": 50.0
            }),
        }
        for s in smart
    ]


def recommend_by_building_function(
    G: nx.Graph,
    query: str,
    current_time: str = None,
    limit: int = 5,
) -> List[dict]:
    """
    Gợi ý tòa theo chức năng cụ thể (VD: 'muốn ăn trưa', 'cần lab máy tính').
    """
    q = normalize_text(query)
    if not q:
        return []

    results: List[dict] = []
    for node in G.nodes():
        if not is_node_open(G, node, current_time):
            continue
        profile = get_building_profile(G, node)
        services = profile.get("services", [])
        matched = match_services_to_query(query, services)
        if not matched:
            continue
        sem_ai = CampusSemanticAI(G)
        sem = sem_ai.score_query(query).get(node, 0.0)
        raw = len(matched) * 25 + sem * 30
        results.append(enrich_suggestion(G, {
            "node": node,
            "score": _normalize_score(raw),
            "reason": build_function_reason(
                G, node, query,
                f"Phù hợp vì có {', '.join(s['name'] for s in matched)}",
            ),
            "matched_services": matched,
            "method": "Building Function Match",
        }, query))

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]


# =====================================================================
# SEMANTIC MAP LINKING — mô tả user → tọa độ đồ thị
# =====================================================================
def semantic_map_linking(
    G: nx.Graph,
    query: str,
    current_lat: Optional[float] = None,
    current_lon: Optional[float] = None,
) -> Optional[dict]:
    """
    Ánh xạ mô tả tự nhiên (VD: 'Gần thư viện') sang node + GPS trên đồ thị.
    """
    q = normalize_text(query)
    if not q:
        return None

    semantic_ai = CampusSemanticAI(G)
    scores = semantic_ai.score_query(query)

    # Ưu tiên từ khóa địa danh trong câu
    for node, data in G.nodes(data=True):
        aliases = [normalize_text(node)] + [normalize_text(a) for a in data.get("aliases", [])]
        for alias in aliases:
            if alias and len(alias) >= 3 and alias in q:
                scores[node] = scores.get(node, 0) + 0.5

    if "gan" in q.split() or "gần" in query.lower():
        anchor_terms = ["thu vien", "can tin", "nha xe", "gym", "atm", "cong"]
        for term in anchor_terms:
            if term in q:
                for node, data in G.nodes(data=True):
                    blob = normalize_text(node + " " + " ".join(data.get("aliases", [])))
                    if term in blob:
                        scores[node] = scores.get(node, 0) + 0.4

    if not scores:
        return None

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    best_node, best_score = ranked[0]
    if best_score < 0.05:
        return None

    gps = G.nodes[best_node]["gps"]
    result = {
        "node": best_node,
        "gps": {"lat": gps[0], "lon": gps[1]},
        "confidence": round(min(1.0, best_score), 3),
        "matched_query": query,
        "alternatives": [
            {"node": n, "score": round(s, 3)}
            for n, s in ranked[1:4]
            if s > 0.05
        ],
    }

    if current_lat is not None and current_lon is not None:
        result["distance_from_user_m"] = round(
            haversine(current_lat, current_lon, gps[0], gps[1]), 1
        )
    return result


# =====================================================================
# CONTEXT RECOMMENDER — thời gian + vị trí (+ GNN embedding)
# =====================================================================
def context_recommender(
    G: nx.Graph,
    current_lat: float,
    current_lon: float,
    current_time_str: str,
    destination: Optional[str] = None,
    query: Optional[str] = None,
    weather: str = "normal",
    user_interests: Optional[List[str]] = None,
    limit: int = 6,
    user_profile: Optional[dict] = None,
) -> List[dict]:
    """Gợi ý địa điểm dựa trên thời gian, vị trí GPS và ngữ cảnh (wrapper AI đầy đủ)."""
    try:
        from engine.gnn_engine import gnn_node_embedding
        embeddings = gnn_node_embedding(G)
    except Exception:
        embeddings = {}

    base = get_smart_recommendations(
        G, current_lat, current_lon,
        destination=destination,
        query=query,
        weather=weather,
        current_time_str=current_time_str,
        user_interests=user_interests,
        limit=limit,
        user_profile=user_profile,
    )

    for item in base:
        emb = embeddings.get(item["node"], [])
        item["gnn_embedding_preview"] = emb[:4] if emb else []
        item["crowd_pct"] = round(predict_crowd_level(G, item["node"], current_time_str) * 100)
    return base


# =====================================================================
# TASK 3: CROWD PREDICTION + NCF RECOMMENDER — PYTORCH MODELS
# =====================================================================
import os
import json
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# CrowdPredictor — khớp với kiến trúc mới trong train_models.py
# ---------------------------------------------------------------------------
class CrowdPredictor(nn.Module):
    """MLP sâu với BatchNorm — khớp với train_models.CrowdPredictor."""

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


# ---------------------------------------------------------------------------
# NCFRecommender — Neural Collaborative Filtering
# ---------------------------------------------------------------------------
class NCFRecommender(nn.Module):
    """
    GMF + MLP Neural Collaborative Filtering.
    Học từ lịch sử tương tác user × item để cá nhân hóa đề xuất.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 32,
        mlp_layers: list = None,
        context_dim: int = 2,
    ):
        super().__init__()
        mlp_layers = mlp_layers or [128, 64, 32]

        self.user_embed_gmf = nn.Embedding(num_users, embed_dim)
        self.item_embed_gmf = nn.Embedding(num_items, embed_dim)
        self.user_embed_mlp = nn.Embedding(num_users, embed_dim)
        self.item_embed_mlp = nn.Embedding(num_items, embed_dim)

        mlp_input = embed_dim * 2 + context_dim
        layers = []
        for out_dim in mlp_layers:
            layers += [nn.Linear(mlp_input, out_dim), nn.ReLU(), nn.Dropout(0.2)]
            mlp_input = out_dim
        self.mlp = nn.Sequential(*layers)

        self.output = nn.Linear(embed_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, item_ids, context):
        u_gmf = self.user_embed_gmf(user_ids)
        i_gmf = self.item_embed_gmf(item_ids)
        gmf_out = u_gmf * i_gmf

        u_mlp = self.user_embed_mlp(user_ids)
        i_mlp = self.item_embed_mlp(item_ids)
        mlp_in = torch.cat([u_mlp, i_mlp, context], dim=1)
        mlp_out = self.mlp(mlp_in)

        combined = torch.cat([gmf_out, mlp_out], dim=1)
        return self.sigmoid(self.output(combined)).squeeze(1)

    def score_item(self, user_idx: int, item_idx: int, hour: float, dow: int) -> float:
        """Tính điểm NCF cho một cặp (user, item) tại ngữ cảnh giờ/ngày."""
        self.eval()
        with torch.no_grad():
            u = torch.tensor([user_idx], dtype=torch.long)
            i = torch.tensor([item_idx], dtype=torch.long)
            ctx = torch.tensor([[hour / 24.0, dow / 6.0]], dtype=torch.float32)
            return float(self.forward(u, i, ctx).item())

    def top_k_for_user(
        self,
        user_idx: int,
        item_indices: List[int],
        hour: float,
        dow: int,
        k: int = 8,
    ) -> List[Tuple[int, float]]:
        """Trả về top-k item indices và điểm NCF cho một user."""
        self.eval()
        with torch.no_grad():
            u = torch.tensor([user_idx] * len(item_indices), dtype=torch.long)
            i = torch.tensor(item_indices, dtype=torch.long)
            ctx = torch.tensor(
                [[hour / 24.0, dow / 6.0]] * len(item_indices),
                dtype=torch.float32,
            )
            scores = self.forward(u, i, ctx).tolist()
        ranked = sorted(zip(item_indices, scores), key=lambda x: -x[1])
        return ranked[:k]


# ---------------------------------------------------------------------------
# Singleton: load models khi import
# ---------------------------------------------------------------------------
_CROWD_MODEL = None
_CROWD_METADATA = None
_NCF_MODEL = None
_NCF_METADATA = None


def load_crowd_model():
    global _CROWD_MODEL, _CROWD_METADATA
    engine_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(engine_dir, "crowd_model.pth")
    metadata_path = os.path.join(engine_dir, "model_metadata.json")

    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        return
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            _CROWD_METADATA = json.load(f)["crowd"]
        _CROWD_MODEL = CrowdPredictor(_CROWD_METADATA["input_dim"])
        _CROWD_MODEL.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        _CROWD_MODEL.eval()
        print("✅ [Recommender] Crowd Predictor nạp thành công.")
    except Exception as e:
        print(f"⚠️ [Recommender] Crowd model lỗi: {e}")
        _CROWD_MODEL = None


def load_ncf_model():
    global _NCF_MODEL, _NCF_METADATA
    engine_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(engine_dir, "ncf_model.pth")
    metadata_path = os.path.join(engine_dir, "model_metadata.json")

    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        return
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "ncf" not in meta:
            return
        _NCF_METADATA = meta["ncf"]
        _NCF_MODEL = NCFRecommender(
            num_users=_NCF_METADATA["num_users"],
            num_items=_NCF_METADATA["num_items"],
            embed_dim=_NCF_METADATA.get("embed_dim", 32),
            mlp_layers=_NCF_METADATA.get("mlp_layers", [128, 64, 32]),
            context_dim=_NCF_METADATA.get("context_dim", 2),
        )
        _NCF_MODEL.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        _NCF_MODEL.eval()
        print("✅ [Recommender] NCF Recommender nạp thành công.")
    except Exception as e:
        print(f"⚠️ [Recommender] NCF model lỗi: {e}")
        _NCF_MODEL = None


load_crowd_model()
load_ncf_model()


# ---------------------------------------------------------------------------
# predict_crowd_level
# ---------------------------------------------------------------------------
def predict_crowd_level(G: nx.Graph, node_id: str, current_time_str: str) -> float:
    """Dự báo mức độ đông đúc (0–1) cho một node tại thời điểm cho trước."""
    # 1. Crowdsourcing thời gian thực ưu tiên cao nhất
    live = get_live_crowd(node_id)
    if live is not None:
        return live

    # 2. Mô hình PyTorch CrowdPredictor (input_dim mới = 29)
    if _CROWD_MODEL is not None and _CROWD_METADATA is not None:
        try:
            nodes = _CROWD_METADATA["nodes"]
            weather_types = _CROWD_METADATA["weather_types"]
            input_dim = _CROWD_METADATA["input_dim"]

            # Snap node về danh sách đã train
            lookup_node = node_id
            if lookup_node not in nodes:
                base = node_id.split("_")[0] if "_" in node_id else node_id
                matches = [n for n in nodes if n.startswith(base)]
                lookup_node = matches[0] if matches else nodes[0]

            node_vec = np.zeros(len(nodes), dtype=np.float32)
            node_vec[nodes.index(lookup_node)] = 1.0

            # Thời tiết mặc định = normal
            weather_vec = np.zeros(len(weather_types), dtype=np.float32)
            weather_vec[0] = 1.0

            curr_t = parse_time(current_time_str)
            hour_val = (curr_t.hour + curr_t.minute / 60.0) if curr_t else 12.0

            from engine.context_features import is_weekend
            from datetime import datetime
            now = datetime.now()
            dow_val = now.weekday()       # 0=Mon … 6=Sun
            month_val = now.month         # 1–12
            is_exam = 1.0 if month_val in (1, 5, 6, 12) else 0.0

            # Ghép đặc trưng: node(22) + hour + dow + month + is_exam + weather(3) = 29
            features = np.concatenate([
                node_vec,
                [hour_val / 24.0, dow_val / 6.0, (month_val - 1) / 11.0, is_exam],
                weather_vec,
            ])

            # Đảm bảo đúng input_dim (tương thích ngược nếu model cũ dim khác)
            if len(features) != input_dim:
                if len(features) > input_dim:
                    features = features[:input_dim]
                else:
                    features = np.pad(features, (0, input_dim - len(features)))

            tensor = torch.tensor(features[np.newaxis, :], dtype=torch.float32)
            with torch.no_grad():
                pred = _CROWD_MODEL(tensor).item()
            return round(pred, 2)
        except Exception as e:
            print(f"⚠️ [Crowd Model Error] {e}")

    # 3. Fallback bằng luật
    curr_t = parse_time(current_time_str)
    if not curr_t:
        return 0.0

    node_data = G.nodes.get(node_id, {})
    aliases = " ".join(node_data.get("aliases", [])).lower()
    base_crowd = 0.2

    if "can tin" in aliases or "an" in aliases:
        if time(6, 0) <= curr_t <= time(9, 0):
            return 0.55
        if time(11, 30) <= curr_t <= time(13, 0):
            return 0.95
        return 0.4

    if "thu vien" in aliases or "tu hoc" in aliases:
        if time(8, 0) <= curr_t <= time(11, 0) or time(14, 0) <= curr_t <= time(16, 30):
            return 0.8
        return 0.3

    if "the thao" in aliases or "gym" in aliases:
        if time(16, 30) <= curr_t <= time(18, 30):
            return 0.85
        from engine.context_features import is_weekend
        if is_weekend() and time(8, 0) <= curr_t <= time(11, 0):
            return 0.7
        return 0.2

    if "su kien" in aliases or node_id == "Tòa G":
        from engine.context_features import is_weekend
        if is_weekend():
            return 0.65
        return 0.35

    return base_crowd


# ---------------------------------------------------------------------------
# submit_crowd_report
# ---------------------------------------------------------------------------
def submit_crowd_report(node_id: str, level: float) -> dict:
    """Crowdsourcing ẩn danh — cập nhật độ đông thời gian thực."""
    report_live_crowd(node_id, level)
    return {"node": node_id, "reported_level": round(level, 2), "status": "accepted"}


# ---------------------------------------------------------------------------
# crowd_prediction
# ---------------------------------------------------------------------------
def crowd_prediction(G: nx.Graph, node_id: str, current_time_str: str) -> dict:
    """Dự báo độ đông đúc — trả về level + nhãn."""
    pred = predict_crowd_level(G, node_id, current_time_str)
    level = effective_crowd_level(G, node_id, pred)
    if level >= 0.85:
        label = "rat dong"
    elif level >= 0.6:
        label = "dong vua"
    elif level >= 0.35:
        label = "binh thuong"
    else:
        label = "vang"
    return {
        "node": node_id,
        "crowd_level": round(level, 2),
        "crowd_predicted": round(pred, 2),
        "crowd_pct": round(level * 100),
        "label": label,
        "live_report": get_live_crowd(node_id) is not None,
    }


# ---------------------------------------------------------------------------
# ncf_recommend
# ---------------------------------------------------------------------------
def ncf_recommend(
    G: nx.Graph,
    user_profile: dict,
    current_time_str: Optional[str] = None,
    limit: int = 8,
) -> List[dict]:
    """
    Đề xuất cá nhân hóa bằng Neural Collaborative Filtering.

    Tìm user profile gần nhất trong tập đã train (dựa trên role + interests),
    sau đó dùng NCF model để score tất cả items và trả về top-k.

    Trả về list dict với: node, ncf_score, score, reason, gps, source, method
    """
    if _NCF_MODEL is None or _NCF_METADATA is None:
        return []

    try:
        from datetime import datetime
        curr_t = parse_time(current_time_str) if current_time_str else None
        hour_val = (curr_t.hour + curr_t.minute / 60.0) if curr_t else datetime.now().hour
        dow_val = datetime.now().weekday()

        user_to_idx = _NCF_METADATA["user_to_idx"]
        item_to_idx = _NCF_METADATA["item_to_idx"]
        profiles = _NCF_METADATA.get("user_profiles", [])

        # --- Tìm user profile hoặc sử dụng current_user ---
        role = user_profile.get("role", "student")
        if "current_user" in user_to_idx:
            u_idx = user_to_idx["current_user"]
        else:
            interests = set(i.lower().replace(" ", "_") for i in user_profile.get("interests", []))

            best_uid = None
            best_score = -1
            for p in profiles:
                if p["role"] != role:
                    continue
                p_interests = set(i.lower().replace(" ", "_") for i in p["interests"])
                overlap = len(interests & p_interests)
                if overlap > best_score:
                    best_score = overlap
                    best_uid = p["user_id"]

            # Fallback: lấy user đầu tiên cùng role
            if best_uid is None:
                for p in profiles:
                    if p["role"] == role:
                        best_uid = p["user_id"]
                        break
            if best_uid is None:
                best_uid = profiles[0]["user_id"]

            u_idx = user_to_idx[best_uid]

        # --- Score tất cả items ---
        all_item_indices = list(range(len(_NCF_METADATA["item_to_idx"])))
        ranked = _NCF_MODEL.top_k_for_user(u_idx, all_item_indices, hour_val, dow_val, k=limit * 2)

        # Ánh xạ ngược item_idx → node name
        idx_to_item = {v: k for k, v in item_to_idx.items()}

        results = []
        for i_idx, score in ranked:
            node = idx_to_item.get(i_idx)
            if not node or node not in G.nodes:
                continue
            if not is_node_open(G, node, current_time_str):
                continue
            gps = G.nodes[node]["gps"]
            results.append({
                "node": node,
                "ncf_score": round(score, 4),
                "score": round(score * 100, 1),
                "reason": f"NCF: phù hợp hồ sơ {role} (điểm {score:.2f})",
                "gps": gps,
                "source": "ncf",
                "method": "Neural Collaborative Filtering",
            })
            if len(results) >= limit:
                break

        return results
    except Exception as e:
        print(f"⚠️ [NCF Error] {e}")
        return []


# =====================================================================
# TASK 4: COLLABORATIVE FILTERING (CLB / Lab / sở thích)
# =====================================================================
_CLUB_RULES = [
    {
        "tags": ["c++", "codeforces", "sql", "code", "thuat toan", "lap trinh", "cntt"],
        "match": ["may tinh", "lab", "thuc hanh", "nha c"],
        "label": "Lab CNTT / Phòng máy",
        "score": 50,
    },
    {
        "tags": ["robot", "iot", "arduino", "dien tu"],
        "match": ["phong thi nghiem", "nha a", "lab"],
        "label": "CLB Robot / Thực nghiệm",
        "score": 45,
    },
    {
        "tags": ["genshin", "tft", "game", "esport", "lol"],
        "match": ["phong nghi", "canteen"],
        "label": "CLB Game / Esport",
        "score": 35,
    },
    {
        "tags": ["football", "bayern", "chelsea", "the thao", "bong da", "cau long"],
        "match": ["the duc", "the thao", "gym", "clb"],
        "label": "CLB Thể thao",
        "score": 45,
    },
    {
        "tags": ["english", "ielts", "toeic", "ngoai ngu"],
        "match": ["thu vien", "tu hoc"],
        "label": "CLB Ngoại ngữ / Self-study",
        "score": 30,
    },
    {
        "tags": ["am nhac", "music", "band", "guitar"],
        "match": ["nha g", "phong nghi"],
        "label": "CLB Âm nhạc",
        "score": 28,
    },
]


def collaborative_filtering(
    G: nx.Graph,
    user_interests: List[str],
    current_lat: Optional[float] = None,
    current_lon: Optional[float] = None,
) -> List[dict]:
    """Gợi ý CLB / Lab / địa điểm theo sở thích sinh viên."""
    suggestions = []
    interests_str = " ".join(user_interests).lower()

    for node, data in G.nodes(data=True):
        aliases = " ".join(data.get("aliases", [])).lower()
        total = 0
        matched_labels: List[str] = []

        for rule in _CLUB_RULES:
            if any(t in interests_str for t in rule["tags"]):
                if any(m in aliases for m in rule["match"]):
                    total += rule["score"]
                    matched_labels.append(rule["label"])

        if total > 0:
            profile = get_building_profile(G, node)
            entry = {
                "node": node,
                "match_score": total,
                "categories": matched_labels,
                "type": data.get("type", "building"),
                "gps": data.get("gps"),
                "tagline": profile.get("tagline", ""),
                "function_summary": profile.get("function_summary", ""),
                "services": profile.get("services", []),
            }
            if current_lat is not None and current_lon is not None:
                entry["distance_m"] = round(
                    haversine(current_lat, current_lon, *data["gps"]), 1
                )
            suggestions.append(entry)

    suggestions.sort(key=lambda x: x["match_score"], reverse=True)
    return suggestions[:8]


# =====================================================================
# TEST LOCAL
# =====================================================================
if __name__ == "__main__":
    from engine.graph_builder_v2 import build_flat_campus_graph

    campus_graph = build_flat_campus_graph()
    now = "12:00"

    print("--- TEST AI SEMANTIC (TOP 3) ---")
    for r in recommend_locations(campus_graph, "tim cho mat me yen tinh hoc bai", now, limit=3):
        print(f"  {r['node']}: {r['score']}/100 — {r['reason']}")

    print("\n--- TEST SMART (GPS + ĐÍCH) ---")
    lat, lon = 10.8710, 106.8020
    dest = "Tòa D"
    for s in get_smart_recommendations(
        campus_graph, lat, lon,
        destination=dest,
        query="an trua",
        current_time_str=now,
        limit=5,
    ):
        print(f"  [{s['source']}] {s['node']} ({s['score']}) — {s['reason']}")

    print("\n--- TEST CROWD ---")
    print(f"Căn tin 12:00: {predict_crowd_level(campus_graph, 'Tòa D', '12:00') * 100:.0f}%")

    print("\n--- TEST SUBMIT CROWD REPORT ---")
    print(submit_crowd_report("Tòa D", 0.9))
