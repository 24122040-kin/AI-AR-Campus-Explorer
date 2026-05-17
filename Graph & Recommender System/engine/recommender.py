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
    for node, data in G.nodes(data=True):
        if not is_node_open(G, node, current_time):
            continue

        sem = semantic_scores.get(node, 0.0) * 40.0
        rule = _rule_based_score(G, node, dict(needs), weather)
        raw = sem + rule
        if raw <= 0 and sem < 8:
            continue

        ranked.append({
            "node": node,
            "score": _normalize_score(max(raw, sem)),
            "raw_score": round(raw, 2),
            "semantic_score": round(sem, 2),
            "method": "AI Semantic + Intent",
            "reason": _build_reason(G, node, sem / 40.0, False, 9999, 0.0, None),
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
) -> List[dict]:
    """
    Gợi ý thông minh dựa trên GPS, điểm đến, câu hỏi tự nhiên và sở thích.
    Kết hợp: TF-IDF, luật thời gian, độ đông, lộ trình A*, hướng địa lý.
    """
    nearest_node, dist_nearest = _nearest_node(G, current_lat, current_lon)
    curr_t = parse_time(current_time_str) if current_time_str else None
    if not curr_t:
        return []

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

    candidates: Dict[str, dict] = {}

    def _add(node: str, raw: float, reason: str, source: str, priority: int = 5) -> None:
        if node == nearest_node and dist_nearest < 30 and source != "nearby_context":
            return
        if not is_node_open(G, node, current_time_str):
            return
        entry = candidates.get(node)
        if entry and entry["raw_score"] >= raw:
            return
        dist_m = haversine(current_lat, current_lon, *G.nodes[node]["gps"])
        crowd = predict_crowd_level(G, node, current_time_str)
        candidates[node] = {
            "node": node,
            "raw_score": raw,
            "score": _normalize_score(raw),
            "reason": reason,
            "priority": priority,
            "distance_m": round(dist_m, 1),
            "crowd_level": round(crowd, 2),
            "on_route": node in path_nodes and node != destination,
            "source": source,
            "gps": G.nodes[node]["gps"],
        }

    # --- Gợi ý theo thời gian (giữ logic cũ, mở rộng) ---
    def in_time_range(start_str: str, end_str: str) -> bool:
        return parse_time(start_str) <= curr_t <= parse_time(end_str)

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
        raw += _interest_score(node, data, user_interests or []) * 0.4

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
        }, query)
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
    )

    for item in base:
        emb = embeddings.get(item["node"], [])
        item["gnn_embedding_preview"] = emb[:4] if emb else []
        item["crowd_pct"] = round(predict_crowd_level(G, item["node"], current_time_str) * 100)
    return base


# =====================================================================
# TASK 3: CROWD PREDICTION
# =====================================================================
def predict_crowd_level(G: nx.Graph, node_id: str, current_time_str: str) -> float:
    curr_t = parse_time(current_time_str)
    if not curr_t:
        return 0.0

    node_data = G.nodes.get(node_id, {})
    aliases = " ".join(node_data.get("aliases", [])).lower()
    base_crowd = 0.2

    if "can tin" in aliases or "an" in aliases:
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
        return 0.2

    return base_crowd


def crowd_prediction(G: nx.Graph, node_id: str, current_time_str: str) -> dict:
    """Dự báo độ đông đúc — trả về level + nhãn."""
    level = predict_crowd_level(G, node_id, current_time_str)
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
        "crowd_pct": round(level * 100),
        "label": label,
    }


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
