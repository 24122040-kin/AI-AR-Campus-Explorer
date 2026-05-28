# engine/building_catalog.py
"""
Danh mục chức năng / dịch vụ từng tòa — dùng cho gợi ý và hiển thị AR.
"""
from typing import Dict, List, Optional

import networkx as nx

from engine.nlp_processor import normalize_text

# Mỗi service: id, tên hiển thị, icon, category, từ khóa tìm kiếm
_BUILDING_PROFILES: Dict[str, dict] = {
    "Tòa A": {
        "tagline": "Khu thực nghiệm & phòng lab kỹ thuật",
        "services": [
            {"id": "lab", "name": "Phòng thí nghiệm", "icon": "🔬", "category": "hoc_tap",
             "keywords": ["thuc nghiem", "lab", "kỹ thuật"]},
            {"id": "group_study", "name": "Học nhóm / ôn tập", "icon": "👥", "category": "hoc_tap",
             "keywords": ["hoc nhom", "on tap"]},
        ],
    },
    "Tòa B": {
        "tagline": "Phòng tự học yên tĩnh",
        "services": [
            {"id": "self_study", "name": "Phòng tự học", "icon": "📖", "category": "hoc_tap",
             "keywords": ["tu hoc", "yen tinh", "tap trung"]},
            {"id": "group_quiet", "name": "Học nhóm nhỏ", "icon": "🤫", "category": "hoc_tap",
             "keywords": ["hoc nhom"]},
        ],
    },
    "Tòa C": {
        "tagline": "Lab máy tính & thực hành CNTT",
        "services": [
            {"id": "computer_lab", "name": "Phòng máy / Lab CNTT", "icon": "💻", "category": "cntt",
             "keywords": ["may tinh", "lab", "lap trinh", "code"]},
            {"id": "practice", "name": "Thực hành môn học", "icon": "⌨️", "category": "hoc_tap",
             "keywords": ["thuc hanh"]},
        ],
    },
    "Tòa D": {
        "tagline": "Thư viện, căn tin & quầy giao trình",
        "services": [
            {"id": "library", "name": "Thư viện / đọc sách", "icon": "📚", "category": "hoc_tap",
             "keywords": ["thu vien", "doc sach", "muon sach"]},
            {"id": "canteen", "name": "Căn tin / ăn uống", "icon": "🍽️", "category": "an_uong",
             "keywords": ["can tin", "an trua", "an", "doi bung", "com"]},
            {"id": "bookstore", "name": "Quầy giao trình / sách", "icon": "📕", "category": "hoc_tap",
             "keywords": ["giao trinh", "mua sach"]},
        ],
    },
    "Tòa E": {
        "tagline": "Lý thuyết & nghỉ trưa",
        "services": [
            {"id": "lecture", "name": "Phòng học lý thuyết", "icon": "🎓", "category": "hoc_tap",
             "keywords": ["ly thuyet", "bai giang"]},
            {"id": "nap", "name": "Khu nghỉ trưa", "icon": "😴", "category": "nghi_ngoi",
             "keywords": ["nghi trua", "ngu trua"]},
        ],
    },
    "Tòa F": {
        "tagline": "Phòng nghỉ & chỗ ngả lưng",
        "services": [
            {"id": "rest", "name": "Phòng nghỉ sinh viên", "icon": "🛋️", "category": "nghi_ngoi",
             "keywords": ["nghi", "ngu", "met", "buon ngu"]},
        ],
    },
    "Tòa G": {
        "tagline": "Khu sự kiện / hoạt động ngoài trời",
        "services": [
            {"id": "event", "name": "Sân / khu sự kiện", "icon": "🎪", "category": "giai_tri",
             "keywords": ["su kien", "hoat dong"]},
        ],
    },
    "Nhà thể dục": {
        "tagline": "Gym, thể thao & CLB",
        "services": [
            {"id": "gym", "name": "Tập gym / fitness", "icon": "🏋️", "category": "the_thao",
             "keywords": ["gym", "tap", "the luc"]},
            {"id": "sports", "name": "Cầu lông, bóng bàn, CLB thể thao", "icon": "🏸", "category": "the_thao",
             "keywords": ["cau long", "bong ban", "the thao", "clb"]},
        ],
    },
    "Nhà xe": {
        "tagline": "Bãi giữ xe sinh viên",
        "services": [
            {"id": "parking", "name": "Gửi / lấy xe máy", "icon": "🛵", "category": "tien_ich",
             "keywords": ["gui xe", "lay xe", "xe may", "parking"]},
            {"id": "exit", "name": "Ra về cuối ngày", "icon": "🚪", "category": "tien_ich",
             "keywords": ["ra ve", "ve nha"]},
        ],
    },
    "ATM": {
        "tagline": "Rút tiền & giao dịch nhanh",
        "services": [
            {"id": "atm", "name": "Cây ATM", "icon": "🏧", "category": "tien_ich",
             "keywords": ["atm", "rut tien", "tien mat"]},
        ],
    },
    "Nhà điều hành": {
        "tagline": "Hành chính & giao vụ (khu hạn chế)",
        "services": [
            {"id": "admin", "name": "Phòng ban / giao vụ", "icon": "🏛️", "category": "hanh_chinh",
             "keywords": ["giao vu", "giay to", "hanh chinh"]},
            {"id": "tuition", "name": "Đóng học phí", "icon": "💳", "category": "hanh_chinh",
             "keywords": ["hoc phi", "dong tien"]},
        ],
    },
    "Cổng trường": {
        "tagline": "Lối vào chính campus",
        "services": [
            {"id": "entrance", "name": "Check-in / vào campus", "icon": "🚧", "category": "tien_ich",
             "keywords": ["cong", "vao truong"]},
        ],
    },
}

_CATEGORY_LABELS = {
    "hoc_tap": "Học tập",
    "an_uong": "Ăn uống",
    "nghi_ngoi": "Nghỉ ngơi",
    "the_thao": "Thể thao",
    "cntt": "CNTT / Lab",
    "tien_ich": "Tiện ích",
    "hanh_chinh": "Hành chính",
    "giai_tri": "Giải trí",
}


def get_building_profile(G: nx.Graph, node_id: str) -> dict:
    """Hồ sơ đầy đủ chức năng của một tòa."""
    if node_id not in G.nodes:
        return {}

    data = G.nodes[node_id]
    
    # Hỗ trợ tìm kiếm profile cho các phòng/tầng (fallback về tòa nhà gốc)
    base_node = node_id.split("_")[0] if "_" in node_id else node_id
    catalog = _BUILDING_PROFILES.get(node_id) or _BUILDING_PROFILES.get(base_node) or {}
    
    # Tinh chỉnh dịch vụ cho phòng cụ thể
    services = list(data.get("services") or catalog.get("services", []))
    if "_" in node_id:
        room_name = node_id.split("_")[-1]
        if "Căn tin" in room_name:
            services = [s for s in services if s["id"] == "canteen"]
        elif "Thư viện" in room_name:
            services = [s for s in services if s["id"] == "library"]
        elif "Quầy giáo trình" in room_name:
            services = [s for s in services if s["id"] == "bookstore"]
        elif "Tự học" in room_name:
            services = [s for s in services if s["id"] in ("self_study", "group_study", "group_quiet")]
            
    tagline = data.get("tagline") or catalog.get("tagline", "")
    if "_" in node_id:
        tagline = f"{node_id.split('_')[-1]} — thuộc {base_node}"

    features = data.get("features", {})
    amenity_tags = []
    if features.get("has_ac"):
        amenity_tags.append("Điều hòa")
    if features.get("has_tables"):
        amenity_tags.append("Bàn ghế học tập")
    if features.get("noise_level", 1) <= 0.3:
        amenity_tags.append("Yên tĩnh")
    elif features.get("noise_level", 0) >= 0.7:
        amenity_tags.append("Sôi động")

    service_names = [s["name"] for s in services]
    function_summary = " · ".join(service_names) if service_names else tagline

    return {
        "node": node_id,
        "tagline": tagline,
        "function_summary": function_summary,
        "services": services,
        "amenities": amenity_tags,
        "type": data.get("type", "building"),
        "restricted": bool(data.get("restricted", False)),
        "open_time": data.get("open_time"),
        "close_time": data.get("close_time"),
        "gps": data.get("gps"),
    }


def match_services_to_query(query: str, services: List[dict]) -> List[dict]:
    """Lọc dịch vụ khớp câu hỏi người dùng."""
    q = normalize_text(query)
    if not q:
        return []
    matched = []
    for svc in services:
        keys = [normalize_text(svc.get("name", ""))] + [normalize_text(k) for k in svc.get("keywords", [])]
        if any(k and k in q for k in keys) or any(k and len(k) >= 4 and k in q for k in keys):
            matched.append(svc)
    return matched


def build_function_reason(
    G: nx.Graph,
    node_id: str,
    query: Optional[str] = None,
    context_hint: str = "",
) -> str:
    """Câu gợi ý có nêu rõ chức năng tòa."""
    profile = get_building_profile(G, node_id)
    if not profile:
        return context_hint or f"Ghé {node_id}."

    summary = profile["function_summary"]
    tagline = profile["tagline"]

    if query:
        matched = match_services_to_query(query, profile["services"])
        if matched:
            names = ", ".join(s["name"] for s in matched[:3])
            base = f"{node_id} có {names}"
        else:
            base = f"{node_id}: {summary}"
    else:
        base = f"{node_id} — {tagline}" if tagline else f"{node_id}: {summary}"

    if context_hint:
        return f"{context_hint} ({base})."
    return f"{base}."


def enrich_suggestion(G: nx.Graph, item: dict, query: Optional[str] = None) -> dict:
    """Bổ sung chức năng tòa vào object gợi ý."""
    node = item.get("node")
    if not node:
        return item

    profile = get_building_profile(G, node)
    item["tagline"] = profile.get("tagline", "")
    item["function_summary"] = profile.get("function_summary", "")
    item["services"] = profile.get("services", [])
    item["amenities"] = profile.get("amenities", [])

    if query and profile.get("services"):
        item["matched_services"] = match_services_to_query(query, profile["services"])
    else:
        item["matched_services"] = []

    if (
        profile.get("function_summary")
        and item.get("reason")
        and profile["function_summary"] not in item["reason"]
        and "Bạn có thể" not in item["reason"]
    ):
        item["reason"] = (
            f"{item['reason'].rstrip('.')} — "
            f"Bạn có thể: {profile['function_summary']}."
        )
    return item


def list_all_building_guides(G: nx.Graph) -> List[dict]:
    """Danh sách chức năng toàn campus — cho màn hình tra cứu."""
    guides = []
    for node in sorted(G.nodes()):
        p = get_building_profile(G, node)
        if p:
            guides.append(p)
    return guides
