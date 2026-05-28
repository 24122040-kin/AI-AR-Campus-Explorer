# engine/graph_builder_v2.py
import networkx as nx
import matplotlib.pyplot as plt
from engine.building_catalog import _BUILDING_PROFILES
from engine.campus_knowledge import CAMPUS_LINH_TRUNG, get_cluster_for_node
from engine.utils import haversine


def build_campus_graph() -> nx.Graph:
    """
    Xây dựng đồ thị campus (Node = phòng/địa điểm, Edge = đường đi).
    Alias chính thức theo spec dự án.
    """
    return build_flat_campus_graph()


def build_flat_campus_graph() -> nx.Graph:
    G = nx.Graph()

    # Khung giờ mặc định
    DEFAULT_OPEN  = "06:00"
    DEFAULT_CLOSE = "18:00"
    REST_OPEN     = "11:30"
    REST_CLOSE    = "12:30"

    # ---------------------------------------------------------
    # 1. ĐỊNH NGHĨA NODE ĐA TẦNG
    # ---------------------------------------------------------
    nodes_data = {
        # --- TÒA A ---
        "Tòa A_Tầng 1_Sảnh": {
            "gps": (10.876200, 106.800500), "type": "building",
            "features": {"has_ac": 0, "has_tables": 0, "noise_level": 0.5, "capacity": 200},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["toa a", "nha a", "sanh toa a", "tang 1 toa a"]
        },
        "Tòa A_Tầng 2_Phòng thí nghiệm A201": {
            "gps": (10.876240, 106.800500), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.6, "capacity": 50},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["phong thi nghiem a201", "lab a201", "thuc nghiem a201"]
        },
        "Tòa A_Tầng 3_Phòng thí nghiệm A301": {
            "gps": (10.876280, 106.800500), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.5, "capacity": 50},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["phong thi nghiem a301", "lab a301", "thuc nghiem a301"]
        },
        
        # --- TÒA B ---
        "Tòa B_Tầng 1_Sảnh": {
            "gps": (10.876500, 106.800700), "type": "building",
            "features": {"has_ac": 0, "has_tables": 0, "noise_level": 0.4, "capacity": 100},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["toa b", "nha b", "sanh toa b"]
        },
        "Tòa B_Tầng 2_Tự học B201": {
            "gps": (10.876540, 106.800700), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.2, "capacity": 80},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["phong tu hoc b201", "tu hoc b201", "hoc nhom b201", "tu hoc yen tinh"]
        },
        "Tòa B_Tầng 3_Phòng máy B301": {
            "gps": (10.876580, 106.800700), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.4, "capacity": 60},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["phong may b301", "lab b301", "thuc hanh b301", "phong lab b301"]
        },

        # --- TÒA C ---
        "Tòa C_Tầng 1_Sảnh": {
            "gps": (10.876800, 106.801000), "type": "building",
            "features": {"has_ac": 0, "has_tables": 0, "noise_level": 0.4, "capacity": 100},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["toa c", "nha c", "sanh toa c"]
        },
        "Tòa C_Tầng 2_Lab máy tính 202": {
            "gps": (10.876840, 106.801000), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.5, "capacity": 60},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["lab may tinh 202", "phong may 202", "lab cntt", "may tinh", "phong thuc hanh may tinh"]
        },
        "Tòa C_Tầng 3_Văn phòng khoa": {
            "gps": (10.876880, 106.801000), "type": "building",
            "features": {"has_ac": 1, "has_tables": 0, "noise_level": 0.2, "capacity": 40},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["van phong khoa", "vp khoa", "giao vu khoa"]
        },

        # --- TÒA D ---
        "Tòa D_Tầng 1_Căn tin": {
            "gps": (10.877200, 106.801500), "type": "building",
            "features": {"has_ac": 0, "has_tables": 1, "noise_level": 0.7, "capacity": 400},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["toa d", "nha d", "can tin", "canteen", "an trua", "com can tin", "doi bung", "an uong"]
        },
        "Tòa D_Tầng 2_Thư viện": {
            "gps": (10.877240, 106.801500), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.1, "capacity": 200},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["thu vien", "doc sach", "muon sach", "thu vien khtn", "cho doc sach"]
        },
        "Tòa D_Tầng 3_Quầy giáo trình": {
            "gps": (10.877280, 106.801500), "type": "building",
            "features": {"has_ac": 1, "has_tables": 0, "noise_level": 0.3, "capacity": 80},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["quay giao trinh", "mua sach", "tiem sach"]
        },

        # --- TÒA E ---
        "Tòa E_Tầng 1_Phòng học 101": {
            "gps": (10.877000, 106.800200), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.4, "capacity": 120},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["toa e", "nha e", "phong hoc 101", "ly thuyet"]
        },
        "Tòa E_Tầng 2_Phòng nghỉ trưa": {
            "gps": (10.877040, 106.800200), "type": "building",
            "features": {"has_ac": 1, "has_tables": 0, "noise_level": 0.2, "capacity": 50},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["phong nghi trua", "cho ngu trua", "nghi trua"]
        },

        # --- TÒA F ---
        "Tòa F_Tầng 1_Phòng nghỉ 102": {
            "gps": (10.877200, 106.799700), "type": "building",
            "features": {"has_ac": 1, "has_tables": 0, "noise_level": 0.2, "capacity": 50},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["toa f", "nha f", "phong nghi 102", "cho nga lung", "buon ngu", "met qua"]
        },
        "Tòa F_Tầng 2_Phòng tự học F201": {
            "gps": (10.877240, 106.799700), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.2, "capacity": 50},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["phong tu hoc f201", "tu hoc f201"]
        },

        # --- TÒA G & KHÁC ---
        "Tòa G": {
            "gps": (10.877500, 106.800000), "type": "building",
            "features": {"has_ac": 0, "has_tables": 0, "noise_level": 0.5, "capacity": 200},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["nha g", "san toa g"]
        },
        "Nhà thể dục": {
            "gps": (10.878200, 106.801200), "type": "building",
            "features": {"has_ac": 0, "has_tables": 0, "noise_level": 0.8, "capacity": 1000},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["nha the duc", "gym", "the thao", "clb", "tap gym", "cau long", "bong ban"]
        },
        "Nhà xe": {
            "gps": (10.875800, 106.801200), "type": "facility",
            "features": {"has_ac": 0, "has_tables": 0, "noise_level": 0.9, "capacity": 1000},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["bai giu xe", "parking", "gui xe", "lay xe", "cat xe", "xe may", "nha de xe"]
        },
        "ATM": {
            "gps": (10.875700, 106.800900), "type": "facility",
            "features": {"has_ac": 1, "has_tables": 0, "noise_level": 0.5, "capacity": 5},
            "open_time": "00:00", "close_time": "23:59",
            "aliases": ["cay atm", "rut tien", "het tien", "ngan hang", "tien mat"]
        },
        "Nhà điều hành": {
            "gps": (10.875000, 106.799800), "type": "admin",
            "features": {"has_ac": 1, "has_tables": 0, "noise_level": 0.1, "capacity": 100},
            "open_time": REST_OPEN, "close_time": REST_CLOSE,
            "restricted": True,
            "aliases": ["phong ban", "giao vu", "hanh chinh", "giay to", "dong hoc phi", "staff only"]
        },
        "Cổng trường": {
            "gps": (10.875600, 106.800800), "type": "facility",
            "features": {"has_ac": 0, "has_tables": 0, "noise_level": 0.6, "capacity": 200},
            "open_time": "00:00", "close_time": "23:59",
            "aliases": ["cong truong", "cong chinh", "cổng", "cong", "entrance", "main gate"]
        },
    }

    # Thêm Node vào đồ thị
    for name, data in nodes_data.items():
        base_building = name.split("_")[0] if "_" in name else name
        catalog = _BUILDING_PROFILES.get(base_building, {})
        indoor = data["type"] in ("building", "admin", "facility") and name not in (
            "Nhà xe", "Cổng trường", "Tòa G",
        )
        G.add_node(
            name,
            pos=(data["gps"][1], data["gps"][0]),  # X=Longitude, Y=Latitude
            gps=data["gps"],
            type=data["type"],
            features=data["features"],
            open_time=data["open_time"],
            close_time=data["close_time"],
            aliases=data["aliases"],
            restricted=data.get("restricted", data.get("type") == "admin"),
            tagline=catalog.get("tagline", ""),
            services=catalog.get("services", []),
            indoor=indoor,
            poi_cluster=get_cluster_for_node(base_building),
            campus_id=CAMPUS_LINH_TRUNG["id"],
        )

    # ---------------------------------------------------------
    # 2. ĐỊNH NGHĨA EDGE ĐA TẦNG — liên kết ngoài trời & trong nhà
    # ---------------------------------------------------------
    edges = [
        # --- CÁC HÀNH LANG TẦNG 1 (CÓ MÁI CHE) ---
        ("Tòa A_Tầng 1_Sảnh",        "Tòa B_Tầng 1_Sảnh",          {"has_roof": True,  "status": "open", "edge_type": "corridor"}),
        ("Tòa B_Tầng 1_Sảnh",        "Tòa C_Tầng 1_Sảnh",          {"has_roof": True,  "status": "open", "edge_type": "corridor"}),
        ("Tòa C_Tầng 1_Sảnh",        "Tòa D_Tầng 1_Căn tin",       {"has_roof": True,  "status": "open", "edge_type": "corridor"}),
        ("Tòa D_Tầng 1_Căn tin",     "Tòa E_Tầng 1_Phòng học 101", {"has_roof": True,  "status": "open", "edge_type": "corridor"}),
        ("Tòa E_Tầng 1_Phòng học 101", "Tòa F_Tầng 1_Phòng nghỉ 102",{"has_roof": True,  "status": "open", "edge_type": "corridor"}),
        ("Tòa F_Tầng 1_Phòng nghỉ 102","Tòa G",                    {"has_roof": True,  "status": "open", "edge_type": "corridor"}),
        
        # --- LIÊN KẾT DỌC (STAIRS & ELEVATORS) ---
        # Tòa A: Chỉ có thang bộ
        ("Tòa A_Tầng 1_Sảnh", "Tòa A_Tầng 2_Phòng thí nghiệm A201", {"has_roof": True, "status": "open", "edge_type": "stairs"}),
        ("Tòa A_Tầng 2_Phòng thí nghiệm A201", "Tòa A_Tầng 3_Phòng thí nghiệm A301", {"has_roof": True, "status": "open", "edge_type": "stairs"}),
        
        # Tòa B: Có thang bộ và thang máy
        ("Tòa B_Tầng 1_Sảnh", "Tòa B_Tầng 2_Tự học B201", {"has_roof": True, "status": "open", "edge_type": "stairs"}),
        ("Tòa B_Tầng 2_Tự học B201", "Tòa B_Tầng 3_Phòng máy B301", {"has_roof": True, "status": "open", "edge_type": "stairs"}),
        ("Tòa B_Tầng 1_Sảnh", "Tòa B_Tầng 3_Phòng máy B301", {"has_roof": True, "status": "open", "edge_type": "elevator"}),

        # Tòa C: Chỉ có thang bộ
        ("Tòa C_Tầng 1_Sảnh", "Tòa C_Tầng 2_Lab máy tính 202", {"has_roof": True, "status": "open", "edge_type": "stairs"}),
        ("Tòa C_Tầng 2_Lab máy tính 202", "Tòa C_Tầng 3_Văn phòng khoa", {"has_roof": True, "status": "open", "edge_type": "stairs"}),

        # Tòa D: Có thang bộ và thang máy
        ("Tòa D_Tầng 1_Căn tin", "Tòa D_Tầng 2_Thư viện", {"has_roof": True, "status": "open", "edge_type": "stairs"}),
        ("Tòa D_Tầng 2_Thư viện", "Tòa D_Tầng 3_Quầy giáo trình", {"has_roof": True, "status": "open", "edge_type": "stairs"}),
        ("Tòa D_Tầng 1_Căn tin", "Tòa D_Tầng 2_Thư viện", {"has_roof": True, "status": "open", "edge_type": "elevator"}),
        ("Tòa D_Tầng 2_Thư viện", "Tòa D_Tầng 3_Quầy giáo trình", {"has_roof": True, "status": "open", "edge_type": "elevator"}),

        # Tòa E: Thang bộ
        ("Tòa E_Tầng 1_Phòng học 101", "Tòa E_Tầng 2_Phòng nghỉ trưa", {"has_roof": True, "status": "open", "edge_type": "stairs"}),

        # Tòa F: Thang bộ
        ("Tòa F_Tầng 1_Phòng nghỉ 102", "Tòa F_Tầng 2_Phòng tự học F201", {"has_roof": True, "status": "open", "edge_type": "stairs"}),

        # --- CẦU NỐI GIỮA CÁC TÒA (BRIDGES) ---
        ("Tòa B_Tầng 2_Tự học B201", "Tòa C_Tầng 2_Lab máy tính 202", {"has_roof": True, "status": "open", "edge_type": "bridge"}),
        ("Tòa D_Tầng 2_Thư viện", "Tòa E_Tầng 2_Phòng nghỉ trưa", {"has_roof": True, "status": "open", "edge_type": "bridge"}),

        # --- ĐƯỜNG NGOÀI TRỜI (KHÔNG CÓ MÁI CHE) ---
        ("Nhà xe",       "Tòa B_Tầng 1_Sảnh",          {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        ("Nhà xe",       "Tòa C_Tầng 1_Sảnh",          {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        ("Tòa D_Tầng 1_Căn tin", "Nhà thể dục",        {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        ("Tòa E_Tầng 1_Phòng học 101", "Nhà thể dục",  {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        
        ("Nhà xe",       "Tòa D_Tầng 1_Căn tin",       {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        ("Nhà xe",       "Nhà điều hành",              {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        ("Cổng trường",  "Nhà xe",                     {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        ("Cổng trường",  "Tòa A_Tầng 1_Sảnh",          {"has_roof": False, "status": "open", "edge_type": "walkway"}),

        ("Nhà điều hành","ATM",                        {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        ("ATM",          "Tòa E_Tầng 1_Phòng học 101", {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        ("ATM",          "Tòa F_Tầng 1_Phòng nghỉ 102",{"has_roof": False, "status": "open", "edge_type": "walkway"}),
        ("Tòa D_Tầng 1_Căn tin", "ATM",                {"has_roof": False, "status": "open", "edge_type": "walkway"}),
    ]

    for u, v, attr in edges:
        lat1, lon1 = G.nodes[u]["gps"]
        lat2, lon2 = G.nodes[v]["gps"]
        
        # Nếu đi thang máy/thang bộ trong cùng tòa, khoảng cách thực là nhỏ nhưng ta gán chi phí ảo (VD: 10m/tầng)
        if "_" in u and "_" in v and u.split("_")[0] == v.split("_")[0]:
            dist = 10.0 if attr["edge_type"] == "stairs" else 12.0
        else:
            dist = round(haversine(lat1, lon1, lat2, lon2), 2)
            
        attr["weight"] = dist
        G.add_edge(u, v, **attr)

    return G


def get_canvas_bounds(G) -> dict:
    """
    Tính min/max tọa độ của đồ thị để frontend có thể scale động.
    Trả về dict chứa min_x, max_x, min_y, max_y.
    """
    xs = [d["pos"][0] for _, d in G.nodes(data=True)]
    ys = [d["pos"][1] for _, d in G.nodes(data=True)]
    return {
        "min_x": min(xs), "max_x": max(xs),
        "min_y": min(ys), "max_y": max(ys),
    }


def visualize_flat_graph(G: nx.Graph):
    """Vẽ đồ thị dựa trên tọa độ GPS (dùng cho debug local)."""
    plt.figure(figsize=(10, 8))
    pos = nx.get_node_attributes(G, "pos")
    edges = G.edges(data=True)

    open_edges   = [(u, v) for u, v, d in edges if d["status"] == "open"]
    repair_edges = [(u, v) for u, v, d in edges if d["status"] == "repairing"]

    nx.draw_networkx_edges(G, pos, edgelist=open_edges,   width=2, edge_color="#888888")
    nx.draw_networkx_edges(G, pos, edgelist=repair_edges, width=2, edge_color="red", style="dashed")

    node_colors = []
    for _, data in G.nodes(data=True):
        if   data.get("type") == "facility": node_colors.append("lightgreen")
        elif data.get("type") == "admin":    node_colors.append("lightgrey")
        else:                                node_colors.append("skyblue")

    nx.draw_networkx_nodes(G, pos, node_size=800, node_color=node_colors, edgecolors="black")
    edge_labels = {(u, v): f"{d['weight']}m" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="red")
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

    plt.title("Bản đồ 2D AI AR Campus (GPS Thực tế)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    campus_graph = build_campus_graph()
    visualize_flat_graph(campus_graph)
