# engine/graph_builder_v2.py
import networkx as nx
import matplotlib.pyplot as plt
from engine.building_catalog import _BUILDING_PROFILES
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
    # 1. ĐỊNH NGHĨA NODE
    # features là dict để tránh positional indexing dễ vỡ
    # features keys: has_ac, has_tables, noise_level, capacity
    # ---------------------------------------------------------
    nodes_data = {
        "Tòa A": {
            "gps": (10.871000, 106.801500), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.5, "capacity": 300},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["nha a", "phong thi nghiem"]
        },
        "Nhà thể dục": {
            "gps": (10.871800, 106.803200), "type": "building",
            "features": {"has_ac": 0, "has_tables": 0, "noise_level": 0.8, "capacity": 1000},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["nha the duc", "gym", "the thao", "clb", "tap gym", "cau long", "bong ban"]
        },
        "Tòa B": {
            "gps": (10.871000, 106.802000), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.2, "capacity": 100},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["nha b", "phong tu hoc", "yen tinh", "hoc nhom"]
        },
        "Tòa C": {
            "gps": (10.871000, 106.802500), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.4, "capacity": 80},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["nha c", "lab", "thuc hanh", "may tinh"]
        },
        "Tòa D": {
            "gps": (10.871000, 106.803000), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.6, "capacity": 500},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["nha d", "quay giao trinh", "thu vien", "can tin", "cho doc sach",
                        "an trua", "doi bung", "mua sach", "muon sach"]
        },
        "Tòa E": {
            "gps": (10.871000, 106.803500), "type": "building",
            "features": {"has_ac": 1, "has_tables": 0, "noise_level": 0.2, "capacity": 50},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["nha e", "hoc ly thuyet", "phong nghi", "cho ngu trua", "nghi trua"]
        },
        "Tòa F": {
            "gps": (10.871000, 106.804000), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.2, "capacity": 50},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["nha f", "phong nghi", "cho ngu trua", "nghi trua", "buon ngu", "met qua"]
        },
        "Tòa G": {
            "gps": (10.871000, 106.804500), "type": "building",
            "features": {"has_ac": 0, "has_tables": 0, "noise_level": 0.5, "capacity": 200},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["nha g"]
        },
        "Nhà xe": {
            "gps": (10.870300, 106.801000), "type": "facility",
            "features": {"has_ac": 0, "has_tables": 0, "noise_level": 0.9, "capacity": 1000},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["bai giu xe", "parking", "gui xe", "lay xe", "cat xe",
                        "ra ve", "xe may", "nha de xe"]
        },
        "ATM": {
            "gps": (10.870800, 106.802750), "type": "facility",
            "features": {"has_ac": 1, "has_tables": 0, "noise_level": 0.5, "capacity": 5},
            "open_time": "00:00", "close_time": "23:59",
            "aliases": ["cay atm", "rut tien", "het tien", "ngan hang", "tien mat"]
        },
        "Nhà điều hành": {
            "gps": (10.869800, 106.803000), "type": "admin",
            "features": {"has_ac": 1, "has_tables": 0, "noise_level": 0.1, "capacity": 100},
            "open_time": REST_OPEN, "close_time": REST_CLOSE,
            "restricted": True,
            "aliases": ["phong ban", "giao vu", "hanh chinh", "giay to",
                        "dong hoc phi", "dong tien", "phong nghi", "cho ngu trua",
                        "cam", "khong phan su", "staff only"]
        },
        "Cổng trường": {
            "gps": (10.870500, 106.801200), "type": "facility",
            "features": {"has_ac": 0, "has_tables": 0, "noise_level": 0.6, "capacity": 200},
            "open_time": "00:00", "close_time": "23:59",
            "aliases": ["cong truong", "cong chinh", "cổng", "cong", "entrance", "main gate"]
        },
    }

    # Thêm Node vào đồ thị
    for name, data in nodes_data.items():
        catalog = _BUILDING_PROFILES.get(name, {})
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
        )

    # ---------------------------------------------------------
    # 2. ĐỊNH NGHĨA EDGE — tự động tính weight (mét) bằng Haversine
    # ---------------------------------------------------------
    edges = [
        # Hành lang có mái che
        ("Tòa A",        "Tòa B",          {"has_roof": True,  "status": "open"}),
        ("Tòa B",        "Tòa C",          {"has_roof": True,  "status": "open"}),
        ("Tòa C",        "Tòa D",          {"has_roof": True,  "status": "open"}),
        ("Tòa D",        "Tòa E",          {"has_roof": True,  "status": "open"}),
        ("Tòa E",        "Tòa F",          {"has_roof": True,  "status": "open"}),
        ("Tòa F",        "Tòa G",          {"has_roof": True,  "status": "open"}),
        
        # Đường ngoài trời
        ("Nhà xe",       "Tòa B",          {"has_roof": False, "status": "open"}),
        ("Nhà xe",       "Tòa C",          {"has_roof": False, "status": "open"}),
        ("Tòa D",        "Nhà thể dục",    {"has_roof": False, "status": "open"}),
        ("Tòa E",        "Nhà thể dục",    {"has_roof": False, "status": "open"}),
        
        # CÁC LỐI ĐI BỊ CẤM (Theo dấu X đỏ trên bản đồ) -> status = repairing
        ("Nhà xe",       "Tòa D",          {"has_roof": False, "status": "open"}),
        ("Nhà xe",       "Nhà điều hành",  {"has_roof": False, "status": "open"}),
        ("Cổng trường",  "Nhà xe",         {"has_roof": False, "status": "open"}),
        ("Cổng trường",  "Tòa A",          {"has_roof": False, "status": "open"}),

        ("Nhà điều hành","ATM",            {"has_roof": False, "status": "open"}),
        ("ATM",          "Tòa E",          {"has_roof": False, "status": "open"}),
        ("ATM",          "Tòa F",          {"has_roof": False, "status": "open"}),
        ("Tòa D",        "ATM",            {"has_roof": False, "status": "open"}),
    ]

    for u, v, attr in edges:
        lat1, lon1 = G.nodes[u]["gps"]
        lat2, lon2 = G.nodes[v]["gps"]
        attr["weight"] = round(haversine(lat1, lon1, lat2, lon2), 2)
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
