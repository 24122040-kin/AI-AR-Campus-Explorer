# engine/graph_builder_v2.py
import networkx as nx
import matplotlib.pyplot as plt
from engine.building_catalog import _BUILDING_PROFILES
from engine.campus_knowledge import CAMPUS_LINH_TRUNG, get_cluster_for_node
from engine.utils import haversine


def build_campus_graph() -> nx.Graph:
    """
    Xây dựng đồ thị campus (Node = tòa nhà/tiện ích, Edge = đường đi).
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
    # 1. ĐỊNH NGHĨA NODE FLAT BUILDING-LEVEL
    # ---------------------------------------------------------
    nodes_data = {
        "Tòa A": {
            "gps": (10.877500, 106.797500), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.53, "capacity": 300},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["toa a", "nha a", "sanh toa a", "tang 1 toa a", "phong thi nghiem a201", "lab a201", "thuc nghiem a201", "phong thi nghiem a301", "lab a301", "thuc nghiem a301"]
        },
        "Tòa B": {
            "gps": (10.877500, 106.798000), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.33, "capacity": 240},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["toa b", "nha b", "sanh toa b", "phong tu hoc b201", "tu hoc b201", "hoc nhom b201", "tu hoc yen tinh", "phong may b301", "lab b301", "thuc hanh b301", "phong lab b301"]
        },
        "Tòa C": {
            "gps": (10.877500, 106.798500), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.37, "capacity": 200},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["toa c", "nha c", "sanh toa c", "lab may tinh 202", "phong may 202", "lab cntt", "may tinh", "phong thuc hanh may tinh", "van phong khoa", "vp khoa", "giao vu khoa"]
        },
        "Tòa D": {
            "gps": (10.878000, 106.798750), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.37, "capacity": 380},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["toa d", "nha d", "thu vien", "doc sach", "muon sach", "thu vien khtn", "cho doc sach", "quay giao trinh", "mua sach", "tiem sach"]
        },
        "Căn tin": {
            "gps": (10.878050, 106.798700), "type": "facility",
            "features": {"has_ac": 0, "has_tables": 1, "noise_level": 0.65, "capacity": 300},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["can tin", "canteen", "an trua", "com can tin", "doi bung", "an uong", "tra sua", "an vat"]
        },
        "Tòa E": {
            "gps": (10.877500, 106.799000), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.3, "capacity": 170},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["toa e", "nha e", "phong hoc 101", "ly thuyet", "phong nghi trua", "cho ngu trua", "nghi trua"]
        },
        "Tòa F": {
            "gps": (10.877500, 106.799500), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.2, "capacity": 100},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["toa f", "nha f", "phong nghi 102", "cho nga lung", "buon ngu", "met qua", "phong tu hoc f201", "tu hoc f201"]
        },
        "Tòa G": {
            "gps": (10.877500, 106.800000), "type": "building",
            "features": {"has_ac": 0, "has_tables": 0, "noise_level": 0.5, "capacity": 200},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["toa g", "nha g", "san toa g"]
        },
        "Nhà thể dục": {
            "gps": (10.878700, 106.799250), "type": "building",
            "features": {"has_ac": 0, "has_tables": 0, "noise_level": 0.8, "capacity": 1000},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["nha the duc", "gym", "the thao", "clb", "tap gym", "cau long", "bong ban"]
        },
        "Nhà xe": {
            "gps": (10.876300, 106.797500), "type": "facility",
            "features": {"has_ac": 0, "has_tables": 0, "noise_level": 0.9, "capacity": 1000},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["bai giu xe", "parking", "gui xe", "lay xe", "cat xe", "xe may", "nha de xe"]
        },
        "ATM": {
            "gps": (10.876800, 106.799000), "type": "facility",
            "features": {"has_ac": 0, "has_tables": 0, "noise_level": 0.5, "capacity": 5},
            "open_time": "00:00", "close_time": "23:59",
            "aliases": ["cay atm", "rut tien", "het tien", "ngan hang", "tien mat"]
        },
        "Nhà điều hành": {
            "gps": (10.876100, 106.799200), "type": "admin",
            "features": {"has_ac": 1, "has_tables": 0, "noise_level": 0.1, "capacity": 100},
            "open_time": REST_OPEN, "close_time": REST_CLOSE,
            "restricted": True,
            "aliases": ["phong ban", "giao vu", "hanh chinh", "giay to", "dong hoc phi", "staff only"]
        },
        "Cổng trường": {
            "gps": (10.876000, 106.798500), "type": "facility",
            "features": {"has_ac": 0, "has_tables": 0, "noise_level": 0.6, "capacity": 200},
            "open_time": "00:00", "close_time": "23:59",
            "aliases": ["cong truong", "cong chinh", "cong", "entrance", "main gate"]
        },
    }

    # Thêm Node vào đồ thị
    for name, data in nodes_data.items():
        catalog = _BUILDING_PROFILES.get(name, {})
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
            poi_cluster=get_cluster_for_node(name),
            campus_id=CAMPUS_LINH_TRUNG["id"],
        )

    # ---------------------------------------------------------
    # 2. ĐỊNH NGHĨA EDGE FLAT BUILDING-LEVEL
    # ---------------------------------------------------------
    edges = [
        # Đường hành lang có mái che (Horizontal line: A -> B -> C -> E -> F -> G)
        ("Tòa A",        "Tòa B",          {"has_roof": True,  "status": "open", "edge_type": "corridor"}),
        ("Tòa B",        "Tòa C",          {"has_roof": True,  "status": "open", "edge_type": "corridor"}),
        ("Tòa C",        "Tòa E",          {"has_roof": True,  "status": "open", "edge_type": "corridor"}),
        ("Tòa E",        "Tòa F",          {"has_roof": True,  "status": "open", "edge_type": "corridor"}),
        ("Tòa F",        "Tòa G",          {"has_roof": True,  "status": "open", "edge_type": "corridor"}),
        
        # Tam giác Căn tin (C -> D -> E) - Có mái che
        ("Tòa C",        "Tòa D",          {"has_roof": True,  "status": "open", "edge_type": "corridor"}),
        ("Tòa D",        "Tòa E",          {"has_roof": True,  "status": "open", "edge_type": "corridor"}),
        
        # Nhà thi đấu (Nhà thể dục) nối E và F (Không mái che)
        ("Tòa E",        "Nhà thể dục",    {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        ("Tòa F",        "Nhà thể dục",    {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        
        # Cổng trường đến Nhà xe và Nhà điều hành (Không mái che)
        ("Cổng trường",  "Nhà xe",         {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        ("Cổng trường",  "Nhà điều hành",  {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        
        # Nhà điều hành nối E và F (Không mái che)
        ("Nhà điều hành", "Tòa E",          {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        ("Nhà điều hành", "Tòa F",          {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        
        # Kết nối ATM (Còn lại đều có mái che -> True)
        ("Nhà điều hành", "ATM",            {"has_roof": True,  "status": "open", "edge_type": "walkway"}),
        ("ATM",          "Tòa E",          {"has_roof": True,  "status": "open", "edge_type": "walkway"}),
        ("ATM",          "Tòa F",          {"has_roof": True,  "status": "open", "edge_type": "walkway"}),
        ("Tòa D",        "ATM",            {"has_roof": True,  "status": "open", "edge_type": "walkway"}),
        ("Tòa D",        "Căn tin",        {"has_roof": True,  "status": "open", "edge_type": "corridor"}),
    ]

    for u, v, attr in edges:
        lat1, lon1 = G.nodes[u]["gps"]
        lat2, lon2 = G.nodes[v]["gps"]
        dist = round(haversine(lat1, lon1, lat2, lon2), 2)
        attr["weight"] = dist
        G.add_edge(u, v, **attr)

    # ---------------------------------------------------------
    # 3. ĐỊNH NGHĨA INDOOR MULTI-FLOOR NODES & EDGES (Tầng toà nhà)
    # ---------------------------------------------------------
    indoor_configs = {
        "Tòa A": {
            "center": (10.877500, 106.797500),
            "floors": {
                "G": ["Phòng A.G01", "Phòng A.G02"],
                "1": ["Phòng A.101", "Phòng A.102"],
                "2": ["Phòng A.201 (Lab AI)", "Phòng A.202"],
                "3": ["Phòng A.301", "Phòng A.302"]
            }
        },
        "Tòa B": {
            "center": (10.877500, 106.798000),
            "floors": {
                "G": ["Phòng B.G01"],
                "1": ["Phòng B.101", "Phòng B.102"],
                "2": ["Phòng tự học B201", "Phòng B.202"],
                "3": ["Phòng máy B301", "Phòng B.302"]
            }
        },
        "Tòa C": {
            "center": (10.877500, 106.798500),
            "floors": {
                "G": ["Văn phòng Khoa C"],
                "1": ["Phòng C.101", "Phòng C.102"],
                "2": ["Phòng máy 202", "Phòng C.202"]
            }
        },
        "Tòa D": {
            "center": (10.878000, 106.798750),
            "floors": {
                "G": ["Quầy Giáo trình D"],
                "1": ["Thư viện Tòa D", "Phòng Y tế"],
                "2": ["Thư viện Tầng 2"]
            }
        }
    }

    # Đặt thuộc tính mặc định cho các node campus ngoài trời
    for n in G.nodes:
        G.nodes[n]["building"] = n if n in indoor_configs else None
        G.nodes[n]["floor"] = "G" if n in indoor_configs else None
        G.nodes[n]["is_indoor"] = False

    for b_name, config in indoor_configs.items():
        lat_c, lon_c = config["center"]
        floors = config["floors"]
        
        # Thang và Sảnh xuyên suốt các tầng
        for floor, rooms in floors.items():
            lobby = f"Sảnh {b_name} ({floor})"
            stairs = f"Cầu thang bộ {b_name} ({floor})"
            elevator = f"Thang máy {b_name} ({floor})"
            
            # Tọa độ phụ trợ
            # Lobbies ở tâm tòa nhà
            # Cầu thang góc Đông Bắc (offset y+, x-)
            # Thang máy góc Tây Bắc (offset y+, x+)
            G.add_node(lobby, pos=(lon_c, lat_c), gps=(lat_c, lon_c), type="lobby", building=b_name, floor=floor, is_indoor=True, open_time="06:00", close_time="18:00", aliases=[lobby.lower()])
            G.add_node(stairs, pos=(lon_c - 0.00008, lat_c + 0.00008), gps=(lat_c + 0.00008, lon_c - 0.00008), type="stairs", building=b_name, floor=floor, is_indoor=True, open_time="06:00", close_time="18:00", aliases=[stairs.lower(), f"cầu thang {b_name}".lower()])
            G.add_node(elevator, pos=(lon_c + 0.00008, lat_c + 0.00008), gps=(lat_c + 0.00008, lon_c + 0.00008), type="elevator", building=b_name, floor=floor, is_indoor=True, open_time="06:00", close_time="18:00", aliases=[elevator.lower(), f"thang máy {b_name}".lower()])
            
            # Kết nối ngang trên tầng
            G.add_edge(lobby, stairs, has_roof=True, status="open", edge_type="corridor", weight=10.0)
            G.add_edge(lobby, elevator, has_roof=True, status="open", edge_type="corridor", weight=10.0)
            
            # Thêm các phòng
            for idx, room in enumerate(rooms):
                # Phòng bố trí góc Nam (y-)
                # Room 0: Tây Nam, Room 1: Đông Nam
                if len(rooms) == 1:
                    r_lat = lat_c - 0.00008
                    r_lon = lon_c
                else:
                    r_lat = lat_c - 0.00008
                    r_lon = lon_c - 0.00008 if idx == 0 else lon_c + 0.00008
                    
                # Tạo aliases thông minh
                r_aliases = [room.lower(), room.replace("Phòng ", "").lower(), f"{b_name} {room}".lower()]
                G.add_node(room, pos=(r_lon, r_lat), gps=(r_lat, r_lon), type="room", building=b_name, floor=floor, is_indoor=True, open_time="06:00", close_time="18:00", aliases=r_aliases)
                G.add_edge(lobby, room, has_roof=True, status="open", edge_type="corridor", weight=12.0)
            
            # Kết nối Sảnh G với Node tòa nhà campus ngoài trời
            if floor == "G":
                G.add_edge(b_name, lobby, has_roof=True, status="open", edge_type="corridor", weight=2.0)

        # Kết nối dọc các tầng (Cầu thang & Thang máy)
        floor_keys = list(floors.keys())  # ["G", "1", "2", "3"]
        for i in range(len(floor_keys) - 1):
            f_curr = floor_keys[i]
            f_next = floor_keys[i+1]
            
            stairs_curr = f"Cầu thang bộ {b_name} ({f_curr})"
            stairs_next = f"Cầu thang bộ {b_name} ({f_next})"
            elevator_curr = f"Thang máy {b_name} ({f_curr})"
            elevator_next = f"Thang máy {b_name} ({f_next})"
            
            # Cầu thang: weight = 15m đi bộ lên tầng
            G.add_edge(stairs_curr, stairs_next, has_roof=True, status="open", edge_type="stairs", weight=15.0)
            # Thang máy: weight = 8m (nhanh hơn đi bộ)
            G.add_edge(elevator_curr, elevator_next, has_roof=True, status="open", edge_type="elevator", weight=8.0)

    return G


def get_canvas_bounds(G) -> dict:
    """
    Tính min/max tọa độ của đồ thị để frontend có thể scale động.
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
