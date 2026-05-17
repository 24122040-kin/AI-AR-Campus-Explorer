# main.py
"""
AI AR Campus API — Phiên bản đã refactor và tối ưu.
Các cải tiến so với bản cũ:
  - Xóa hàm haversine trùng lặp, dùng engine.utils.haversine
  - Input validation với Pydantic / FastAPI Query
  - Tính khoảng cách còn lại chính xác (calc_remaining_distance)
  - Canvas scale động từ bounds thực của đồ thị
  - Score AI gợi ý chuẩn hóa về [0, 100]
  - HTML tách thành hàm riêng, escape {{ }} rõ ràng
  - Hằng số WALKING_SPEED_MPM thay cho magic number
  - Thêm tính năng Gợi ý chủ động (Proactive Recommendation)
  - Tích hợp path_coords cho AR Tracking và Info Panel cho Web UI
  - Thêm hiển thị Bảng tọa độ GPS trực tiếp trên Web UI để test
  - TÍCH HỢP MỚI: Quản lý không gian động (Dynamic Edge) & Geofencing
"""

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse

from engine.graph_builder_v2 import build_campus_graph, get_canvas_bounds
from engine.optimizer import (
    pathfinding_optimizer,
    multi_stop_routing,
    is_node_open,
    calc_remaining_distance,
    dynamic_edge_update,
    geofencing_logic,
    restricted_zone_alert,
)
from engine.nlp_processor import find_node_by_keyword
from engine.building_catalog import get_building_profile, list_all_building_guides
from engine.recommender import (
    recommend_location,
    recommend_locations,
    recommend_by_building_function,
    get_proactive_recommendations,
    get_smart_recommendations,
    context_recommender,
    semantic_map_linking,
    collaborative_filtering,
    crowd_prediction,
    predict_crowd_level,
)
from engine.gnn_engine import (
    gnn_node_embedding,
    graph_attention_layer,
    graph_rag_context,
    inductive_learning,
)
from engine.utils import haversine, get_current_time_str, WALKING_SPEED_MPM

# ---------------------------------------------------------------------------
# Khởi tạo ứng dụng và đồ thị + GNN
# ---------------------------------------------------------------------------
app = FastAPI(title="AI AR Campus API — GNN & Semantic Navigator")

G          = build_campus_graph()
list_nodes = sorted(G.nodes())
_bounds    = get_canvas_bounds(G)

# Khởi tạo embedding GNN lúc startup
_gnn_ready = False
try:
    gnn_node_embedding(G)
    _gnn_ready = True
except Exception as _gnn_err:
    print(f"[WARN] GNN init: {_gnn_err}")


# ===========================================================================
# API MỚI: Quản lý không gian động (Dynamic Edge)
# ===========================================================================
@app.get("/api_update_edge")
def update_edge(u: str, v: str, status: str):
    """Giả lập sự kiện đóng/mở đường đi (repairing, open, closed)"""
    success = dynamic_edge_update(G, u, v, status)
    if success:
        return {"status": "success", "message": f"Đã cập nhật tuyến {u} - {v} thành {status}"}
    raise HTTPException(status_code=404, detail="Không tìm thấy tuyến đường này trên đồ thị")


# ===========================================================================
# API MỚI: Gợi ý chủ động theo ngữ cảnh (Proactive Recommender)
# ===========================================================================
@app.get("/api_proactive_recommend")
def proactive_recommend(
    current_lat: float = Query(...),
    current_lon: float = Query(...),
    destination: Optional[str] = Query(None, description="Điểm đích người dùng đang hướng tới"),
    query: Optional[str] = Query(None, description="Nhu cầu tự nhiên, VD: an trua, hoc bai"),
    weather: str = Query("normal"),
    interests: Optional[str] = Query(None, description="Sở thích, phân tách bởi dấu phẩy"),
    limit: int = Query(5, ge=1, le=10),
):
    if destination and destination not in G.nodes:
        raise HTTPException(status_code=400, detail=f"Node đích không tồn tại: '{destination}'")

    now = get_current_time_str()
    user_interests = [i.strip() for i in interests.split(",") if i.strip()] if interests else None
    suggestions = get_proactive_recommendations(
        G, current_lat, current_lon, now,
        destination=destination,
        query=query,
        weather=weather,
        user_interests=user_interests,
        limit=limit,
    )

    return {
        "status": "success",
        "suggestions": suggestions,
        "context": {
            "destination": destination,
            "query": query,
            "weather": weather,
        },
    }


@app.get("/api_smart_recommend")
def smart_recommend(
    current_lat: float = Query(...),
    current_lon: float = Query(...),
    destination: Optional[str] = Query(None),
    query: Optional[str] = Query(None),
    weather: str = Query("normal"),
    interests: Optional[str] = Query(None),
    limit: int = Query(6, ge=1, le=12),
):
    """API gợi ý AI đầy đủ: tọa độ + đích + câu hỏi + sở thích."""
    if destination and destination not in G.nodes:
        raise HTTPException(status_code=400, detail=f"Node đích không tồn tại: '{destination}'")

    now = get_current_time_str()
    user_interests = [i.strip() for i in interests.split(",") if i.strip()] if interests else None
    items = get_smart_recommendations(
        G, current_lat, current_lon,
        destination=destination,
        query=query,
        weather=weather,
        current_time_str=now,
        user_interests=user_interests,
        limit=limit,
    )
    return {"status": "success", "recommendations": items, "current_time": now}


@app.get("/api_context_recommend")
def api_context_recommend(
    current_lat: float = Query(...),
    current_lon: float = Query(...),
    destination: Optional[str] = Query(None),
    query: Optional[str] = Query(None),
    weather: str = Query("normal"),
    interests: Optional[str] = Query(None),
    limit: int = Query(6, ge=1, le=12),
):
    """context_recommender — gợi ý theo thời gian + vị trí + GNN."""
    if destination and destination not in G.nodes:
        raise HTTPException(status_code=400, detail=f"Node đích không tồn tại: '{destination}'")
    now = get_current_time_str()
    user_interests = [i.strip() for i in interests.split(",") if i.strip()] if interests else None
    items = context_recommender(
        G, current_lat, current_lon, now,
        destination=destination, query=query,
        weather=weather, user_interests=user_interests, limit=limit,
    )
    return {"status": "success", "recommendations": items, "gnn_ready": _gnn_ready}


@app.get("/api_semantic_map")
def api_semantic_map(
    query: str = Query(..., min_length=2),
    current_lat: Optional[float] = Query(None),
    current_lon: Optional[float] = Query(None),
):
    """semantic_map_linking — mô tả → tọa độ đồ thị."""
    result = semantic_map_linking(G, query, current_lat, current_lon)
    if not result:
        raise HTTPException(status_code=404, detail="Không ánh xạ được mô tả sang bản đồ.")
    return {"status": "success", **result}


@app.get("/api_crowd")
def api_crowd(node: str = Query(...)):
    if node not in G.nodes:
        raise HTTPException(status_code=400, detail=f"Node không tồn tại: '{node}'")
    now = get_current_time_str()
    return {"status": "success", **crowd_prediction(G, node, now)}


@app.get("/api_collaborative")
def api_collaborative(
    interests: str = Query(..., description="Sở thích, phân tách bởi dấu phẩy"),
    current_lat: Optional[float] = Query(None),
    current_lon: Optional[float] = Query(None),
):
    """collaborative_filtering — gợi ý CLB/Lab."""
    user_interests = [i.strip() for i in interests.split(",") if i.strip()]
    items = collaborative_filtering(G, user_interests, current_lat, current_lon)
    return {"status": "success", "suggestions": items}


@app.get("/api_gnn_embeddings")
def api_gnn_embeddings():
    return {
        "status": "success",
        "gnn_ready": _gnn_ready,
        "embeddings": gnn_node_embedding(G),
        "edge_attention": graph_attention_layer(G),
    }


@app.get("/api_graph_rag")
def api_graph_rag(
    current_lat: Optional[float] = Query(None),
    current_lon: Optional[float] = Query(None),
):
    """graph_rag_context — ngữ cảnh cấu trúc campus cho Member 2 / LLM."""
    now = get_current_time_str()
    ctx = graph_rag_context(G, now, current_lat, current_lon)
    return {"status": "success", "rag": ctx}


@app.post("/api_add_location")
def api_add_location(
    node_id: str = Query(...),
    lat: float = Query(...),
    lon: float = Query(...),
    connect_to: str = Query(..., description="Node láng giềng, phân tách dấu phẩy"),
    node_type: str = Query("building"),
):
    """inductive_learning — thêm địa điểm mới không train lại."""
    neighbors = [n.strip() for n in connect_to.split(",") if n.strip()]
    invalid = [n for n in neighbors if n not in G.nodes]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Node láng giềng không tồn tại: {invalid}")
    result = inductive_learning(G, node_id, (lat, lon), neighbors, node_type=node_type)
    global list_nodes
    list_nodes = sorted(G.nodes())
    return result


@app.get("/api_building_guide")
def api_building_guide():
    """Tra cứu chức năng từng tòa trên campus."""
    return {"status": "success", "buildings": list_all_building_guides(G)}


@app.get("/api_building_info")
def api_building_info(node: str = Query(...)):
    if node not in G.nodes:
        raise HTTPException(status_code=400, detail=f"Node không tồn tại: '{node}'")
    now = get_current_time_str()
    profile = get_building_profile(G, node)
    profile["is_open"] = is_node_open(G, node, now)
    profile["crowd"] = crowd_prediction(G, node, now)
    return {"status": "success", **profile}


@app.get("/api_restricted_alert")
def api_restricted_alert(
    current_lat: float = Query(...),
    current_lon: float = Query(...),
    radius: float = Query(30.0),
):
    alerts = restricted_zone_alert(G, current_lat, current_lon, radius)
    return {"status": "success", "alerts": alerts, "in_restricted_zone": len(alerts) > 0}


# ===========================================================================
# 1. API: Cấu trúc đồ thị & Trạng thái đóng/mở
# ===========================================================================
@app.get("/api_get_graph")
def get_graph():
    now   = get_current_time_str()
    nodes = []
    for n in G.nodes():
        profile = get_building_profile(G, n)
        nodes.append({
            "id": n,
            "x": G.nodes[n]["pos"][0],
            "y": G.nodes[n]["pos"][1],
            "type": G.nodes[n]["type"],
            "is_open": is_node_open(G, n, now),
            "features": G.nodes[n].get("features", {}),
            "hours": f"{G.nodes[n].get('open_time', 'N/A')} - {G.nodes[n].get('close_time', 'N/A')}",
            "tagline": profile.get("tagline", ""),
            "function_summary": profile.get("function_summary", ""),
            "services": profile.get("services", []),
        })
    edges = [
        {"source": u, "target": v, "status": d["status"], "has_roof": d["has_roof"]}
        for u, v, d in G.edges(data=True)
    ]
    return {
        "nodes":        nodes,
        "edges":        edges,
        "current_time": now,
        "bounds":       _bounds,   # min_x, max_x, min_y, max_y cho canvas scale
    }


# ===========================================================================
# 2. API: Lập lộ trình đa điểm
# ===========================================================================
@app.get("/api_get_route")
def get_route(
    waypoints: str = Query(..., description="Danh sách node cách nhau bởi dấu phẩy"),
    weather:   str = Query("normal", description="normal | sunny | rainy"),
):
    pts = [p.strip() for p in waypoints.split(",") if p.strip()]

    if len(pts) < 2:
        raise HTTPException(status_code=400, detail="Cần ít nhất điểm bắt đầu và đích!")

    # Validate từng node
    invalid = [p for p in pts if p not in G.nodes]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Node không tồn tại: {invalid}")

    now  = get_current_time_str()
    path, all_open = multi_stop_routing(G, pts, weather, now)

    if not path:
        raise HTTPException(status_code=404, detail="Không tìm thấy lộ trình qua các điểm này!")

    route_coords = []
    for i, n in enumerate(path):
        lat, lon = G.nodes[n]["gps"]
        seg = {
            "node": n,
            "gps": [lat, lon],
            "order": i,
            "is_open": is_node_open(G, n, now),
            "crowd": predict_crowd_level(G, n, now),
        }
        if i < len(path) - 1:
            u, v = n, path[i + 1]
            ed = G[u][v]
            seg["next_bearing_hint"] = _bearing_hint(lat, lon, G.nodes[v]["gps"][0], G.nodes[v]["gps"][1])
            seg["edge"] = {
                "has_roof": ed.get("has_roof"),
                "weight_m": ed.get("weight"),
                "status": ed.get("status"),
            }
        route_coords.append(seg)

    total_m = sum(
        G[path[i]][path[i + 1]].get("weight", 0) for i in range(len(path) - 1)
    )

    return {
        "status": "success",
        "path": path,
        "coordinates": route_coords,
        "ar_waypoints": [{"node": c["node"], "gps": c["gps"]} for c in route_coords],
        "total_distance_m": round(total_m, 2),
        "estimated_mins": round(total_m / WALKING_SPEED_MPM, 1),
        "all_open": all_open,
        "routing_engine": "A* + GNN-GAT" if _gnn_ready else "A*",
    }


def _bearing_hint(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    import math
    d_lon = math.radians(lon2 - lon1)
    y = math.sin(d_lon) * math.cos(math.radians(lat2))
    x = (
        math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
        - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(d_lon)
    )
    deg = (math.degrees(math.atan2(y, x)) + 360) % 360
    dirs = ["Bắc", "Đông Bắc", "Đông", "Đông Nam", "Nam", "Tây Nam", "Tây", "Tây Bắc"]
    return dirs[int((deg + 22.5) / 45) % 8]


# ===========================================================================
# 3. API: Tìm kiếm thông minh (NLP + AI Recommender)
# ===========================================================================
@app.get("/api_search")
def search_semantic(
    query: str = Query(..., min_length=1, description="Câu hỏi tìm kiếm"),
    weather: str = Query("normal", description="Truyền từ UI để tính context")
):
    now = get_current_time_str()

    # Lớp 1: Khớp tên / alias (NLP cơ bản)
    node = find_node_by_keyword(G, query)
    if node:
        gps = G.nodes[node]["gps"]
        return {
            "status":       "success",
            "matched_node": node,
            "is_open":      is_node_open(G, node, now),
            "score":        None,
            "method":       "Keyword Match",
            "info":         G.nodes[node],
            "gps":          {"lat": gps[0], "lon": gps[1]},
        }

    # Lớp 1b: semantic_map_linking
    linked = semantic_map_linking(G, query)
    if linked:
        n = linked["node"]
        return {
            "status": "success",
            "matched_node": n,
            "is_open": is_node_open(G, n, now),
            "score": linked["confidence"] * 100,
            "method": "Semantic Map Linking",
            "info": G.nodes[n],
            "gps": linked["gps"],
            "alternatives": linked.get("alternatives", []),
        }

    # Lớp 2a: Gợi ý theo chức năng tòa (ăn, lab, thư viện...)
    by_func = recommend_by_building_function(G, query, now, limit=5)
    if by_func:
        top = by_func[0]
        return {
            "status": "success",
            "matched_node": top["node"],
            "is_open": True,
            "score": top["score"],
            "method": "Chức năng tòa (Building Services)",
            "info": G.nodes[top["node"]],
            "gps": {"lat": G.nodes[top["node"]]["gps"][0], "lon": G.nodes[top["node"]]["gps"][1]},
            "function_summary": top.get("function_summary"),
            "matched_services": top.get("matched_services", []),
            "recommendations": by_func,
        }

    # Lớp 2b: AI gợi ý ngữ cảnh — trả về top địa điểm đang mở
    ranked = recommend_locations(G, query, now, weather, limit=5)
    if ranked:
        top = ranked[0]
        return {
            "status":         "success",
            "matched_node":   top["node"],
            "is_open":        True,
            "score":          top["score"],
            "method":         f"AI Semantic (Điểm: {top['score']}/100)",
            "info":           G.nodes[top["node"]],
            "recommendations": [
                {**r, "info": G.nodes[r["node"]], "is_open": True}
                for r in ranked
            ],
        }

    return {"status": "error", "message": "Không tìm thấy địa điểm hoặc không rõ yêu cầu."}


# ===========================================================================
# 4. API: Tracking vị trí Real-time (TÍCH HỢP GEOFENCING)
# ===========================================================================
@app.get("/api_realtime_tracking")
def realtime_tracking(
    current_lat: float = Query(...),
    current_lon: float = Query(...),
    end:         str   = Query(...),
    weather:     str   = Query("normal"),
):
    if end not in G.nodes:
        raise HTTPException(status_code=400, detail=f"Node đích không tồn tại: '{end}'")

    # Tìm node gần nhất với vị trí GPS hiện tại
    nearest_node = min(
        G.nodes(),
        key=lambda n: haversine(current_lat, current_lon, *G.nodes[n]["gps"]),
    )
    dist_to_nearest = haversine(current_lat, current_lon, *G.nodes[nearest_node]["gps"])

    # THÊM MỚI: Quét Geofencing
    alerts = geofencing_logic(G, current_lat, current_lon, radius=25.0)

    # Đã đến nơi
    if nearest_node == end and dist_to_nearest < 5:
        return {"status": "arrived", "message": "Bạn đã đến nơi!", "geofence_alerts": alerts}

    now  = get_current_time_str()
    path, dest_open = pathfinding_optimizer(G, nearest_node, end, weather, now)

    if not path:
        raise HTTPException(status_code=404, detail="Không tìm thấy đường đi!")

    # Tính khoảng cách còn lại đúng cách (chỉ từ nearest_node đến đích)
    total_remaining = calc_remaining_distance(G, path, dist_to_nearest)
    estimated_mins  = round(total_remaining / WALKING_SPEED_MPM, 1)

    # Lấy tọa độ từng node trên path cho Team AR vẽ mũi tên
    path_coords = [{"node": n, "gps": G.nodes[n]["gps"]} for n in path]

    # Gợi ý AI dọc đường tới đích
    route_suggestions = get_smart_recommendations(
        G, current_lat, current_lon,
        destination=end,
        weather=weather,
        current_time_str=now,
        limit=4,
    )

    return {
        "status":                "tracking",
        "snapped_node":          nearest_node,
        "dist_to_node":          round(dist_to_nearest, 2),
        "total_remaining_meters":round(total_remaining, 2),
        "estimated_mins":        estimated_mins,
        "path":                  path,
        "path_coords":           path_coords,
        "dest_open":             dest_open,
        "node_info":             G.nodes[nearest_node],
        "geofence_alerts":       alerts,
        "route_suggestions":     route_suggestions,
    }


# ===========================================================================
# 5. Giao diện Web UI
# ===========================================================================
def _build_options_html() -> str:
    return "".join(f"<option value='{n}'>{n}</option>" for n in list_nodes)


@app.get("/", response_class=HTMLResponse)
def web_ui():
    opts = _build_options_html()
    # bounds dùng để tính scale canvas động ở JS
    b = _bounds

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>AI AR Campus — Smart Dashboard</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: #f0f2f5;
            display: flex;
            gap: 20px;
            padding: 20px;
            min-height: 100vh;
            overflow-x: hidden;
        }}
        .sidebar {{
            width: 340px;
            flex-shrink: 0;
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,.1);
            height: fit-content;
        }}
        .main-content {{
            flex-grow: 1;
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,.1);
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .gps-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 12px;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .gps-table th, .gps-table td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        .gps-table th {{
            background: #1a73e8;
            color: white;
            font-weight: bold;
        }}
        h2 {{ color: #1a73e8; margin-bottom: 10px; font-size: 22px; text-align: center; }}
        label {{
            font-size: 12px;
            font-weight: bold;
            color: #5f6368;
            display: block;
            margin-top: 15px;
            text-transform: uppercase;
        }}
        select, input {{
            width: 100%;
            padding: 9px 10px;
            border-radius: 8px;
            border: 1px solid #dadce0;
            font-size: 14px;
            margin-top: 5px;
            outline: none;
        }}
        button {{
            width: 100%;
            padding: 10px;
            border-radius: 8px;
            border: none;
            background: #1a73e8;
            color: white;
            font-weight: bold;
            font-size: 14px;
            cursor: pointer;
            margin-top: 20px;
            transition: background .2s;
        }}
        button:hover {{ background: #174ea6; }}
        canvas {{
            background: #fff;
            border: 1px solid #e8eaed;
            border-radius: 10px;
            margin-top: 15px;
            cursor: crosshair;
            box-shadow: inset 0 0 10px rgba(0,0,0,.02);
        }}
        #status-msg {{
            margin-top: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 5px solid #1a73e8;
            font-size: 14px;
            line-height: 1.6;
            color: #3c4043;
            min-height: 50px;
            width: 100%;
        }}
        .legend {{
            display: flex;
            gap: 15px;
            margin-bottom: 10px;
            font-size: 12px;
            font-weight: 500;
            color: #70757a;
            justify-content: center;
            width: 100%;
            flex-wrap: wrap;
        }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; }}
        .dot {{ width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }}
        .search-box {{
            background: #fff8e1;
            padding: 15px;
            border-radius: 8px;
            border: 1px dashed #ffb300;
            margin-bottom: 15px;
        }}
        .status-tag {{
            font-size: 11px;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: bold;
            display: inline-block;
            margin-top: 5px;
        }}
        .open-tag   {{ background:#e6fffa; color:#00875a; border:1px solid #00875a; }}
        .closed-tag {{ background:#fff5f5; color:#e53e3e; border:1px solid #e53e3e; }}
        .method-tag {{
            font-size: 10px;
            background: #e8f0fe;
            color: #1a73e8;
            padding: 2px 5px;
            border-radius: 3px;
            display: block;
            margin-top: 5px;
            font-weight: normal;
        }}
        #clock {{ text-align:center; font-weight:bold; color:#e53e3e; margin-bottom:15px; font-size:18px; }}
        
        /* CSS CHO BẢNG INFO PANEL & SUGGESTIONS CHỦ ĐỘNG */
        .info-panel {{
            margin-top: 15px;
            padding: 15px;
            background: #e8f0fe;
            border-radius: 10px;
            border-left: 5px solid #1a73e8;
            display: none;
        }}
        .info-item {{ font-size: 13px; margin-bottom: 5px; color: #3c4043; }}
        .info-services {{
            margin-top: 8px;
            padding: 10px;
            background: #fff;
            border-radius: 6px;
            font-size: 12px;
            line-height: 1.5;
        }}
        .service-chip {{
            display: inline-block;
            background: #e8f0fe;
            color: #1a73e8;
            padding: 3px 8px;
            border-radius: 12px;
            margin: 2px 4px 2px 0;
            font-size: 11px;
        }}
        .suggestion-functions {{
            font-size: 11px;
            color: #1a73e8;
            margin-top: 6px;
            line-height: 1.4;
        }}
        
        #proactive-suggestions {{
            margin-top: 15px;
            display: none;
            flex-direction: column;
            gap: 10px;
            width: 100%;
        }}
        .suggestion-card {{
            background: #e8f0fe;
            border: 1px solid #8ab4f8;
            padding: 12px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .suggestion-card:hover {{
            background: #d2e3fc;
            transform: translateY(-2px);
        }}
        .suggestion-title {{
            font-weight: bold;
            color: #1a73e8;
            font-size: 13px;
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .suggestion-desc {{
            font-size: 12px;
            color: #5f6368;
            margin-top: 5px;
            line-height: 1.4;
        }}
        .suggestion-meta {{
            font-size: 11px;
            color: #80868b;
            margin-top: 4px;
        }}

        /* CSS CHO TOAST (GEOFENCING ALERTS) */
        .toast-container {{
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .toast {{
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: bold;
            font-size: 13px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            animation: slideIn 0.3s ease-out;
            max-width: 300px;
            line-height: 1.4;
        }}
        .toast-danger {{ background: #ea4335; border-left: 5px solid #b31404; }}
        .toast-success {{ background: #34a853; border-left: 5px solid #188038; }}
        .toast-info {{ background: #4285f4; border-left: 5px solid #1a73e8; }}
        @keyframes slideIn {{
            from {{ transform: translateX(100%); opacity: 0; }}
            to {{ transform: translateX(0); opacity: 1; }}
        }}
    </style>
</head>
<body>
    <div class="toast-container" id="toast-box"></div>

    <div class="sidebar">
        <h2>🌍 AI AR Navigator</h2>
        <div id="clock">🕒 Đang tải giờ...</div>

        <div style="background:#fff3e0; padding:10px; border-radius:8px; border:1px dashed #ff9800; margin-bottom: 15px;">
            <label style="color:#e65100; margin-top:0;">🚧 Giả lập chặn đường:</label>
            <div style="display:flex; gap:10px; margin-top:5px;">
                <button style="background:#e65100; margin-top:0;" onclick="toggleRoad('Tòa A', 'Nhà xe', 'repairing')">Chặn A - Nhà xe</button>
                <button style="background:#2ecc71; margin-top:0;" onclick="toggleRoad('Tòa A', 'Nhà xe', 'open')">Mở lại</button>
            </div>
            <div style="font-size:11px; color:#555; margin-top:5px;">Click thử chặn rồi tìm lộ trình để xem thuật toán tự bẻ lái!</div>
        </div>

        <div class="search-box">
            <label style="color:#f57c00; margin-top:0;">🧠 Hỏi AI Gợi ý:</label>
            <input type="text" id="ai-search" placeholder="VD: tìm chỗ mát mẻ yên tĩnh học bài">
            <button onclick="semanticSearch()"
                    style="background:#ffb300; margin-top:8px; color:#333;">
                TÌM ĐỊA ĐIỂM TỐI ƯU
            </button>
            <div id="search-msg"
                 style="font-size:13px; color:#e65100; margin-top:8px; font-weight:bold;"></div>
        </div>

        <label>📍 Điểm xuất phát:</label>
        <select id="start">{opts}</select>

        <label>🛑 Điểm dừng trung gian (tuỳ chọn):</label>
        <select id="stop">
            <option value="">-- Không ghé điểm nào --</option>
            {opts}
        </select>

        <label>🎯 Đích đến:</label>
        <select id="end">{opts}</select>

        <label>🌤️ Thời tiết:</label>
        <select id="weather">
            <option value="normal">Mát mẻ bình thường</option>
            <option value="sunny">Nắng gắt / Mưa lớn</option>
        </select>

        <button onclick="calculateRoute()">🗺️ TÌM ĐƯỜNG ĐI</button>

        <div id="info-panel" class="info-panel">
            <b id="info-title" style="color: #1a73e8; font-size: 14px; display: block; margin-bottom: 8px;">Tên tòa nhà</b>
            <div class="info-item" id="info-tagline" style="font-style:italic;color:#5f6368;"></div>
            <div class="info-item" id="info-hours">🕒 Giờ: --</div>
            <div class="info-item" id="info-features">✨ Đặc điểm: --</div>
            <div class="info-services" id="info-services"></div>
        </div>

        <div style="margin-top:12px;">
            <label style="margin-top:0;color:#1a73e8;">📋 Chức năng các tòa</label>
            <button type="button" onclick="loadBuildingGuide()"
                    style="background:#e8f0fe;color:#1a73e8;margin-top:6px;">
                XEM DANH SÁCH DỊCH VỤ
            </button>
            <div id="building-guide" style="max-height:180px;overflow-y:auto;margin-top:8px;font-size:12px;"></div>
        </div>

        <div id="status-msg">Click lên bản đồ để giả lập vị trí GPS của bạn.</div>
        
        <div id="gps-data-panel"></div>
        
        <div id="proactive-suggestions"></div>
    </div>

    <div class="main-content">
        <div class="legend">
            <div class="legend-item">
                <span class="dot" style="background:#85C1E9"></span> Đang mở cửa
            </div>
            <div class="legend-item">
                <span class="dot" style="background:#fab1a0"></span> Đã đóng cửa
            </div>
            <div class="legend-item">
                <span class="dot" style="background:#2ecc71"></span> Trên lộ trình
            </div>
            <div class="legend-item">
                <span style="border-top:3px solid #888; width:20px; display:inline-block;"></span>
                Có mái che
            </div>
            <div class="legend-item">
                <span style="border-top:2px dashed #E74C3C; width:20px; display:inline-block;"></span>
                Đang sửa
            </div>
            <div class="legend-item">
                <span class="dot" style="background:#3498db; border:2px solid white;"></span>
                Vị trí bạn
            </div>
        </div>
        <canvas id="mapCanvas" width="850" height="500"></canvas>
    </div>

    <script>
        // Bounds GPS thực từ server — dùng để scale canvas động
        const BOUNDS = {{
            minX: {b['min_x']}, maxX: {b['max_x']},
            minY: {b['min_y']}, maxY: {b['max_y']}
        }};
        const MARGIN = 70;

        let graphData = null;
        const canvas  = document.getElementById('mapCanvas');
        const ctx     = canvas.getContext('2d');
        let alertedNodes = new Set(); // Tránh spam toast cho cùng 1 địa điểm

        // --- Khởi tạo ---
        window.onload = async () => {{
            const res  = await fetch('/api_get_graph');
            graphData  = await res.json();
            document.getElementById('clock').innerText =
                "🕒 Giờ hệ thống: " + graphData.current_time;
            drawGraph([]);
        }};

        // --- Chuyển tọa độ GPS -> pixel canvas (scale động) ---
        function toCanvas(lon, lat) {{
            const W = canvas.width  - MARGIN * 2;
            const H = canvas.height - MARGIN * 2;
            const rangeX = BOUNDS.maxX - BOUNDS.minX || 1;
            const rangeY = BOUNDS.maxY - BOUNDS.minY || 1;
            const cx = MARGIN + ((lon - BOUNDS.minX) / rangeX) * W;
            const cy = canvas.height - MARGIN - ((lat - BOUNDS.minY) / rangeY) * H;
            return {{ cx, cy }};
        }}

        // --- Hàm hiển thị Toast Notifications ---
        function showToast(msg, level) {{
            const box = document.getElementById('toast-box');
            const toast = document.createElement('div');
            toast.className = `toast toast-${{level}}`;
            toast.innerHTML = msg;
            box.appendChild(toast);
            setTimeout(() => toast.remove(), 5000);
        }}

        // --- Hàm thay đổi trạng thái đường ---
        async function toggleRoad(u, v, status) {{
            try {{
                await fetch(`/api_update_edge?u=${{u}}&v=${{v}}&status=${{status}}`);
                const res = await fetch('/api_get_graph');
                graphData = await res.json();
                drawGraph([]);
                showToast(`Đã thay đổi tuyến ${{u}}-${{v}} thành: ${{status}}`, status === 'open' ? 'success' : 'danger');
                calculateRoute(); 
            }} catch (e) {{
                console.error(e);
            }}
        }}

        // --- Hàm hiển thị Info Panel ---
        function showInfo(name, data, profile) {{
            const panel = document.getElementById('info-panel');
            panel.style.display = 'block';
            document.getElementById('info-title').innerText = "📍 " + name;
            const taglineEl = document.getElementById('info-tagline');
            const tagline = (profile && profile.tagline) || data.tagline || '';
            taglineEl.innerText = tagline ? tagline : '';
            taglineEl.style.display = tagline ? 'block' : 'none';

            const hours = data.hours || (data.open_time + " - " + data.close_time);
            document.getElementById('info-hours').innerText = "🕒 Mở cửa: " + hours;

            const f = data.features || {{}};
            const ac = f.has_ac ? 'Có' : 'Không';
            const noise = f.noise_level !== undefined ? f.noise_level : '--';
            const cap = f.capacity !== undefined ? f.capacity : '--';
            document.getElementById('info-features').innerText = `❄️ AC: ${{ac}} | 🔇 Ồn: ${{noise}} | 👥 Sức chứa: ${{cap}}`;

            const svcBox = document.getElementById('info-services');
            const services = (profile && profile.services) || data.services || [];
            if (services.length) {{
                svcBox.innerHTML = '<b>🏢 Chức năng tại đây:</b><br>' +
                    services.map(s => `<span class="service-chip">${{s.icon || ''}} ${{s.name}}</span>`).join('');
            }} else if (profile && profile.function_summary) {{
                svcBox.innerHTML = '<b>🏢 Chức năng:</b> ' + profile.function_summary;
            }} else {{
                svcBox.innerHTML = '';
            }}
        }}

        async function loadBuildingGuide() {{
            const box = document.getElementById('building-guide');
            box.innerHTML = '⏳ Đang tải...';
            try {{
                const res = await fetch('/api_building_guide');
                const data = await res.json();
                if (!data.buildings) {{ box.innerHTML = 'Không có dữ liệu.'; return; }}
                box.innerHTML = data.buildings.map(b => `
                    <div style="margin-bottom:10px;padding:8px;background:#f8f9fa;border-radius:6px;cursor:pointer;"
                         onclick="document.getElementById('end').value='${{b.node}}';showInfo('${{b.node}}', {{}}, b);">
                        <b>${{b.node}}</b> — <span style="color:#5f6368">${{b.tagline || ''}}</span><br>
                        <span style="color:#1a73e8">${{b.function_summary || ''}}</span>
                    </div>
                `).join('');
            }} catch (e) {{
                box.innerHTML = '❌ Lỗi tải danh sách.';
            }}
        }}

        // --- Hàm vẽ bảng tọa độ GPS ---
        function renderGPSTable(coordsArray) {{
            const panel = document.getElementById('gps-data-panel');
            if (!coordsArray || coordsArray.length === 0) {{
                panel.innerHTML = '';
                return;
            }}
            
            let tableHTML = `
                <table class="gps-table">
                    <thead>
                        <tr>
                            <th>Trạm</th>
                            <th>Tọa độ (Lat, Lon)</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            coordsArray.forEach(point => {{
                tableHTML += `
                    <tr>
                        <td><b>${{point.node}}</b></td>
                        <td>${{point.gps[0].toFixed(5)}}, ${{point.gps[1].toFixed(5)}}</td>
                    </tr>
                `;
            }});
            
            tableHTML += `</tbody></table>`;
            panel.innerHTML = tableHTML;
        }}

        // --- Vẽ toàn bộ đồ thị ---
        function drawGraph(pathArray, userPos = null) {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (!graphData) return;

            const nodeMap = Object.fromEntries(graphData.nodes.map(n => [n.id, n]));
            const pathSet = new Set();
            for (let i = 0; i < pathArray.length - 1; i++) {{
                pathSet.add(pathArray[i] + '|' + pathArray[i + 1]);
                pathSet.add(pathArray[i + 1] + '|' + pathArray[i]);
            }}

            // Vẽ cạnh
            graphData.edges.forEach(edge => {{
                const n1 = nodeMap[edge.source];
                const n2 = nodeMap[edge.target];
                if (!n1 || !n2) return;
                const p1 = toCanvas(n1.x, n1.y);
                const p2 = toCanvas(n2.x, n2.y);
                const inPath = pathSet.has(edge.source + '|' + edge.target);

                ctx.beginPath();
                ctx.moveTo(p1.cx, p1.cy);
                ctx.lineTo(p2.cx, p2.cy);

                if (inPath) {{
                    ctx.strokeStyle = '#2ecc71';
                    ctx.lineWidth   = 6;
                    ctx.setLineDash([]);
                }} else if (edge.status === 'repairing') {{
                    ctx.strokeStyle = '#e74c3c';
                    ctx.lineWidth   = 3;
                    ctx.setLineDash([8, 5]);
                }} else {{
                    ctx.strokeStyle = edge.has_roof ? '#b2c0d8' : '#dfe6e9';
                    ctx.lineWidth   = edge.has_roof ? 3 : 2;
                    ctx.setLineDash([]);
                }}
                ctx.stroke();
            }});

            // Vẽ node
            ctx.setLineDash([]);
            graphData.nodes.forEach(node => {{
                const p      = toCanvas(node.x, node.y);
                const active = pathArray.includes(node.id);
                
                // Highlight khu vực bị hạn chế (admin)
                let fillColor = active ? '#2ecc71' : (node.type === 'admin' ? '#f5b041' : (node.is_open ? '#85C1E9' : '#fab1a0'));

                // Vòng tròn chính
                ctx.beginPath();
                ctx.arc(p.cx, p.cy, 18, 0, 2 * Math.PI);
                ctx.fillStyle   = fillColor;
                ctx.fill();
                ctx.strokeStyle = active ? '#27ae60' : '#b2bec3';
                ctx.lineWidth   = active ? 4 : 1.5;
                ctx.stroke();

                // Chấm nhỏ mở/đóng
                ctx.beginPath();
                ctx.arc(p.cx + 13, p.cy - 13, 5, 0, 2 * Math.PI);
                ctx.fillStyle = node.is_open ? '#00b894' : '#d63031';
                ctx.fill();

                // Nhãn
                ctx.fillStyle  = '#2d3436';
                ctx.font       = active ? 'bold 12px Arial' : '11px Arial';
                ctx.textAlign  = 'center';
                ctx.fillText(node.id, p.cx, p.cy - 26);
            }});

            if (userPos) {{
                ctx.beginPath();
                ctx.arc(userPos.x, userPos.y, 10, 0, 2 * Math.PI);
                ctx.fillStyle   = '#3498db';
                ctx.fill();
                ctx.strokeStyle = 'white';
                ctx.lineWidth   = 3;
                ctx.stroke();
            }}
        }}

        // --- AI Semantic Search ---
        async function semanticSearch() {{
            const query = document.getElementById('ai-search').value.trim();
            const wthr  = document.getElementById('weather').value;
            const msg   = document.getElementById('search-msg');
            if (!query) return;

            msg.innerHTML = '⏳ Đang suy luận yêu cầu...';
            try {{
                const res  = await fetch(`/api_search?query=${{encodeURIComponent(query)}}&weather=${{wthr}}`);
                const data = await res.json();
                if (data.status === 'success') {{
                    const tag    = data.is_open
                        ? '<span class="status-tag open-tag">ĐANG MỞ CỬA</span>'
                        : '<span class="status-tag closed-tag">ĐÃ ĐÓNG CỬA</span>';
                    const method = '<span class="method-tag">Dựa trên: ' + data.method + '</span>';
                    msg.innerHTML = 'Đề xuất: <b>' + data.matched_node + '</b><br>' + tag + method;
                    document.getElementById('end').value = data.matched_node;
                    
                    const fnSum = data.function_summary || (data.recommendations && data.recommendations[0] && data.recommendations[0].function_summary);
                    if (fnSum) msg.innerHTML += '<br><span style="color:#1a73e8">🏢 ' + fnSum + '</span>';
                    if (data.info) showInfo(data.matched_node, data.info, data.recommendations ? data.recommendations[0] : null);
                    if (data.recommendations && data.recommendations.length > 1) {{
                        renderSuggestions(data.recommendations.map(r => ({{
                            node: r.node,
                            reason: r.reason || 'Khớp nhu cầu tìm kiếm',
                            score: r.score,
                            distance_m: null,
                            on_route: false
                        }})));
                    }}
                }} else {{
                    msg.innerHTML = '❌ ' + data.message;
                }}
            }} catch (e) {{
                msg.innerHTML = '❌ Lỗi kết nối server.';
            }}
        }}
        
        // --- Tính lộ trình ---
        async function calculateRoute() {{
            const start = document.getElementById('start').value;
            const stop  = document.getElementById('stop').value;
            const end   = document.getElementById('end').value;
            const wthr  = document.getElementById('weather').value;

            const pts = [start];
            if (stop) pts.push(stop);
            pts.push(end);

            const statusDiv = document.getElementById('status-msg');
            statusDiv.innerHTML = '⏳ Đang tìm đường...';

            try {{
                const res  = await fetch('/api_get_route?waypoints=' + pts.join(',') + '&weather=' + wthr);
                const data = await res.json();

                if (res.ok) {{
                    const tag = data.all_open
                        ? '<span class="status-tag open-tag">CÁC ĐIỂM ĐỀU MỞ CỬA</span>'
                        : '<span class="status-tag closed-tag">⚠️ CÓ ĐIỂM ĐÃ ĐÓNG CỬA!</span>';
                    statusDiv.innerHTML = '<b>Lộ trình:</b> ' + data.path.join(' ➔ ') + '<br>' + tag;
                    drawGraph(data.path);
                    
                    // GỌI HÀM RENDER BẢNG GPS
                    if (data.coordinates) renderGPSTable(data.coordinates);
                    
                }} else {{
                    statusDiv.innerHTML = '<span style="color:red">❌ ' + (data.detail || data.message) + '</span>';
                    renderGPSTable([]); // Xóa bảng nếu lỗi
                }}
            }} catch (e) {{
                statusDiv.innerHTML = '<span style="color:red">❌ Lỗi kết nối server.</span>';
                renderGPSTable([]);
            }}
        }}

        function renderSuggestions(suggestions) {{
            const suggDiv = document.getElementById('proactive-suggestions');
            suggDiv.innerHTML = '';
            if (!suggestions || suggestions.length === 0) {{
                suggDiv.style.display = 'none';
                return;
            }}
            suggDiv.style.display = 'flex';
            suggestions.forEach(s => {{
                const card = document.createElement('div');
                card.className = 'suggestion-card';
                const meta = [];
                if (s.score != null) meta.push('Điểm AI: ' + s.score + '/100');
                if (s.distance_m != null) meta.push('~' + Math.round(s.distance_m) + 'm');
                if (s.on_route) meta.push('Trên đường tới đích');
                const funcLine = s.function_summary
                    ? '<div class="suggestion-functions">🏢 ' + s.function_summary + '</div>'
                    : '';
                const svcLine = (s.matched_services && s.matched_services.length)
                    ? '<div class="suggestion-functions">✓ ' + s.matched_services.map(x => x.name).join(', ') + '</div>'
                    : '';
                card.innerHTML =
                    '<div class="suggestion-title">✨ Gợi ý: ' + s.node + '</div>' +
                    '<div class="suggestion-desc">' + (s.reason || '') + '</div>' +
                    funcLine + svcLine +
                    (meta.length ? '<div class="suggestion-meta">' + meta.join(' · ') + '</div>' : '');
                card.onclick = () => {{
                    document.getElementById('end').value = s.node;
                    calculateRoute();
                    suggDiv.style.display = 'none';
                }};
                suggDiv.appendChild(card);
            }});
        }}

        // --- Giả lập Tracking GPS bằng click chuột ---
        canvas.addEventListener('mousedown', async (e) => {{
            const rect   = canvas.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;

            const W      = canvas.width  - MARGIN * 2;
            const H      = canvas.height - MARGIN * 2;
            const lon    = BOUNDS.minX + ((clickX - MARGIN) / W) * (BOUNDS.maxX - BOUNDS.minX);
            const lat    = BOUNDS.minY + ((canvas.height - MARGIN - clickY) / H) * (BOUNDS.maxY - BOUNDS.minY);
            const end    = document.getElementById('end').value;
            const wthr   = document.getElementById('weather').value;

            try {{
                const res  = await fetch(
                    `/api_realtime_tracking?current_lat=${{lat}}&current_lon=${{lon}}&end=${{end}}&weather=${{wthr}}`
                );
                const data = await res.json();
                const statusDiv = document.getElementById('status-msg');

                if (data.status === 'tracking') {{
                    const tag = data.dest_open
                        ? '<span class="status-tag open-tag">ĐÍCH ĐANG MỞ CỬA</span>'
                        : '<span class="status-tag closed-tag">ĐÍCH ĐÃ ĐÓNG CỬA</span>';
                    statusDiv.innerHTML =
                        `📍 Gần nhất: <b>${{data.snapped_node}}</b><br>` +
                        `📏 Còn lại: <b>${{data.total_remaining_meters}}m</b><br>` +
                        `⏱️ Ước tính: ${{data.estimated_mins}} phút<br>${{tag}}`;
                    drawGraph(data.path, {{ x: clickX, y: clickY }});
                    
                    if(data.node_info) showInfo(data.snapped_node, data.node_info);
                    
                    // RENDER BẢNG TỌA ĐỘ THEO LỘ TRÌNH CÒN LẠI
                    if(data.path_coords) renderGPSTable(data.path_coords);

                    // XỬ LÝ HIỂN THỊ GEOFENCING ALERTS
                    if (data.geofence_alerts) {{
                        data.geofence_alerts.forEach(alert => {{
                            if (!alertedNodes.has(alert.node)) {{
                                showToast(alert.msg, alert.level);
                                alertedNodes.add(alert.node); // Đánh dấu đã cảnh báo
                            }}
                        }});
                    }}
                    
                }} else if (data.status === 'arrived') {{
                    statusDiv.innerHTML = '🎉 Bạn đã đến nơi!';
                    drawGraph([end], {{ x: clickX, y: clickY }});
                    renderGPSTable([]); // Xóa bảng khi đã đến nơi
                }} else {{
                    statusDiv.innerHTML = '<span style="color:red">❌ ' + (data.detail || 'Lỗi.') + '</span>';
                }}
                
                // ---- GỢI Ý AI: theo GPS + đích + câu hỏi ----
                try {{
                    const aiQuery = document.getElementById('ai-search').value.trim();
                    const wthr = document.getElementById('weather').value;
                    let pUrl = `/api_proactive_recommend?current_lat=${{lat}}&current_lon=${{lon}}&destination=${{encodeURIComponent(end)}}&weather=${{wthr}}&limit=6`;
                    if (aiQuery) pUrl += `&query=${{encodeURIComponent(aiQuery)}}`;
                    const pRes = await fetch(pUrl);
                    const pData = await pRes.json();
                    let merged = (pData.suggestions || []);
                    if (data.route_suggestions && data.route_suggestions.length) {{
                        const seen = new Set(merged.map(x => x.node));
                        data.route_suggestions.forEach(rs => {{
                            if (!seen.has(rs.node)) merged.push(rs);
                        }});
                    }}
                    renderSuggestions(merged);
                }} catch (err) {{
                    console.error("Lỗi khi tải gợi ý chủ động:", err);
                }}

            }} catch (e) {{
                document.getElementById('status-msg').innerHTML =
                    '<span style="color:red">❌ Lỗi kết nối server.</span>';
            }}
        }});
    </script>
</body>
</html>"""
    return html