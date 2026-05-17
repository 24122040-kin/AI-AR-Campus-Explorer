from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from config.settings import settings
from core.database import db
from core.traffic_analyzer import traffic_analyzer


router = APIRouter(tags=["map-data"])


@router.get("/api/map", response_class=HTMLResponse)
async def get_map(
    lat: float = settings.map_default_lat,
    lon: float = settings.map_default_lon,
    zoom: int = settings.map_default_zoom,
):
    import folium
    from folium.plugins import HeatMap, MarkerCluster, MeasureControl

    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles="OpenStreetMap")
    MeasureControl(primary_length_unit="meters").add_to(m)
    cluster = MarkerCluster(name="Dia diem").add_to(m)
    for loc in await db.fetchall("SELECT * FROM locations LIMIT 1000"):
        imgs = await db.get_images_for_location(loc["id"])
        img_html = (
            f'<br><img src="/api/image/{imgs[0]["id"]}" width="200" style="border-radius:6px"/>'
            if imgs
            else ""
        )
        color = {1: "blue", 2: "blue", 3: "orange", 4: "red", 5: "darkred"}.get(loc.get("importance", 1), "blue")
        folium.Marker(
            [loc["lat"], loc["lon"]],
            popup=folium.Popup(f"<b>{loc['name']}</b><br>{loc.get('description','')}{img_html}", max_width=240),
            tooltip=loc["name"],
            icon=folium.Icon(color=color, icon="camera", prefix="fa"),
        ).add_to(cluster)
    poi_layer = folium.FeatureGroup(name="POI local").add_to(m)
    for poi in await db.fetchall("SELECT * FROM pois WHERE is_active=1 LIMIT 500"):
        folium.CircleMarker(
            [poi["lat"], poi["lon"]],
            radius=7,
            color="#e74c3c",
            fill=True,
            fill_color="#e74c3c",
            fill_opacity=0.8,
            popup=f"{poi['name']} ({poi['type']})",
        ).add_to(poi_layer)
    edge_layer = folium.FeatureGroup(name="Duong tat/hem").add_to(m)
    for e in await db.get_all_custom_edges():
        folium.PolyLine(
            [(e["from_lat"], e["from_lon"]), (e["to_lat"], e["to_lon"])],
            color="#27ae60",
            weight=4,
            opacity=0.85,
        ).add_to(edge_layer)
    heat = traffic_analyzer.heatmap_data()
    if heat:
        HeatMap(
            [[d["lat"], d["lon"], d["intensity"]] for d in heat],
            name="Tac nghe",
            radius=22,
            blur=18,
            gradient={"0.0": "blue", "0.4": "lime", "0.7": "orange", "1.0": "red"},
        ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()


@router.get("/api/image/{image_id}")
async def get_image(image_id: int):
    row = await db.fetchone("SELECT filepath FROM images WHERE id=?", (image_id,))
    if not row:
        raise HTTPException(404)
    p = Path(row["filepath"])
    if not p.exists():
        raise HTTPException(404, "File missing")
    return FileResponse(str(p), media_type=mimetypes.guess_type(str(p))[0] or "image/jpeg")


@router.get("/api/campus/boundary")
async def campus_boundary():
    """Return campus polygon and bbox for frontend map rendering."""
    from core.campus_scope import campus_bbox
    return campus_bbox()


@router.get("/api/locations/all")
async def all_locations():
    """Return all locations with floor info — used by the local map."""
    rows = await db.fetchall(
        "SELECT id, name, lat, lon, floor, category, importance, description FROM locations ORDER BY floor, importance DESC"
    )
    return {"locations": rows}


@router.get("/api/edges/all")
async def all_edges():
    """Return all custom edges with geometry — used by the local map."""
    rows = await db.fetchall("SELECT * FROM custom_edges ORDER BY id")
    import json as _json
    result = []
    for r in rows:
        e = dict(r)
        if e.get("geometry"):
            try:
                e["geometry"] = _json.loads(e["geometry"])
            except Exception:
                e["geometry"] = None
        result.append(e)
    return {"edges": result}


@router.get("/api/localmap", response_class=HTMLResponse)
async def local_map(mode: str = ""):
    """Interactive Leaflet map - simplified version with external JS"""
    locations = await db.fetchall(
        "SELECT id, name, lat, lon, floor, category, importance, description FROM locations ORDER BY floor"
    )
    edges = await db.fetchall("SELECT * FROM custom_edges")

    import json as _json

    locs_json = _json.dumps(locations)
    edges_json = _json.dumps([
        {**dict(e), "geometry": _json.loads(e["geometry"]) if e.get("geometry") else None}
        for e in edges
    ])

    floor_colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#f97316", "#ec4899", "#84cc16", "#6b7280"]
    floor_colors_json = _json.dumps(floor_colors)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Local Map — ĐHKHTN CS2</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; height: 100vh; display: flex; flex-direction: column; }}
  #toolbar {{ padding: 8px 12px; background: #1e293b; border-bottom: 1px solid #334155; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
  #toolbar h2 {{ font-size: 14px; color: #94a3b8; margin-right: 4px; }}
  .tbtn {{ padding: 6px 12px; border-radius: 6px; border: 1px solid #475569; background: #334155; color: #e2e8f0; cursor: pointer; font-size: 12px; transition: .15s; }}
  .tbtn:hover {{ background: #475569; }}
  .tbtn.active {{ background: #3b82f6; border-color: #3b82f6; color: #fff; }}
  #floor-filter {{ display: flex; gap: 4px; align-items: center; font-size: 12px; }}
  #floor-filter label {{ color: #94a3b8; }}
  #floor-sel {{ background: #334155; border: 1px solid #475569; color: #e2e8f0; border-radius: 4px; padding: 4px 8px; font-size: 12px; }}
  #map {{ flex: 1; }}
  #status {{ padding: 6px 12px; background: #1e293b; font-size: 11px; color: #64748b; border-top: 1px solid #334155; }}
  .panel-strip {{ display: none; padding: 12px; background: #1e293b; border-top: 1px solid #334155; flex-direction: column; gap: 12px; }}
  .panel-strip.show {{ display: flex; }}
  .panel-strip > div {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
  .panel-strip input, .panel-strip select {{ background: #334155; border: 1px solid #475569; color: #e2e8f0; border-radius: 4px; padding: 6px 8px; font-size: 12px; }}
  .legend {{ background: #1e293b; padding: 8px; border-radius: 6px; border: 1px solid #334155; font-size: 11px; line-height: 1.8; }}
  .legend-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }}
  .loc-marker-badge {{
    display: flex; align-items: center; justify-content: center;
    border-radius: 50%; color: #fff; font-weight: 700;
    border: 2px solid #fff; box-shadow: 0 2px 6px rgba(0,0,0,.45);
    cursor: pointer; line-height: 1; text-align: center;
    min-width: 26px; min-height: 26px; padding: 0 3px;
  }}
</style>
</head>
<body>

<div id="toolbar">
  <h2>🗺 ĐHKHTN CS2</h2>
  <button class="tbtn active" id="btn-view" onclick="setMode('view')">👁 Xem</button>
  <button class="tbtn" id="btn-add-road" onclick="setMode('add-road')">➕ Tạo đường</button>
  <div id="floor-filter">
    <label>Tầng:</label>
    <select id="floor-sel" onchange="filterFloor(this.value)">
      <option value="all">Tất cả</option>
    </select>
  </div>
  <button class="tbtn" onclick="closeMap()">✕ Đóng</button>
</div>

<!-- Panel tạo đường -->
<div id="road-panel" class="panel-strip">
  <!-- Bước 1: Chọn 2 điểm -->
  <div id="road-step-1" style="display:none;flex-direction:column;gap:8px">
    <div style="display:flex;align-items:center;gap:8px">
      <span style="font-size:13px;color:#e2e8f0;font-weight:600">Bước 1: Chọn 2 điểm</span>
      <button class="tbtn" onclick="cancelRoad()" style="margin-left:auto">✕ Huỷ</button>
    </div>
    <div style="padding:8px;background:#334155;border-radius:6px">
      <div style="font-size:12px;color:#94a3b8;margin-bottom:4px">📍 Điểm A (xuất phát):</div>
      <div id="point-a-display" style="font-size:13px;color:#22c55e;font-weight:600">Chưa chọn - click marker trên bản đồ</div>
    </div>
    <div style="padding:8px;background:#334155;border-radius:6px">
      <div style="font-size:12px;color:#94a3b8;margin-bottom:4px">📍 Điểm B (đích):</div>
      <div id="point-b-display" style="font-size:13px;color:#ef4444;font-weight:600">Chưa chọn - click marker trên bản đồ</div>
    </div>
    <span id="road-hint" style="font-size:11px;color:#64748b;text-align:center">Click vào marker trên bản đồ để chọn điểm A...</span>
  </div>
  
  <!-- Bước 2: Chọn cách tạo đường -->
  <div id="road-step-2" style="display:none;flex-direction:column;gap:8px">
    <span style="font-size:13px;color:#e2e8f0;font-weight:600">Bước 2: Chọn cách tạo đường</span>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px">
      <button class="tbtn active" onclick="selectRoadMethod('straight')" style="padding:12px;font-size:13px">
        📏 Nối thẳng
        <div style="font-size:10px;color:#94a3b8;margin-top:4px">Hành lang, đường ngang</div>
      </button>
      <button class="tbtn active" onclick="selectRoadMethod('stairs')" style="padding:12px;font-size:13px">
        🪜 Cầu thang
        <div style="font-size:10px;color:#94a3b8;margin-top:4px">Nối 2 tầng khác nhau</div>
      </button>
      <button class="tbtn active" onclick="selectRoadMethod('tracking')" style="padding:12px;font-size:13px">
        🚶 Tracking
        <div style="font-size:10px;color:#94a3b8;margin-top:4px">Đường ngoài trời</div>
      </button>
    </div>
    <div style="padding:8px;background:#334155;border-radius:6px;font-size:11px;color:#94a3b8">
      💡 <b>Lưu ý:</b> GPS không hoạt động trong nhà. Dùng "Nối thẳng" hoặc "Cầu thang" cho đường trong nhà.
    </div>
    <button class="tbtn" onclick="cancelRoad()">✕ Huỷ</button>
  </div>
  
  <!-- Bước 3: Tracking -->
  <div id="road-step-3" style="display:none;flex-direction:column;gap:8px">
    <span style="font-size:13px;color:#e2e8f0;font-weight:600">Bước 3: Tracking đi bộ</span>
    <div style="padding:8px;background:#334155;border-radius:6px;text-align:center">
      <div id="walk-status" style="font-size:14px;color:#10b981;font-weight:600;margin-bottom:4px">Chưa bắt đầu</div>
      <div style="font-size:11px;color:#94a3b8">Bấm "Bắt đầu đi" rồi đi bộ theo đường cần tạo</div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
      <button class="tbtn active" id="btn-start-walk" onclick="startWalkTracking()" style="padding:12px;font-size:14px">
        ▶️ Bắt đầu đi
      </button>
      <button class="tbtn active" id="btn-stop-walk" onclick="stopWalkTracking()" style="display:none;padding:12px;font-size:14px;background:#ef4444;border-color:#ef4444">
        ⏹ Kết thúc
      </button>
    </div>
    <button class="tbtn" onclick="resetWalkTracking()">🔄 Reset (xóa tracking)</button>
    <button class="tbtn" onclick="cancelRoad()">✕ Huỷ tạo đường</button>
  </div>
  
  <!-- Bước 4: Form thông tin đường -->
  <div id="road-step-4" style="display:none;flex-direction:column;gap:8px">
    <span style="font-size:13px;color:#e2e8f0;font-weight:600">Bước 4: Thông tin đường</span>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px">
      <div>
        <label style="font-size:10px;color:#94a3b8;display:block;margin-bottom:2px">Tên đường *</label>
        <input id="road-name" placeholder="Hành lang A..." style="width:100%"/>
      </div>
      <div>
        <label style="font-size:10px;color:#94a3b8;display:block;margin-bottom:2px">Loại đường</label>
        <select id="road-type" style="width:100%">
          <option value="corridor">Hành lang</option>
          <option value="alley">Hẻm</option>
          <option value="path">Đường mòn</option>
          <option value="shortcut">Đường tắt</option>
          <option value="stairs">Cầu thang</option>
          <option value="elevator">Thang máy</option>
        </select>
      </div>
      <div>
        <label style="font-size:10px;color:#94a3b8;display:block;margin-bottom:2px">Bề mặt</label>
        <select id="road-surface" style="width:100%">
          <option value="tile">Gạch lát</option>
          <option value="concrete">Bê tông</option>
          <option value="asphalt">Nhựa đường</option>
          <option value="grass">Cỏ/Đất</option>
          <option value="gravel">Sỏi</option>
        </select>
      </div>
      <div>
        <label style="font-size:10px;color:#94a3b8;display:block;margin-bottom:2px">Độ dốc (°)</label>
        <input id="road-slope" type="number" value="0" min="-45" max="45" style="width:100%"/>
      </div>
      <div>
        <label style="font-size:10px;color:#94a3b8;display:block;margin-bottom:2px">Chiều</label>
        <select id="road-direction" style="width:100%">
          <option value="both">2 chiều</option>
          <option value="oneway">1 chiều (A→B)</option>
        </select>
      </div>
      <div style="display:flex;flex-direction:column;gap:4px;justify-content:center">
        <label style="font-size:11px;color:#94a3b8;display:flex;align-items:center;gap:4px;cursor:pointer">
          <input type="checkbox" id="road-covered" checked style="width:14px;height:14px"/>
          <span>☂️ Có mái che</span>
        </label>
        <label style="font-size:11px;color:#94a3b8;display:flex;align-items:center;gap:4px;cursor:pointer">
          <input type="checkbox" id="road-vehicle" style="width:14px;height:14px"/>
          <span>🚗 Cho xe đi</span>
        </label>
      </div>
    </div>
    <div style="display:grid;grid-template-columns:2fr 1fr;gap:8px;margin-top:4px">
      <button class="tbtn active" onclick="saveRoad()" style="padding:12px;font-size:14px;background:#10b981;border-color:#10b981">
        ✅ Tạo đường
      </button>
      <button class="tbtn" onclick="cancelRoad()">✕ Huỷ</button>
    </div>
  </div>
</div>

<div id="map"></div>
<div id="status">Sẵn sàng</div>

<script>
// Global data
window.LOCATIONS = {locs_json};
window.EDGES = {edges_json};
window.FLOOR_COLORS = {floor_colors_json};
</script>
<script src="/static/js/localmap.js"></script>

</body>
</html>"""
    return HTMLResponse(html)



@router.get("/api/nearby")
async def nearby(lat: float, lon: float, radius: float = 0.01):
    locs = await db.nearby_locations(lat, lon, radius)
    pois = await db.nearby_pois(lat, lon, radius)
    return {"locations": locs, "pois": pois}


@router.get("/api/search")
async def search(q: str, limit: int = 10, locations_only: bool = False):
    locs = await db.search_locations_ranked(q, limit=limit)
    if locations_only:
        return {"locations": locs, "pois": []}
    pois = await db.search_pois(q)
    return {"locations": locs, "pois": pois[:limit]}


@router.get("/api/locations")
async def list_locations(limit: int = 100, offset: int = 0):
    return await db.fetchall(
        "SELECT l.*, COUNT(i.id) AS image_count FROM locations l "
        "LEFT JOIN images i ON i.location_id=l.id "
        "GROUP BY l.id ORDER BY l.importance DESC LIMIT ? OFFSET ?",
        (limit, offset),
    )


@router.get("/api/location/{location_id}/images")
async def get_location_images(location_id: int):
    """Get all images for a specific location."""
    images = await db.get_images_for_location(location_id)
    return {"images": images}


@router.delete("/api/location/{location_id}")
async def delete_location(location_id: int):
    """Delete a location and its associated images."""
    try:
        # Delete associated images first
        images = await db.get_images_for_location(location_id)
        for img in images:
            # Delete image file
            img_path = Path(img["filepath"])
            if img_path.exists():
                img_path.unlink()
            # Delete from database
            await db.execute("DELETE FROM images WHERE id=?", (img["id"],))
        
        # Delete location
        await db.execute("DELETE FROM locations WHERE id=?", (location_id,))
        return {"ok": True, "message": "Location deleted"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/api/edge")
async def create_edge(data: dict):
    """Create a new custom edge (road) between two points."""
    try:
        name = data.get("name", "Đường mới")
        from_lat = float(data["from_lat"])
        from_lon = float(data["from_lon"])
        to_lat = float(data["to_lat"])
        to_lon = float(data["to_lon"])
        road_type = data.get("road_type", "corridor")
        bidirectional = data.get("bidirectional", True)
        from_floor = data.get("from_floor", 1)
        to_floor = data.get("to_floor", 1)
        is_covered = data.get("is_covered", False)
        surface = data.get("surface", "tile")
        slope_deg = data.get("slope_deg", 0)
        geometry = data.get("geometry")  # list of [lat, lon] pairs
        
        edge_id, distance_m = await db.add_custom_edge(
            from_lat=from_lat,
            from_lon=from_lon,
            to_lat=to_lat,
            to_lon=to_lon,
            name=name,
            road_type=road_type,
            bidirectional=bidirectional,
            from_floor=from_floor,
            to_floor=to_floor,
            geometry=geometry,
        )
        
        # Update physical properties
        await db.execute(
            """UPDATE custom_edges
               SET is_covered=?, surface=?, slope_deg=?
               WHERE id=?""",
            (int(is_covered), surface, slope_deg, edge_id),
        )
        
        return {
            "ok": True,
            "id": edge_id,
            "distance_m": distance_m,
            "message": f"Đã tạo đường '{name}'"
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/api/edge/find")
async def find_edge(from_lat: float, from_lon: float, to_lat: float, to_lon: float):
    """Find existing edges between two points."""
    try:
        edges = await db.find_edge_between_points(from_lat, from_lon, to_lat, to_lon)
        return {"ok": True, "edges": edges}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.delete("/api/edge/{edge_id}")
async def delete_edge(edge_id: int):
    """Delete a custom edge."""
    try:
        await db.delete_edge(edge_id)
        return {"ok": True, "message": "Đã xóa đường"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
