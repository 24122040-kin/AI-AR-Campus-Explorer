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
    """
    Interactive Leaflet map of HCMUS Campus 2 (Linh Trung, Thu Duc).
    Shows all locations (colour-coded by floor), custom edges, and
    supports click-to-add-edge mode.
    """
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

    # Floor colour palette — up to 10 floors
    floor_colors = [
        "#3b82f6",  # 1 blue
        "#10b981",  # 2 green
        "#f59e0b",  # 3 amber
        "#ef4444",  # 4 red
        "#8b5cf6",  # 5 purple
        "#06b6d4",  # 6 cyan
        "#f97316",  # 7 orange
        "#ec4899",  # 8 pink
        "#84cc16",  # 9 lime
        "#6b7280",  # 10 gray
    ]
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
  .tbtn {{ padding: 6px 12px; border-radius: 6px; border: 1px solid #475569; background: #334155; color: #e2e8f0; cursor: pointer; font-size: 12px; }}
  .tbtn:hover {{ background: #475569; }}
  .tbtn.active {{ background: #3b82f6; border-color: #3b82f6; color: #fff; }}
  #floor-filter {{ display: flex; gap: 4px; align-items: center; font-size: 12px; }}
  #floor-filter label {{ color: #94a3b8; }}
  #floor-sel {{ background: #334155; border: 1px solid #475569; color: #e2e8f0; border-radius: 4px; padding: 4px 8px; font-size: 12px; }}
  #map {{ flex: 1; }}
  #status {{ padding: 6px 12px; background: #1e293b; font-size: 11px; color: #64748b; border-top: 1px solid #334155; }}
  .panel-strip {{ display: none; padding: 8px 12px; background: #1e293b; border-top: 1px solid #334155; gap: 8px; align-items: center; flex-wrap: wrap; }}
  .panel-strip.show {{ display: flex; }}
  #road-panel {{ flex-direction: column; align-items: stretch; }}
  #road-panel > div {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; padding: 8px 0; }}
  #road-panel input, #road-panel select {{ background: #334155; border: 1px solid #475569; color: #e2e8f0; border-radius: 4px; padding: 4px 8px; font-size: 12px; }}
  #edge-panel input, #edge-panel select {{ background: #334155; border: 1px solid #475569; color: #e2e8f0; border-radius: 4px; padding: 4px 8px; font-size: 12px; }}
  .legend {{ background: #1e293b; padding: 8px; border-radius: 6px; border: 1px solid #334155; font-size: 11px; line-height: 1.8; }}
  .legend-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }}
  .loc-marker-badge {{
    display: flex; align-items: center; justify-content: center;
    border-radius: 50%; color: #fff; font-weight: 700;
    border: 2px solid #fff; box-shadow: 0 2px 6px rgba(0,0,0,.45);
    cursor: pointer; line-height: 1; text-align: center;
    min-width: 26px; min-height: 26px; padding: 0 3px;
    box-sizing: border-box;
  }}
  .stack-pick-btn:hover {{ background: #475569 !important; border-color: #6366f1 !important; }}
  #walk-hud {{ display: none; padding: 8px 12px; background: #0f172a; border-top: 1px solid #334155;
    font-size: 12px; color: #94a3b8; gap: 12px; align-items: center; flex-wrap: wrap; }}
  #walk-hud.show {{ display: flex; }}
</style>
</head>
<body>

<div id="toolbar">
  <h2>🗺 ĐHKHTN CS2</h2>
  <button class="tbtn active" id="btn-view" onclick="setMode('view')">👁 Xem</button>
  <button class="tbtn" id="btn-add-road" onclick="setMode('add-road')">➕ Tạo đường</button>
  <button class="tbtn" id="btn-route" onclick="setMode('route')">🚗 Tìm đường</button>
  <div id="floor-filter">
    <label>Tầng:</label>
    <select id="floor-sel" onchange="filterFloor(this.value)">
      <option value="all">Tất cả</option>
    </select>
  </div>
  <button class="tbtn" onclick="closeMap()">✕ Đóng</button>
</div>

<!-- Panel tạo đường mới -->
<div id="road-panel" class="panel-strip">
  <div id="road-step-1" style="display:none">
    <span style="font-size:13px;color:#e2e8f0;font-weight:600">Bước 1: Chọn 2 điểm</span>
    <span id="road-hint" style="font-size:11px;color:#94a3b8;margin-left:12px">Click điểm A...</span>
  </div>
  
  <div id="road-step-2" style="display:none">
    <span style="font-size:13px;color:#e2e8f0;font-weight:600">Bước 2: Chọn cách tạo đường</span>
    <button class="tbtn active" onclick="selectRoadMethod('straight')">📏 Nối thẳng</button>
    <button class="tbtn active" onclick="selectRoadMethod('tracking')">🚶 Tracking đi bộ</button>
    <button class="tbtn" onclick="cancelRoad()">✕ Huỷ</button>
  </div>
  
  <div id="road-step-3-tracking" style="display:none">
    <span style="font-size:13px;color:#e2e8f0;font-weight:600">Bước 3: Tracking</span>
    <button class="tbtn active" id="btn-start-walk" onclick="startWalkTracking()">▶️ Bắt đầu đi</button>
    <button class="tbtn" id="btn-stop-walk" onclick="stopWalkTracking()" style="display:none">⏹ Kết thúc</button>
    <button class="tbtn" onclick="resetWalkTracking()">🔄 Reset</button>
    <span id="walk-status" style="font-size:11px;color:#94a3b8;margin-left:8px">Chưa bắt đầu</span>
  </div>
  
  <div id="road-step-4-form" style="display:none">
    <span style="font-size:13px;color:#e2e8f0;font-weight:600;margin-bottom:8px;display:block">Bước 4: Thông tin đường</span>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;align-items:end">
      <div>
        <label style="font-size:10px;color:#94a3b8;display:block;margin-bottom:2px">Tên đường</label>
        <input id="road-name" placeholder="Hành lang A..." style="width:100%;padding:6px 8px;background:#334155;border:1px solid #475569;color:#e2e8f0;border-radius:4px;font-size:12px"/>
      </div>
      <div>
        <label style="font-size:10px;color:#94a3b8;display:block;margin-bottom:2px">Loại đường</label>
        <select id="road-type" style="width:100%;padding:6px 8px;background:#334155;border:1px solid #475569;color:#e2e8f0;border-radius:4px;font-size:12px">
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
        <select id="road-surface" style="width:100%;padding:6px 8px;background:#334155;border:1px solid #475569;color:#e2e8f0;border-radius:4px;font-size:12px">
          <option value="tile">Gạch lát</option>
          <option value="concrete">Bê tông</option>
          <option value="asphalt">Nhựa đường</option>
          <option value="grass">Cỏ/Đất</option>
          <option value="gravel">Sỏi</option>
        </select>
      </div>
      <div>
        <label style="font-size:10px;color:#94a3b8;display:block;margin-bottom:2px">Độ dốc (°)</label>
        <input id="road-slope" type="number" value="0" min="-45" max="45" style="width:100%;padding:6px 8px;background:#334155;border:1px solid #475569;color:#e2e8f0;border-radius:4px;font-size:12px"/>
      </div>
      <div>
        <label style="font-size:10px;color:#94a3b8;display:block;margin-bottom:2px">Chiều</label>
        <select id="road-direction" style="width:100%;padding:6px 8px;background:#334155;border:1px solid #475569;color:#e2e8f0;border-radius:4px;font-size:12px">
          <option value="both">2 chiều</option>
          <option value="oneway">1 chiều</option>
        </select>
      </div>
      <div style="display:flex;gap:8px;align-items:center">
        <label style="font-size:11px;color:#94a3b8;display:flex;align-items:center;gap:4px;cursor:pointer">
          <input type="checkbox" id="road-covered" checked style="width:14px;height:14px"/>
          <span>Mái che</span>
        </label>
        <label style="font-size:11px;color:#94a3b8;display:flex;align-items:center;gap:4px;cursor:pointer">
          <input type="checkbox" id="road-vehicle" style="width:14px;height:14px"/>
          <span>Cho xe</span>
        </label>
      </div>
    </div>
    <div style="display:flex;gap:8px;margin-top:12px">
      <button class="tbtn active" onclick="saveRoad()" style="flex:1">✅ Tạo đường</button>
      <button class="tbtn" onclick="cancelRoad()">✕ Huỷ</button>
    </div>
  </div>
</div>

<!-- Panel tìm đường -->
<div id="route-panel" class="panel-strip">
  <span style="font-size:12px;color:#94a3b8">Chọn 2 điểm để tìm đường:</span>
  <span id="route-hint" style="font-size:11px;color:#64748b">Click điểm xuất phát...</span>
  <button class="tbtn active" onclick="findRouteOnMap()">🔍 Tìm đường</button>
  <button class="tbtn" onclick="cancelRoute()">✕ Huỷ</button>
</div>

<div id="map"></div>
<div id="status">Sẵn sàng</div>

<script>
const LOCATIONS = {locs_json};
const EDGES = {edges_json};
const FLOOR_COLORS = {floor_colors_json};
const API = '';
const INITIAL_MODE = {mode!r};

// ── Map init ──────────────────────────────────────────────────────────────────
const map = L.map('map').setView([10.8720, 106.8042], 17);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution: '© OpenStreetMap',
  maxZoom: 21,
}}).addTo(map);

// ── Campus boundary polygon (một lần; tránh chồng khi fetch lặp) ───────────
let _campusBoundaryLayer = null;
let _campusRoadLabelMarkers = [];
fetch(API + '/api/campus/boundary').then(r => r.json()).then(b => {{
  if (!b.polygon) return;
  if (_campusBoundaryLayer) {{
    map.removeLayer(_campusBoundaryLayer);
    _campusBoundaryLayer = null;
  }}
  _campusRoadLabelMarkers.forEach(m => map.removeLayer(m));
  _campusRoadLabelMarkers = [];
  const poly = b.polygon.map(p => [p[0], p[1]]);
  _campusBoundaryLayer = L.polygon(poly, {{
    color: '#3b82f6',
    weight: 2,
    opacity: 0.8,
    fillColor: '#3b82f6',
    fillOpacity: 0.05,
    dashArray: '8,5',
  }}).bindTooltip('ĐHKHTN CS2 — Khuôn viên', {{sticky: true}}).addTo(map);

  const roadLabels = [
    {{ pos: [10.8753, 106.8042], text: 'Đường Marie Curie (Bắc)' }},
    {{ pos: [10.8693, 106.8042], text: 'Đường Isaac Newton (Nam)' }},
    {{ pos: [10.8722, 106.8068], text: 'Quảng trường Sáng tạo (Đông)' }},
  ];
  roadLabels.forEach(r => {{
    const lm = L.marker(r.pos, {{
      icon: L.divIcon({{
        className: '',
        html: `<div style="background:rgba(15,23,42,0.75);color:#94a3b8;
                           font-size:10px;padding:2px 6px;border-radius:4px;
                           white-space:nowrap;border:1px solid #334155">${{r.text}}</div>`,
        iconAnchor: [60, 10],
      }})
    }}).addTo(map);
    _campusRoadLabelMarkers.push(lm);
  }});
}}).catch(() => {{}});

// ── State ─────────────────────────────────────────────────────────────────────
let mode = 'view';
const SNAP_M = 55;

// Road creation workflow
let roadPointA = null, roadPointB = null;
let roadMarkerA = null, roadMarkerB = null;
let roadPreviewLine = null;
let roadMethod = null; // 'straight' or 'tracking'
let walkPoints = [];
let walkPolyline = null;
let walkWatchId = null;
let walkDistance = 0;
let walkLastPos = null;

// Route finding
let routePointA = null, routePointB = null;
let routeMarkerA = null, routeMarkerB = null;
let routePreviewLine = null;

// General
let locMarkers = [];
let edgeLines = [];
let currentFloor = 'all';
let _currentOptionsPopup = null;

function haversineM(lat1, lon1, lat2, lon2) {{
  const R = 6371000;
  const toR = x => x * Math.PI / 180;
  const dLat = toR(lat2 - lat1), dLon = toR(lon2 - lon1);
  const a = Math.sin(dLat/2)**2 + Math.cos(toR(lat1))*Math.cos(toR(lat2))*Math.sin(dLon/2)**2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}}

function collectSnapRows(lat, lon, respectFloorFilter) {{
  const rows = [];
  for (const loc of LOCATIONS) {{
    if (respectFloorFilter && currentFloor !== 'all' && (loc.floor || 1) !== currentFloor) continue;
    const d = haversineM(lat, lon, loc.lat, loc.lon);
    if (d <= SNAP_M) rows.push({{ loc, dM: d }});
  }}
  return rows;
}}

function coordKey6(loc) {{
  return (Number(loc.lat) || 0).toFixed(6) + ',' + (Number(loc.lon) || 0).toFixed(6);
}}

function nearestSavedLocation(lat, lon) {{
  if (!LOCATIONS.length) return null;
  let rows = collectSnapRows(lat, lon, true);
  if (!rows.length) rows = collectSnapRows(lat, lon, false);
  if (!rows.length) return null;
  rows.sort((a, b) => a.dM - b.dM || (a.loc.floor || 1) - (b.loc.floor || 1) || a.loc.id - b.loc.id);
  const best = rows[0];
  const k0 = coordKey6(best.loc);
  const sameStack = rows.filter(r => coordKey6(r.loc) === k0);
  const ambiguous = sameStack.length > 1;
  return {{
    loc: best.loc,
    dM: best.dM,
    sameCoordGroup: ambiguous ? sameStack.map(r => r.loc) : [],
  }};
}}

function openSameCoordPicker(latlng, locs, onPick) {{
  let html = '<div style="min-width:220px;font-size:13px;color:#e2e8f0">';
  html += '<div style="margin-bottom:8px;color:#94a3b8;font-size:11px">Cùng vị trí GPS — chọn đúng tầng / địa điểm:</div>';
  for (const loc of locs) {{
    html += `<button type="button" class="stack-pick-btn" data-id="${{loc.id}}" style="display:block;width:100%;margin:4px 0;padding:8px;border-radius:8px;border:1px solid #475569;background:#334155;cursor:pointer;color:#e2e8f0;text-align:left">Tầng <b>${{loc.floor || 1}}</b> · #${{loc.id}} ${{String((loc.name || '')).slice(0, 48)}}</button>`;
  }}
  html += '</div>';
  const p = L.popup({{ maxWidth: 340 }}).setLatLng(latlng).setContent(html).openOn(map);
  setTimeout(() => {{
    const el = p.getElement();
    if (!el) return;
    el.querySelectorAll('.stack-pick-btn').forEach(btn => {{
      btn.addEventListener('click', () => {{
        const id = parseInt(btn.getAttribute('data-id'), 10);
        const loc = LOCATIONS.find(l => l.id === id);
        map.closePopup();
        if (loc) onPick(loc);
      }});
    }});
  }}, 0);
}}

function setMode(m) {{
  mode = m;
  // Update toolbar buttons
  ['btn-view', 'btn-add-road', 'btn-route'].forEach(id => {{
    const btn = document.getElementById(id);
    if (btn) btn.classList.remove('active');
  }});
  const activeBtn = document.getElementById('btn-' + m.replace('add-road', 'add-road').replace('route', 'route').replace('view', 'view'));
  if (activeBtn) activeBtn.classList.add('active');
  
  // Show/hide panels
  const roadPanel = document.getElementById('road-panel');
  const routePanel = document.getElementById('route-panel');
  if (roadPanel) roadPanel.classList.toggle('show', m === 'add-road');
  if (routePanel) routePanel.classList.toggle('show', m === 'route');
  
  // Reset states
  if (m !== 'add-road') cancelRoad();
  if (m !== 'route') cancelRoute();
  
  // Show step 1 when entering add-road mode
  if (m === 'add-road') {{
    document.getElementById('road-step-1').style.display = 'flex';
    document.getElementById('road-hint').textContent = 'Click điểm A...';
  }}
  
  setStatus(
    m === 'add-road' ? '➕ Chọn 2 điểm để tạo đường'
    : m === 'route' ? '🚗 Chọn 2 điểm để tìm đường'
    : '👁 Xem bản đồ - click marker để xem thông tin'
  );
}}

function closeMap() {{
  if (window.parent && window.parent !== window) {{
    window.parent.postMessage({{type: 'close-localmap'}}, '*');
  }}
}}

// ── Floor colour helper ───────────────────────────────────────────────────────
function floorColor(floor) {{
  return FLOOR_COLORS[(floor - 1) % FLOOR_COLORS.length] || '#3b82f6';
}}

// Nhóm điểm trùng tọa độ — lệch nhẹ marker + vòng chọn để không chồng hình tròn
let _coordGroups = new Map();
function buildLocationCoordGroups() {{
  const g = new Map();
  for (const loc of LOCATIONS) {{
    const k = (Number(loc.lat) || 0).toFixed(6) + ',' + (Number(loc.lon) || 0).toFixed(6);
    if (!g.has(k)) g.set(k, []);
    g.get(k).push(loc);
  }}
  for (const arr of g.values()) {{
    arr.sort((a, b) => (a.floor || 1) - (b.floor || 1) || a.id - b.id);
  }}
  return g;
}}
function displayLatLonForLocation(loc, coordGroups) {{
  const k = (Number(loc.lat) || 0).toFixed(6) + ',' + (Number(loc.lon) || 0).toFixed(6);
  const grp = coordGroups.get(k) || [loc];
  const idx = Math.max(0, grp.findIndex(x => x.id === loc.id));
  const n = grp.length;
  const lat = Number(loc.lat) || 0, lon = Number(loc.lon) || 0;
  if (n <= 1) return [lat, lon];
  const floor = Number(loc.floor) || 1;
  const meters = 2.4 + idx * 3.0 + (floor - 1) * 2.8;
  const angleDeg = -90 + idx * (360 / n);
  const rad = angleDeg * Math.PI / 180;
  const cosLat = Math.cos(lat * Math.PI / 180) || 1e-6;
  const dLat = meters * Math.cos(rad) / 111320;
  const dLon = meters * Math.sin(rad) / (111320 * cosLat);
  return [lat + dLat, lon + dLon];
}}
function markerRingLatLng(lat, lon, id) {{
  if (id != null && _coordGroups && _coordGroups.size) {{
    const loc = LOCATIONS.find(x => x.id === id);
    if (loc) return displayLatLonForLocation(loc, _coordGroups);
  }}
  return [lat, lon];
}}
function selectionRingOpts(color) {{
  return {{ radius: 17, color, weight: 3, opacity: 1, fill: false }};
}}

// ── Show location options popup (normal mode) ─────────────────────────────────
let _currentOptionsPopup = null;
async function showLocationOptionsPopup(loc, latlng) {{
  // Close any existing popup
  if (_currentOptionsPopup) {{
    map.closePopup(_currentOptionsPopup);
    _currentOptionsPopup = null;
  }}

  // Fetch image for this location
  let imgHtml = '';
  try {{
    const r = await fetch(API + '/api/location/' + loc.id + '/images');
    const d = await r.json();
    if (d.images && d.images.length > 0) {{
      const primaryImg = d.images.find(img => img.is_primary) || d.images[0];
      imgHtml = `<img src="/api/image/${{primaryImg.id}}" style="width:100%;max-width:240px;border-radius:8px;margin-bottom:10px" alt="${{loc.name}}"/>`;
    }}
  }} catch(e) {{
    // No image available
  }}

  const popupContent = `
    <div style="min-width:220px;max-width:280px;font-family:-apple-system,sans-serif">
      ${{imgHtml}}
      <div style="margin-bottom:12px">
        <div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:4px">${{loc.name}}</div>
        <div style="font-size:12px;color:#64748b;margin-bottom:2px">
          📍 Tầng ${{loc.floor || 1}} · ${{loc.category || 'địa điểm'}}
        </div>
        ${{loc.description ? `<div style="font-size:12px;color:#475569;margin-top:6px;font-style:italic">${{loc.description}}</div>` : ''}}
        <div style="font-size:10px;color:#94a3b8;margin-top:6px">
          ID: #${{loc.id}} · ${{Number(loc.lat).toFixed(6)}}, ${{Number(loc.lon).toFixed(6)}}
        </div>
      </div>
      <div style="display:flex;flex-direction:column;gap:6px">
        <button class="loc-option-btn" data-action="set-start" data-id="${{loc.id}}" 
                style="width:100%;padding:8px 12px;background:#10b981;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:600">
          🚀 Chọn làm điểm xuất phát
        </button>
        <button class="loc-option-btn" data-action="set-dest" data-id="${{loc.id}}"
                style="width:100%;padding:8px 12px;background:#3b82f6;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:600">
          🎯 Chọn làm điểm đến
        </button>
        <button class="loc-option-btn" data-action="delete" data-id="${{loc.id}}"
                style="width:100%;padding:8px 12px;background:#ef4444;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:600">
          🗑️ Xóa điểm này
        </button>
      </div>
    </div>
  `;

  const popup = L.popup({{ maxWidth: 320, closeButton: true }})
    .setLatLng(latlng)
    .setContent(popupContent)
    .openOn(map);

  _currentOptionsPopup = popup;

  // Add event listeners after popup is rendered
  setTimeout(() => {{
    const el = popup.getElement();
    if (!el) return;
    
    el.querySelectorAll('.loc-option-btn').forEach(btn => {{
      btn.addEventListener('click', async () => {{
        const action = btn.getAttribute('data-action');
        const id = parseInt(btn.getAttribute('data-id'), 10);
        
        if (action === 'set-start') {{
          if (window.parent && window.parent !== window) {{
            window.parent.postMessage({{
              type: 'localnav-set-route-point',
              field: 'from',
              location: loc
            }}, '*');
            setStatus(`✅ Đã chọn "${{loc.name}}" làm điểm xuất phát`);
          }}
        }} else if (action === 'set-dest') {{
          if (window.parent && window.parent !== window) {{
            window.parent.postMessage({{
              type: 'localnav-set-route-point',
              field: 'to',
              location: loc
            }}, '*');
            setStatus(`✅ Đã chọn "${{loc.name}}" làm điểm đến`);
          }}
        }} else if (action === 'delete') {{
          if (!confirm(`Xác nhận xóa địa điểm "${{loc.name}}"?`)) return;
          try {{
            const r = await fetch(API + '/api/location/' + id, {{ method: 'DELETE' }});
            const d = await r.json();
            if (d.ok) {{
              // Remove from local array and re-render
              const idx = LOCATIONS.findIndex(l => l.id === id);
              if (idx >= 0) LOCATIONS.splice(idx, 1);
              renderLocations();
              setStatus(`✅ Đã xóa "${{loc.name}}"`);
            }} else {{
              setStatus('❌ Lỗi xóa địa điểm');
            }}
          }} catch(e) {{
            setStatus('❌ ' + e.message);
          }}
        }}
        
        map.closePopup();
        _currentOptionsPopup = null;
      }});
    }});
  }}, 50);
}}

// ── Render locations ──────────────────────────────────────────────────────────
function renderLocations() {{
  locMarkers.forEach(m => map.removeLayer(m));
  locMarkers = [];
  _coordGroups = buildLocationCoordGroups();

  const floors = new Set();
  LOCATIONS.forEach(loc => floors.add(loc.floor || 1));

  // Update floor filter dropdown
  const sel = document.getElementById('floor-sel');
  const prev = sel.value;
  sel.innerHTML = '<option value="all">Tất cả</option>';
  [...floors].sort().forEach(f => {{
    const opt = document.createElement('option');
    opt.value = f; opt.textContent = `Tầng ${{f}}`;
    sel.appendChild(opt);
    // also populate edge floor selects
    ['ep-from-floor','ep-to-floor'].forEach(id => {{
      const s = document.getElementById(id);
      if (![...s.options].find(o => o.value == f)) {{
        const o = document.createElement('option');
        o.value = f; o.textContent = `Tầng ${{f}}`;
        s.appendChild(o);
      }}
    }});
  }});
  sel.value = prev;

  LOCATIONS.forEach(loc => {{
    const floor = loc.floor || 1;
    if (currentFloor !== 'all' && floor != currentFloor) return;
    const color = floorColor(floor);
    const lid = String(loc.id != null ? loc.id : '?');
    const fs = lid.length >= 4 ? '8px' : lid.length >= 3 ? '9px' : '11px';
    const [mlat, mlon] = displayLatLonForLocation(loc, _coordGroups);
    const icon = L.divIcon({{
      className: '',
      html: `<div class="loc-marker-badge" style="background:${{color}};font-size:${{fs}};">${{lid}}</div>`,
      iconSize: [28, 28],
      iconAnchor: [14, 14],
    }});
    const sk = coordKey6(loc);
    const grp = _coordGroups.get(sk) || [];
    const stackNote = grp.length > 1
      ? `<br><span style="color:#fbbf24;font-size:10px">⚠ ${{grp.length}} điểm cùng vị trí GPS (khác tầng) — lọc «Tầng» hoặc bấm map khi nối cạnh / tuyến.</span>`
      : '';
    const m = L.marker([mlat, mlon], {{icon}})
      .bindPopup(`
        <b>#${{lid}} · ${{loc.name}}</b><br>
        <span style="color:#64748b;font-size:11px">Tầng ${{floor}} · ${{loc.category}}</span>
        ${{loc.description ? '<br><i>' + loc.description + '</i>' : ''}}
        <br><span style="font-size:10px;color:#94a3b8">${{Number(loc.lat).toFixed(6)}}, ${{Number(loc.lon).toFixed(6)}}</span>${{stackNote}}
      `)
      .addTo(map);
    m.setZIndexOffset(500 + floor * 15);

    m.on('click', async (e) => {{
      if (mode === 'add-edge') {{
        L.DomEvent.stopPropagation(e);
        selectEdgePoint(loc.lat, loc.lon, loc.name, loc.floor || 1, loc.id);
      }} else if (mode === 'route-drive') {{
        L.DomEvent.stopPropagation(e);
        selectRouteDrivePointFromLoc(loc);
      }} else if (mode === 'normal') {{
        L.DomEvent.stopPropagation(e);
        await showLocationOptionsPopup(loc, [mlat, mlon]);
      }}
    }});
    locMarkers.push(m);
  }});
}}

// ── Render edges ──────────────────────────────────────────────────────────────
function renderEdges() {{
  edgeLines.forEach(l => map.removeLayer(l));
  edgeLines = [];
  EDGES.forEach(e => {{
    const fromFloor = e.from_floor || 1;
    const toFloor = e.to_floor || 1;
    if (currentFloor !== 'all' && fromFloor != currentFloor && toFloor != currentFloor) return;
    const color = fromFloor !== toFloor ? '#f59e0b' : floorColor(fromFloor);
    const pts = e.geometry && e.geometry.length >= 2
      ? e.geometry.map(p => [p[0], p[1]])
      : [[e.from_lat, e.from_lon], [e.to_lat, e.to_lon]];
    const line = L.polyline(pts, {{
      color, weight: 3, opacity: 0.85,
      dashArray: e.road_type === 'stairs' ? '6,4' : null,
    }})
      .bindPopup(`<b>${{e.name || 'Đường'}}</b><br>${{e.road_type}} · ${{Math.round(e.distance_m)}}m`)
      .addTo(map);
    edgeLines.push(line);
  }});
}}

// ── Floor filter ──────────────────────────────────────────────────────────────
function filterFloor(val) {{
  currentFloor = val === 'all' ? 'all' : parseInt(val);
  renderLocations();
  renderEdges();
}}

// ── Mode switching ────────────────────────────────────────────────────────────
function setMode(m) {{
  if (m !== 'walk' && walkWatchId !== null) discardWalkSilent();
  if (m === 'walk') {{
    applyModeUI('walk');
    startWalk();
    return;
  }}
  if (m !== 'add-edge') cancelEdge();
  if (m !== 'route-drive') cancelRouteDrive();
  applyModeUI(m);
}}

// ── Route drive: 2 saved DB locations → parent app finds driving route ─────
function selectRouteDrivePointFromLoc(loc) {{
  const lat = loc.lat, lon = loc.lon, label = loc.name, floor = loc.floor || 1, id = loc.id;
  if (!routePointA) {{
    routePointA = {{ lat, lon, name: label, floor, id }};
    if (routeMarkerA) map.removeLayer(routeMarkerA);
    const [pla, ploa] = markerRingLatLng(lat, lon, id);
    routeMarkerA = L.circleMarker([pla, ploa], selectionRingOpts('#22c55e'))
      .bindTooltip('Đi: ' + label, {{ permanent: true }}).addTo(map);
    const rh = document.getElementById('rp-hint');
    if (rh) rh.textContent = 'Đi: ' + label + ' — chọn điểm đến…';
  }} else if (!routePointB) {{
    if (haversineM(routePointA.lat, routePointA.lon, lat, lon) < 12
        && routePointA.id === id && (routePointA.floor || 1) === (floor || 1)) {{
      setStatus('Chọn điểm đến khác (khác tầng hoặc địa điểm).');
      return;
    }}
    routePointB = {{ lat, lon, name: label, floor, id }};
    if (routeMarkerB) map.removeLayer(routeMarkerB);
    const [plb, plob] = markerRingLatLng(lat, lon, id);
    routeMarkerB = L.circleMarker([plb, plob], selectionRingOpts('#f97316'))
      .bindTooltip('Đến: ' + label, {{ permanent: true }}).addTo(map);
    if (routePreviewLine) map.removeLayer(routePreviewLine);
    routePreviewLine = L.polyline([[routePointA.lat, routePointA.lon], [lat, lon]], {{ color: '#38bdf8', weight: 4, dashArray: '4,6' }}).addTo(map);
    const rh = document.getElementById('rp-hint');
    if (rh) rh.textContent = routePointA.name + ' → ' + label + ' — bấm «Gửi về form tìm đường».';
  }} else {{
    cancelRouteDrive();
    selectRouteDrivePointFromLoc(loc);
  }}
}}

function cancelRouteDrive() {{
  routePointA = routePointB = null;
  [routeMarkerA, routeMarkerB, routePreviewLine].forEach(l => {{ if (l) map.removeLayer(l); }});
  routeMarkerA = routeMarkerB = routePreviewLine = null;
  const rh = document.getElementById('rp-hint');
  if (rh) rh.textContent = 'Chọn điểm xuất phát (marker hoặc map gần điểm đã lưu ≤' + SNAP_M + 'm)…';
}}

function confirmRouteDrive() {{
  if (!routePointA || !routePointB) return alert('Chọn đủ 2 địa điểm đã lưu trong DB');
  if (window.parent && window.parent !== window) {{
    window.parent.postMessage({{
      type: 'localnav-route-drive',
      from: {{ id: routePointA.id, name: routePointA.name, lat: routePointA.lat, lon: routePointA.lon, floor: routePointA.floor }},
      to: {{ id: routePointB.id, name: routePointB.name, lat: routePointB.lat, lon: routePointB.lon, floor: routePointB.floor }},
    }}, '*');
    setStatus('Đã gửi tuyến về LocalNavBot.');
    cancelRouteDrive();
    applyModeUI('normal');
    return;
  }}
  alert('Mở bản đồ từ app LocalNavBot (iframe) để gửi tuyến.');
}}

// ── Add-edge mode ─────────────────────────────────────────────────────────────
function selectEdgePoint(lat, lon, label, floor, id) {{
  if (!edgePointA) {{
    edgePointA = {{lat, lon, name: label, label, floor: floor || 1, id: id || null}};
    if (edgeMarkerA) map.removeLayer(edgeMarkerA);
    const [eaLat, eaLon] = markerRingLatLng(lat, lon, id);
    edgeMarkerA = L.circleMarker([eaLat, eaLon], selectionRingOpts('#3b82f6'))
      .bindTooltip('A: ' + (label || `${{lat.toFixed(5)}},${{lon.toFixed(5)}}`), {{permanent:true}})
      .addTo(map);
    document.getElementById('ep-hint').textContent = 'Điểm A đã chọn. Click điểm B...';
  }} else if (!edgePointB) {{
    edgePointB = {{lat, lon, name: label, label, floor: floor || 1, id: id || null}};
    if (edgeMarkerB) map.removeLayer(edgeMarkerB);
    const [ebLat, ebLon] = markerRingLatLng(lat, lon, id);
    edgeMarkerB = L.circleMarker([ebLat, ebLon], selectionRingOpts('#ef4444'))
      .bindTooltip('B: ' + (label || `${{lat.toFixed(5)}},${{lon.toFixed(5)}}`), {{permanent:true}})
      .addTo(map);
    if (edgePreviewLine) map.removeLayer(edgePreviewLine);
    edgePreviewLine = L.polyline([[edgePointA.lat, edgePointA.lon],[lat,lon]], {{color:'#f59e0b', dashArray:'6,4', weight:3}}).addTo(map);
    document.getElementById('ep-hint').textContent = 'Điền tên rồi bấm Lưu đường.';
  }} else {{
    // Reset and start over
    cancelEdge();
    selectEdgePoint(lat, lon, label, floor, id);
  }}
}}

map.on('click', (ev) => {{
  if (mode !== 'add-edge' && mode !== 'route-drive') return;
  const lat = ev.latlng.lat, lon = ev.latlng.lng;
  const hit = nearestSavedLocation(lat, lon);
  if (!hit) {{
    setStatus('Không có địa điểm đã lưu trong DB trong bán kính ' + SNAP_M + 'm. Chọn marker, đổi Tầng, hoặc zoom gần hơn.');
    return;
  }}
  const pick = (loc) => {{
    if (mode === 'add-edge') selectEdgePoint(loc.lat, loc.lon, loc.name, loc.floor || 1, loc.id);
    else selectRouteDrivePointFromLoc(loc);
  }};
  if (hit.sameCoordGroup && hit.sameCoordGroup.length > 1) {{
    openSameCoordPicker(ev.latlng, hit.sameCoordGroup, pick);
    return;
  }}
  pick(hit.loc);
}});

async function confirmEdge() {{
  if (!edgePointA || !edgePointB) return alert('Chọn 2 điểm trước');
  if (window.parent && window.parent !== window) {{
    window.parent.postMessage({{
      type: 'localnav-edge-points',
      a: edgePointA,
      b: edgePointB,
    }}, '*');
    setStatus('Đã gửi 2 địa điểm về form thêm đường.');
    cancelEdge();
    return;
  }}
  const name = document.getElementById('ep-name').value.trim() || 'Đường mới';
  const road_type = document.getElementById('ep-type').value;
  const from_floor = parseInt(document.getElementById('ep-from-floor').value) || 1;
  const to_floor = parseInt(document.getElementById('ep-to-floor').value) || 1;
  try {{
    const r = await fetch(API + '/api/edge', {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        from_lat: edgePointA.lat, from_lon: edgePointA.lon,
        to_lat: edgePointB.lat,   to_lon: edgePointB.lon,
        name, road_type, bidirectional: true,
        from_floor, to_floor,
      }}),
    }});
    const d = await r.json();
    if (d.ok) {{
      const dm = typeof d.distance_m === 'number' ? d.distance_m : 0;
      EDGES.push({{
        id: d.id, name, road_type,
        from_lat: edgePointA.lat, from_lon: edgePointA.lon,
        to_lat: edgePointB.lat, to_lon: edgePointB.lon,
        from_floor, to_floor, distance_m: dm, geometry: null,
      }});
      renderEdges();
      setStatus(`✅ Đã lưu đường "${{name}}" (~${{Math.round(dm)}}m)`);
    }} else {{
      setStatus('❌ Lỗi lưu đường');
    }}
  }} catch(e) {{ setStatus('❌ ' + e.message); }}
  cancelEdge();
}}

function cancelEdge() {{
  edgePointA = edgePointB = null;
  [edgeMarkerA, edgeMarkerB, edgePreviewLine].forEach(l => {{ if(l) map.removeLayer(l); }});
  edgeMarkerA = edgeMarkerB = edgePreviewLine = null;
  document.getElementById('ep-hint').textContent = 'Click điểm A trên bản đồ...';
  document.getElementById('ep-name').value = '';
}}

function parseWalkFloors() {{
  if (currentFloor !== 'all') return {{ from: currentFloor, to: currentFloor }};
  const raw = (prompt('Tầng xuất phát → tầng đích của lộ trình (vd: 1-2). Enter = 1-1', '1-1') || '1-1').trim();
  const parts = raw.split(/[-–]/);
  const a = parseInt(parts[0], 10), b = parseInt(parts[1] !== undefined ? parts[1] : parts[0], 10);
  return {{ from: Math.max(1, Number.isFinite(a) ? a : 1), to: Math.max(1, Number.isFinite(b) ? b : (Number.isFinite(a) ? a : 1)) }};
}}

// ── Walk tracking mode ────────────────────────────────────────────────────────
function startWalk() {{
  if (!navigator.geolocation) {{ alert('GPS không khả dụng'); applyModeUI('normal'); return; }}
  walkPoints = [];
  walkLiveM = 0;
  walkLastFix = null;
  walkActive = true;
  if (walkPolyline) map.removeLayer(walkPolyline);
  walkPolyline = L.polyline([], {{color:'#10b981', weight:4}}).addTo(map);

  document.getElementById('btn-walk').textContent = '⏹ Dừng tracking';
  document.getElementById('btn-walk').onclick = stopWalk;

  walkWatchId = navigator.geolocation.watchPosition(pos => {{
    const la = pos.coords.latitude, lo = pos.coords.longitude;
    if (la == null || lo == null || isNaN(la) || isNaN(lo)) return;
    const pt = [la, lo];
    if (walkLastFix) walkLiveM += haversineM(walkLastFix[0], walkLastFix[1], la, lo);
    walkLastFix = pt;
    walkPoints.push(pt);
    walkPolyline.addLatLng(pt);
    map.panTo(pt);
    const wdist = document.getElementById('walk-hud-dist');
    const wpts = document.getElementById('walk-hud-pts');
    if (wdist) {{
      wdist.textContent = walkLiveM < 1000
        ? 'Đã đi: ' + Math.round(walkLiveM) + ' m'
        : 'Đã đi: ' + (walkLiveM / 1000).toFixed(2) + ' km';
    }}
    if (wpts) wpts.textContent = walkPoints.length + ' điểm GPS';
    setStatus(`🚶 ${{walkPoints.length}} điểm · ~${{Math.round(walkLiveM)}} m · GPS ±${{pos.coords.accuracy.toFixed(0)}}m`);
  }}, err => {{
    setStatus('GPS error: ' + err.message);
  }}, {{enableHighAccuracy: true, maximumAge: 0, timeout: 5000}});
}}

async function stopWalk() {{
  walkActive = false;
  if (walkWatchId !== null) {{ navigator.geolocation.clearWatch(walkWatchId); walkWatchId = null; }}
  document.getElementById('btn-walk').textContent = '🚶 Tracking đi bộ';
  document.getElementById('btn-walk').onclick = () => setMode('walk');

  if (walkPoints.length < 2) {{ applyModeUI('normal'); return; }}

  // Douglas-Peucker simplification (epsilon = 2m ≈ 0.00002 deg)
  const simplified = douglasPeucker(walkPoints, 0.00002);
  setStatus(`✅ Đã thu ${{walkPoints.length}} điểm → simplified ${{simplified.length}} điểm`);

  // Ask user to name the path
  const name = prompt(`Đặt tên cho đường vừa đi (${{simplified.length}} điểm)?`, 'Đường đi bộ');
  if (!name) {{ applyModeUI('normal'); return; }}
  const road_type = prompt('Loại đường? (alley/path/shortcut/corridor/stairs)', 'path') || 'path';
  const fl = parseWalkFloors();

  const from = simplified[0], to = simplified[simplified.length - 1];
  try {{
    const r = await fetch(API + '/api/edge', {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        from_lat: from[0], from_lon: from[1],
        to_lat: to[0],     to_lon: to[1],
        name, road_type, bidirectional: true,
        from_floor: fl.from, to_floor: fl.to,
        geometry: simplified,
      }}),
    }});
    const d = await r.json();
    if (d.ok) {{
      const dm = typeof d.distance_m === 'number' ? d.distance_m : 0;
      EDGES.push({{
        id: d.id, name, road_type,
        from_lat: from[0], from_lon: from[1],
        to_lat: to[0], to_lon: to[1],
        from_floor: fl.from, to_floor: fl.to,
        distance_m: dm, geometry: simplified,
      }});
      renderEdges();
      setStatus(`✅ Đã lưu "${{name}}" · ~${{Math.round(dm)}} m · tầng ${{fl.from}}→${{fl.to}}`);
    }}
  }} catch(e) {{ setStatus('❌ ' + e.message); }}
  applyModeUI('normal');
}}

// ── Douglas-Peucker polyline simplification ───────────────────────────────────
function douglasPeucker(points, epsilon) {{
  if (points.length <= 2) return points;
  let maxDist = 0, maxIdx = 0;
  const first = points[0], last = points[points.length - 1];
  for (let i = 1; i < points.length - 1; i++) {{
    const d = pointToSegmentDist(points[i], first, last);
    if (d > maxDist) {{ maxDist = d; maxIdx = i; }}
  }}
  if (maxDist > epsilon) {{
    const left  = douglasPeucker(points.slice(0, maxIdx + 1), epsilon);
    const right = douglasPeucker(points.slice(maxIdx), epsilon);
    return [...left.slice(0, -1), ...right];
  }}
  return [first, last];
}}

function pointToSegmentDist(p, a, b) {{
  const dx = b[0]-a[0], dy = b[1]-a[1];
  if (dx===0 && dy===0) return Math.hypot(p[0]-a[0], p[1]-a[1]);
  const t = Math.max(0, Math.min(1, ((p[0]-a[0])*dx + (p[1]-a[1])*dy) / (dx*dx+dy*dy)));
  return Math.hypot(p[0]-(a[0]+t*dx), p[1]-(a[1]+t*dy));
}}

// ── Legend ────────────────────────────────────────────────────────────────────
const legend = L.control({{position:'bottomright'}});
legend.onAdd = () => {{
  const div = L.DomUtil.create('div','legend');
  div.innerHTML = '<b style="font-size:11px">Địa điểm</b><br>' +
    '<span style="color:#cbd5e1">Số trong vòng = <b>ID</b> DB · Màu = tầng</span><br>' +
    FLOOR_COLORS.slice(0,5).map((c,i) =>
      `<span class="legend-dot" style="background:${{c}}"></span>Tầng ${{i+1}}<br>`
    ).join('') +
    '<hr style="border-color:#475569;margin:4px 0"/>' +
    '<span style="color:#f59e0b">━━</span> Đường liên tầng<br>' +
    '<span style="color:#10b981">━━</span> Đường tracking';
  return div;
}};
legend.addTo(map);

// ── Status bar ────────────────────────────────────────────────────────────────
function setStatus(msg) {{ document.getElementById('status').textContent = msg; }}

// ── Init ──────────────────────────────────────────────────────────────────────
renderLocations();
renderEdges();
applyModeUI('normal');
if (INITIAL_MODE === 'edge-picker') setMode('add-edge');
if (INITIAL_MODE === 'route-picker') setMode('route-drive');
setStatus(`${{LOCATIONS.length}} địa điểm · ${{EDGES.length}} đường`);
</script>
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
