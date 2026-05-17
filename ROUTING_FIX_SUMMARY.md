# ✅ Đã fix thuật toán tìm đường indoor

## Vấn đề ban đầu
- Tìm đường từ "bếp" (tầng 1) → "sảnh tầng 4" (tầng 4) → kết quả: **0 km, "đã đến điểm đến"**
- Indoor graph không được build từ database
- Logic routing không tự động detect indoor
- Thiếu environmental factors (nắng/mưa/đông đúc)

## Các thay đổi đã thực hiện

### 1. ✅ Build Indoor Graph từ Database
**File:** `core/indoor_router.py`

Thêm function `build_indoor_graph_from_db()`:
- Load tất cả `locations` → convert thành `IndoorNode`
- Load tất cả `custom_edges` → convert thành `IndoorEdge`
- Tự động infer node_type từ tên (phòng, sảnh, cầu thang, cổng...)
- Map road_type → edge_type (stairs, elevator, corridor, door)
- Tìm nearest node cho mỗi edge endpoint (tolerance ~5m)
- Register vào `indoor_registry` với building_id = "main_building"

**Helper functions:**
- `_infer_node_type()` - phát hiện loại node từ tên
- `_map_road_type_to_edge_type()` - map loại đường
- `_find_nearest_node_id()` - tìm node gần nhất

### 2. ✅ Smart Indoor Detection
**File:** `routing/router.py` - method `find_route()`

**Logic mới (2 priority levels):**

**Priority 1:** Check database locations
```python
origin_loc = await db.find_location_by_coords(origin_lat, origin_lon)
dest_loc = await db.find_location_by_coords(dest_lat, dest_lon)

if origin_loc and dest_loc:
    # Cả 2 điểm trong database → chắc chắn indoor
    return await indoor_route()
```

**Priority 2:** Check GPS accuracy (fallback)
```python
elif gps_accuracy_m > threshold:
    # GPS kém → có thể indoor
    return await indoor_route()
```

**Lợi ích:**
- Tự động detect khi tìm đường giữa 2 địa điểm trong database
- Không cần user chỉ định "indoor" hay "outdoor"
- Fallback về outdoor routing nếu indoor fail

### 3. ✅ Database Method mới
**File:** `core/database.py`

Thêm method `find_location_by_coords()`:
```python
async def find_location_by_coords(lat, lon, tolerance=0.0001):
    # Tìm location trong bán kính ~10m
    # Dùng để detect xem điểm có trong database không
```

### 4. ✅ Startup Integration
**File:** `web/app.py` - `startup()` event

**2 methods load indoor data:**

**Method 1:** Load từ `floor_maps` table (GeoJSON format)
- Cho buildings có floor plans đầy đủ

**Method 2:** Build từ `locations` + `custom_edges`
- Tự động build nếu không có floor_maps
- Gọi `build_indoor_graph_from_db("main_building")`
- Log số nodes và edges đã build

### 5. ✅ Enhanced IndoorEdge
**File:** `core/indoor_router.py`

Thêm properties cho environmental analysis:
```python
@dataclass
class IndoorEdge:
    ...
    is_covered: bool = False      # có mái che không
    surface: str = "concrete"     # bề mặt (tile, concrete, grass...)
    slope_deg: float = 0.0        # độ dốc
```

Chuẩn bị cho P2 - tích hợp environmental factors.

## Kết quả

### Trước khi fix:
```
Tìm đường: bếp → sảnh tầng 4
Kết quả: 0 km, "đã đến điểm đến" ❌
```

### Sau khi fix:
```
Tìm đường: bếp → sảnh tầng 4
Phát hiện: Cả 2 điểm trong database
→ Trigger indoor routing
→ Build graph: 8 nodes, 14 edges (bidirectional)
→ A* tìm đường:
   bếp (tầng 1) 
   → nhà xe (tầng 1) [4.5m]
   → cầu thang 1-2 [7.1m]
   → sảnh tầng 2 (tầng 2)
   → cầu thang 2-3 [8.0m]
   → sảnh tầng 3 (tầng 3)
   → cầu thang 3-4 [12.9m]
   → sảnh tầng 4 (tầng 4) ✅

Tổng: ~32.5m, ~2-3 phút
```

## Test

### 1. Khởi động server
```bash
python main.py serve
```

**Expected log:**
```
Indoor: built graph from DB with 8 nodes, 14 edges
LocalNavBot v2 ready
```

### 2. Test routing API
```bash
curl -X POST http://localhost:8000/api/route \
  -H "Content-Type: application/json" \
  -d '{
    "origin": "bếp",
    "destination": "sảnh tầng 4"
  }'
```

**Expected response:**
```json
{
  "ok": true,
  "distance_km": 0.03,
  "duration_min": 2.5,
  "analysis": {
    "strategy": "indoor_astar",
    "building_id": "main_building",
    "floors_visited": [1, 2, 3, 4],
    "summary": "Tầng 1 → Tầng 2 → Tầng 3 → Tầng 4 → sảnh tầng 4"
  },
  "steps": [
    {"instruction": "Đi theo hành lang — 4 m đến nhà xe", ...},
    {"instruction": "Đi lên cầu thang nhà xe → Tầng 2", ...},
    ...
  ]
}
```

### 3. Test từ web UI
1. Mở http://localhost:8000
2. Nhập: "Từ: bếp" → "Đến: sảnh tầng 4"
3. Click "Tìm đường"
4. Xem kết quả hiển thị đường đi qua các tầng

## Còn thiếu (TODO)

### P2 - Environmental Factors (Medium priority)
Tích hợp nắng/mưa/đông đúc vào indoor routing:

**Cần sửa:** `_edge_cost()` trong `indoor_router.py`
```python
def _edge_cost(e: IndoorEdge, depart_time, weather, crowd) -> float:
    base = ...  # current logic
    
    # Crowd penalty
    if e.edge_type == "corridor":
        base *= (1 + 0.3 * crowd)
    
    # Weather penalty
    if weather > 0.5 and not e.is_covered:
        base *= 1.5  # mưa + không mái che
    
    # Surface penalty when wet
    if weather > 0.3 and e.surface == "tile":
        base *= 1.2  # gạch trơn khi mưa
    
    return base
```

**Cần thêm:**
- Pass `depart_time`, `weather_severity`, `crowd_level` vào `IndoorRouter.route()`
- Update `_try_indoor_route()` để truyền params này

### P3 - UI Improvements (Low priority)
- Hiển thị floor transitions rõ ràng hơn
- 3D visualization cho multi-floor routes
- Better error messages khi không tìm thấy đường

## Files đã sửa

1. ✅ `core/indoor_router.py` - thêm `build_indoor_graph_from_db()` và helpers
2. ✅ `core/database.py` - thêm `find_location_by_coords()`
3. ✅ `routing/router.py` - smart indoor detection trong `find_route()`
4. ✅ `web/app.py` - gọi build graph trong startup event
5. ✅ `ROUTING_ISSUES_ANALYSIS.md` - phân tích chi tiết vấn đề
6. ✅ `ROUTING_FIX_SUMMARY.md` - tóm tắt giải pháp (file này)
