# 🔍 Phân tích vấn đề thuật toán tìm đường

## Vấn đề báo cáo
**Triệu chứng:** Tìm đường từ "bếp" (tầng 1) đến "sảnh tầng 4" (tầng 4) → kết quả: 0 km, "đã đến điểm đến"

## Nguyên nhân gốc rễ

### 1. **Indoor Graph chưa được build từ database**
**File:** `core/indoor_router.py`
- `IndoorBuildingRegistry` tồn tại nhưng **RỖNG**
- Không có code nào load `locations` và `custom_edges` từ database vào `indoor_registry`
- Database có:
  - 8 locations (ID 1-8) trên các tầng 1-4
  - 7 edges kết nối các điểm, bao gồm cầu thang liên tầng
- Nhưng `indoor_registry._graphs` = {} (empty dict)

**Kết quả:** Indoor router không có dữ liệu để tìm đường!

### 2. **Logic routing không tự động detect indoor**
**File:** `routing/router.py` dòng 963-968

```python
if gps_accuracy_m > settings.indoor_gps_accuracy_threshold_m:
    indoor_result = await self._try_indoor_route(...)
```

**Vấn đề:**
- Chỉ trigger indoor routing khi GPS accuracy kém (>threshold)
- Khi tìm đường từ web form, `gps_accuracy_m` mặc định = 5m
- `indoor_gps_accuracy_threshold_m` có thể = 20m hoặc 50m
- → 5m < 20m → **KHÔNG trigger indoor routing**
- → Fallback về OSM outdoor router
- OSM router không có dữ liệu indoor → **không tìm thấy đường**

### 3. **Không có logic detect multi-floor routing**
**File:** `routing/router.py`

Thuật toán hiện tại:
1. Check GPS accuracy → nếu kém thì thử indoor
2. Nếu không indoor → dùng OSM outdoor router
3. OSM router chỉ có 2D roads, không có floor info

**Thiếu:**
- Không check xem origin và destination có cùng building không
- Không check xem có cần đi qua nhiều tầng không
- Không tự động chọn indoor router khi cả 2 điểm đều trong database locations

### 4. **Environmental factors chưa được tích hợp vào indoor routing**
**File:** `core/indoor_router.py` dòng 42-48

```python
def _edge_cost(e: IndoorEdge) -> float:
    # Chỉ tính: stairs time, elevator time, walk time
    # KHÔNG có: crowd penalty, weather penalty, covered bonus
```

**Thiếu:**
- Không xét đến đông đúc (crowd_level)
- Không xét đến thời tiết (weather_severity)
- Không ưu tiên đường có mái che khi trời mưa
- Không tránh đường trơn/dốc khi mưa

## Database hiện tại

```
LOCATIONS (8 điểm):
ID 1: phòng 303 (Tầng 3)
ID 2: phòng 303 (Tầng 3) - duplicate?
ID 3: sảnh tầng 3 (Tầng 3)
ID 4: sảnh tầng 4 (Tầng 4) ← DESTINATION
ID 5: sảng tầng 2 (Tầng 2)
ID 6: nhà xe (Tầng 1)
ID 7: cổng (Tầng 1)
ID 8: bếp (Tầng 1) ← ORIGIN

EDGES (7 đường):
ID 1: Đường đi bộ (1→1) 1.7m
ID 7: hành lang (1→1) 1.8m - nhà xe → cổng
ID 8: hành lang bếp (1→1) 4.5m - bếp → nhà xe
ID 9: Cầu thang 1-2 (1→2) 7.1m - nhà xe → sảnh tầng 2
ID 10: Cầu thang 2-3 (2→3) 8.0m - sảnh tầng 2 → sảnh tầng 3
ID 11: hành lang tầng ba (3→3) 5.1m - sảnh tầng 3 → phòng 303
ID 12: Cầu thang 3-4 (4→3) 12.9m - sảnh tầng 4 → sảnh tầng 3 (NGƯỢC!)
```

**Đường đi đúng:** bếp(8) → nhà xe(6) → tầng 2(5) → tầng 3(3) → tầng 4(4)
- Edge 8: bếp → nhà xe (4.5m)
- Edge 9: nhà xe → tầng 2 (7.1m)
- Edge 10: tầng 2 → tầng 3 (8.0m)
- Edge 12 (reverse): tầng 3 → tầng 4 (12.9m)
- **Tổng: ~32.5m, ~2-3 phút**

## Giải pháp cần triển khai

### 1. **Build Indoor Graph từ Database** (URGENT)
Tạo function `build_indoor_graph_from_db()`:
- Load tất cả locations từ database
- Convert thành IndoorNode
- Load tất cả custom_edges
- Convert thành IndoorEdge
- Add vào indoor_registry với building_id = "main_building"
- Gọi function này khi khởi động server

### 2. **Smart Indoor Detection**
Thay đổi logic trong `find_route()`:
```python
# OLD: chỉ check GPS accuracy
if gps_accuracy_m > threshold:
    try_indoor()

# NEW: check cả location trong database
origin_loc = await db.find_location_by_coords(origin_lat, origin_lon, tolerance=0.0001)
dest_loc = await db.find_location_by_coords(dest_lat, dest_lon, tolerance=0.0001)

if origin_loc and dest_loc:
    # Cả 2 điểm đều trong database → chắc chắn là indoor
    return await indoor_route(origin_loc, dest_loc)
elif gps_accuracy_m > threshold:
    # GPS kém → có thể indoor
    try_indoor()
```

### 3. **Tích hợp Environmental Factors vào Indoor**
Cập nhật `_edge_cost()` trong `indoor_router.py`:
```python
def _edge_cost(e: IndoorEdge, depart_time: datetime, weather: float, crowd: float) -> float:
    base_cost = ...  # current logic
    
    # Crowd penalty
    if e.edge_type == "corridor":
        base_cost *= (1 + 0.3 * crowd)  # đông đúc → chậm hơn
    
    # Weather penalty for uncovered paths
    if weather > 0.5 and not e.is_covered:
        base_cost *= 1.5  # mưa + không mái che → tránh
    
    # Surface penalty when wet
    if weather > 0.3 and e.surface in ("tile", "concrete"):
        base_cost *= 1.2  # trơn khi mưa
    
    return base_cost
```

### 4. **Fix Edge Direction**
Edge ID 12 có vấn đề:
- Tên: "Cầu thang 3-4"
- Nhưng: from_floor=4, to_floor=3 (ngược!)
- Cần sửa hoặc đảm bảo bidirectional=True

## Priority

1. **P0 - CRITICAL:** Build indoor graph từ database (không có thì không tìm được đường)
2. **P1 - HIGH:** Smart indoor detection (tự động detect khi cả 2 điểm trong DB)
3. **P2 - MEDIUM:** Environmental factors (nắng/mưa/đông đúc)
4. **P3 - LOW:** UI improvements, better error messages

## Files cần sửa

1. `core/indoor_router.py` - thêm `build_indoor_graph_from_db()`
2. `routing/router.py` - sửa logic `find_route()` và `_try_indoor_route()`
3. `core/database.py` - thêm `find_location_by_coords()`
4. `main.py` hoặc `web/app.py` - gọi build graph khi startup
