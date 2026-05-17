# 🧪 Test Routing Fix - Hướng dẫn kiểm tra

## Chuẩn bị

### 1. Kiểm tra database có dữ liệu
```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/navbot.db')
cursor = conn.cursor()

print('=== LOCATIONS ===')
cursor.execute('SELECT id, name, floor FROM locations ORDER BY id')
for r in cursor.fetchall():
    print(f'  ID {r[0]}: {r[1]} (Tầng {r[2]})')

print('\n=== EDGES ===')
cursor.execute('SELECT id, name, from_floor, to_floor, distance_m FROM custom_edges ORDER BY id')
for r in cursor.fetchall():
    print(f'  ID {r[0]}: {r[1]} (Tầng {r[2]}→{r[3]}) {r[4]:.1f}m')

conn.close()
"
```

**Expected output:**
```
=== LOCATIONS ===
  ID 1: phòng 303 (Tầng 3)
  ID 2: phòng 303 (Tầng 3)
  ID 3: sảnh tầng 3 (Tầng 3)
  ID 4: sảnh tầng 4 (Tầng 4)
  ID 5: sảng tầng 2 (Tầng 2)
  ID 6: nhà xe (Tầng 1)
  ID 7: cổng (Tầng 1)
  ID 8: bếp (Tầng 1)

=== EDGES ===
  ID 1: Đường đi bộ (Tầng 1→1) 1.7m
  ID 7: hành lang (Tầng 1→1) 1.8m
  ID 8: hành lang bếp (Tầng 1→1) 4.5m
  ID 9: Cầu thang 1-2 (Tầng 1→2) 7.1m
  ID 10: Cầu thang 2-3 (Tầng 2→3) 8.0m
  ID 11: hành lang tầng ba (Tầng 3→3) 5.1m
  ID 12: Cầu thang 3-4 (Tầng 4→3) 12.9m
```

## Test Cases

### Test 1: Khởi động server và kiểm tra log

```bash
python main.py serve
```

**Kiểm tra log startup:**

✅ **PASS nếu thấy:**
```
Indoor: built graph from DB with 8 nodes, 14 edges
LocalNavBot v2 ready
```

❌ **FAIL nếu thấy:**
```
No indoor data found in database
```
hoặc
```
Indoor map pre-load: [error message]
```

### Test 2: API routing - tìm đường cùng tầng

**Request:**
```bash
curl -X POST http://localhost:8000/api/route \
  -H "Content-Type: application/json" \
  -d '{
    "origin": "bếp",
    "destination": "nhà xe"
  }'
```

**Expected response:**
```json
{
  "ok": true,
  "distance_km": 0.004,
  "duration_min": 0.1,
  "analysis": {
    "strategy": "indoor_astar",
    "building_id": "main_building",
    "floors_visited": [1]
  },
  "steps": [
    {
      "instruction": "Đi theo hành lang — 4 m đến nhà xe",
      "distance_m": 4.5,
      "lat": 10.903213,
      "lon": 106.795578
    }
  ]
}
```

✅ **PASS:** distance > 0, có steps, strategy = "indoor_astar"
❌ **FAIL:** distance = 0 hoặc "No route found"

### Test 3: API routing - tìm đường multi-floor (MAIN TEST)

**Request:**
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
  "duration_min": 2.0,
  "analysis": {
    "strategy": "indoor_astar",
    "building_id": "main_building",
    "floors_visited": [1, 2, 3, 4],
    "summary": "Tầng 1 → Tầng 2 → Tầng 3 → Tầng 4 → sảnh tầng 4"
  },
  "steps": [
    {"instruction": "Đi theo hành lang — 4 m đến nhà xe", ...},
    {"instruction": "Đi lên cầu thang nhà xe → Tầng 2", ...},
    {"instruction": "Đi lên cầu thang ... → Tầng 3", ...},
    {"instruction": "Đi lên cầu thang ... → Tầng 4", ...}
  ]
}
```

✅ **PASS:**
- `distance_km` > 0 (khoảng 0.03)
- `duration_min` > 0 (khoảng 2-3 phút)
- `floors_visited` = [1, 2, 3, 4]
- `steps` có ít nhất 4 bước (qua 3 cầu thang)
- `strategy` = "indoor_astar"

❌ **FAIL:**
- `distance_km` = 0
- "No route found"
- `strategy` != "indoor_astar" (nghĩa là dùng outdoor router)

### Test 4: Web UI - tìm đường từ form

1. Mở browser: http://localhost:8000
2. Trong form tìm đường:
   - **Từ:** bếp
   - **Đến:** sảnh tầng 4
3. Click "Tìm đường"

**Expected:**
- Hiển thị bản đồ với route
- Hiển thị thông tin:
  - Khoảng cách: ~30m
  - Thời gian: ~2-3 phút
  - Các bước đi qua tầng 1 → 2 → 3 → 4
- Không có lỗi "No route found"

### Test 5: Tìm đường với tọa độ GPS trực tiếp

**Request:**
```bash
curl -X POST http://localhost:8000/api/route \
  -H "Content-Type: application/json" \
  -d '{
    "origin_lat": 10.903213,
    "origin_lon": 106.795578,
    "dest_lat": 10.903193,
    "dest_lon": 106.795648
  }'
```

**Expected:**
- Tự động detect 2 điểm trong database
- Trigger indoor routing
- Trả về route hợp lệ

✅ **PASS:** strategy = "indoor_astar", có route
❌ **FAIL:** strategy != "indoor_astar" hoặc no route

### Test 6: Fallback to outdoor routing

**Request:** Tìm đường đến điểm KHÔNG có trong database
```bash
curl -X POST http://localhost:8000/api/route \
  -H "Content-Type: application/json" \
  -d '{
    "origin": "bếp",
    "destination_lat": 10.9,
    "destination_lon": 106.8
  }'
```

**Expected:**
- Không tìm thấy destination trong database
- Fallback về outdoor routing (OSM)
- strategy = "offline_heuristic_astar" hoặc "valhalla"

✅ **PASS:** Có route, strategy không phải "indoor_astar"
❌ **FAIL:** Crash hoặc error

## Debug

### Nếu không build được indoor graph

**Kiểm tra:**
```python
python -c "
import asyncio
from core.database import db
from core.indoor_router import build_indoor_graph_from_db

async def test():
    await db.init()
    graph = await build_indoor_graph_from_db('main_building')
    print(f'Nodes: {len(graph.nodes)}')
    print(f'Edges: {sum(len(adj) for adj in graph.adj.values())}')
    for nid, node in list(graph.nodes.items())[:3]:
        print(f'  {nid}: {node.name} (floor {node.floor})')

asyncio.run(test())
"
```

**Expected:**
```
Nodes: 8
Edges: 14
  loc_1: phòng 303 (floor 3)
  loc_2: phòng 303 (floor 3)
  loc_3: sảnh tầng 3 (floor 3)
```

### Nếu routing không trigger indoor

**Kiểm tra detection logic:**
```python
python -c "
import asyncio
from core.database import db

async def test():
    await db.init()
    
    # Test find_location_by_coords
    loc = await db.find_location_by_coords(10.903213, 106.795578, tolerance=0.0001)
    if loc:
        print(f'✅ Found: {loc[\"name\"]} (floor {loc.get(\"floor\", 1)})')
    else:
        print('❌ Not found - detection will fail!')

asyncio.run(test())
"
```

**Expected:**
```
✅ Found: bếp (floor 1)
```

### Nếu A* không tìm thấy đường

**Kiểm tra graph connectivity:**
```python
python -c "
import asyncio
from core.database import db
from core.indoor_router import build_indoor_graph_from_db, IndoorRouter

async def test():
    await db.init()
    graph = await build_indoor_graph_from_db('main_building')
    
    # Check if nodes exist
    origin = graph.find_node_by_name('bếp')
    dest = graph.find_node_by_name('sảnh tầng 4')
    
    if not origin:
        print('❌ Origin node not found!')
        return
    if not dest:
        print('❌ Dest node not found!')
        return
    
    print(f'✅ Origin: {origin.node_id} ({origin.name})')
    print(f'✅ Dest: {dest.node_id} ({dest.name})')
    
    # Try routing
    router = IndoorRouter(graph)
    route = router.route(origin.node_id, dest.node_id)
    
    if route:
        print(f'✅ Route found: {route.total_distance_m:.1f}m, {route.total_duration_s:.1f}s')
        print(f'   Floors: {route.floors_visited}')
        print(f'   Steps: {len(route.steps)}')
    else:
        print('❌ No route found - graph not connected!')

asyncio.run(test())
"
```

**Expected:**
```
✅ Origin: loc_8 (bếp)
✅ Dest: loc_4 (sảnh tầng 4)
✅ Route found: 32.5m, 150.0s
   Floors: [1, 2, 3, 4]
   Steps: 4
```

## Kết luận

**Tất cả tests PASS** → Fix thành công! ✅

**Có test FAIL** → Xem phần Debug ở trên để tìm nguyên nhân.
