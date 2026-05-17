# ✅ Phát hiện đường trùng lặp khi tạo đường mới

## Vấn đề
Khi tạo đường mới giữa 2 điểm đã có đường, hệ thống không kiểm tra → tạo nhiều đường trùng lặp giữa cùng 2 điểm.

## Giải pháp đã triển khai

### 1. Database Methods (core/database.py)
Thêm 2 methods mới:

```python
async def find_edge_between_points(from_lat, from_lon, to_lat, to_lon, tolerance=0.00001)
```
- Tìm tất cả đường giữa 2 điểm (cả 2 chiều A→B và B→A)
- Tolerance ~1m để xử lý sai số GPS

```python
async def delete_edge(edge_id)
```
- Xóa đường theo ID

### 2. API Routes (web/routes/map_data.py)
Thêm 3 endpoints mới:

**POST /api/edge**
- Tạo đường mới
- Nhận: name, from_lat, from_lon, to_lat, to_lon, road_type, bidirectional, from_floor, to_floor, is_covered, surface, slope_deg, geometry
- Trả về: {ok, id, distance_m, message}

**GET /api/edge/find**
- Tìm đường giữa 2 điểm
- Query params: from_lat, from_lon, to_lat, to_lon
- Trả về: {ok, edges: [...]}

**DELETE /api/edge/{edge_id}**
- Xóa đường theo ID
- Trả về: {ok, message}

### 3. Frontend Logic (web/static/js/localmap.js)
Cập nhật hàm `saveRoad()`:

**Workflow mới:**
1. Kiểm tra xem đã có đường giữa 2 điểm chưa (gọi GET /api/edge/find)
2. Nếu có đường cũ:
   - Hiển thị dialog xác nhận với 2 lựa chọn:
     - **OK** → THAY THẾ đường cũ (xóa cũ, tạo mới)
     - **Cancel** → TẠO THÊM đường mới (giữ cũ, thêm mới)
3. Nếu chọn "Thay thế":
   - Gọi DELETE /api/edge/{id} để xóa đường cũ
   - Xóa khỏi mảng window.EDGES
4. Tạo đường mới (gọi POST /api/edge)
5. Thêm vào mảng window.EDGES và render lại

**Dialog hiển thị:**
```
⚠️ Đã có 1 đường giữa 2 điểm này:
"Hành lang A"

Bấm OK để THAY THẾ đường cũ
Bấm Cancel để TẠO THÊM đường mới
```

## Kết quả
- ✅ Phát hiện đường trùng lặp trước khi tạo
- ✅ Cho phép người dùng chọn: thay thế hoặc tạo thêm
- ✅ Tự động xóa đường cũ nếu chọn thay thế
- ✅ Cập nhật UI ngay lập tức
- ✅ Hiển thị thông báo rõ ràng: "Đã thay thế" hoặc "Đã tạo"

## Test
1. Tạo đường giữa 2 điểm A và B
2. Thử tạo đường mới giữa cùng 2 điểm A và B
3. Xem dialog xác nhận xuất hiện
4. Chọn "OK" → đường cũ bị thay thế
5. Hoặc chọn "Cancel" → tạo thêm đường mới (2 đường song song)

## Files đã sửa
- `core/database.py` - thêm 2 methods
- `web/routes/map_data.py` - thêm 3 API endpoints
- `web/static/js/localmap.js` - cập nhật hàm saveRoad()
