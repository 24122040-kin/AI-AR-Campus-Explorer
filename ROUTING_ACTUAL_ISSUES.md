# 🔍 Vấn đề thực tế với Indoor Routing

## Test Results

### ✅ Routing hoạt động ĐÚNG!

**Test case:** Phòng 303 (tầng 3) → Bếp (tầng 1)

**Kết quả:**
```
Distance: 24.7m
Duration: 48s (0.8 min)
Floors: [1, 2, 3]

Steps:
  1. Đi theo hành lang → sảnh tầng 3 [5.1m]
  2. Đi xuống cầu thang → Tầng 2 [8.0m]
  3. Đi xuống cầu thang → Tầng 1 [7.1m]
  4. Đi theo hành lang → bếp [4.5m]
```

**Đường đi:** Phòng 303 → Sảnh T3 → Cầu thang 3-2 → Sảnh T2 → Cầu thang 2-1 → Nhà xe → Bếp

✅ **ĐÚNG HOÀN TOÀN!** Không có "đi theo hành lang tầng 3" sai như bạn nói.

## ⚠️ Vấn đề phát hiện

### 1. Duplicate Location "phòng 303"

**Database có 2 locations cùng tên:**
- **loc_1:** phòng 303 (10.903244, 106.795617) - **ISOLATED, không có edge nào!**
- **loc_2:** phòng 303 (10.903233, 106.795589) - có edge 11 kết nối với sảnh tầng 3

**Nguyên nhân:**
- Edge 11 ("hành lang tầng ba") chỉ kết nối loc_2, không kết nối loc_1
- Có thể do:
  - Tạo nhầm 2 lần
  - Tọa độ edge không khớp với loc_1

**Giải pháp:**
1. **Xóa loc_1** (duplicate không dùng)
2. **Hoặc** thêm edge kết nối loc_1 với sảnh tầng 3

### 2. Edge 1 tạo self-loop

**Edge 1:** "Đường đi bộ" (tầng 1→1, 1.7m)
- From: (10.903240, 106.795522) - không match location nào
- To: (10.903255, 106.795525) - không match location nào
- Kết quả: tạo edge từ loc_7 (cổng) → loc_7 (cổng) (self-loop)

**Nguyên nhân:**
- Tọa độ edge không khớp với bất kỳ location nào
- `_find_nearest_node_id()` chọn loc_7 (cổng) vì gần nhất

**Giải pháp:**
- Xóa edge 1 (không cần thiết)
- Hoặc sửa tọa độ để match đúng locations

### 3. Tolerance quá nhỏ ban đầu

**Ban đầu:** tolerance = 0.00005 (~5m)
- Quá chặt → nhiều edges không match được nodes
- Dẫn đến isolated nodes

**Đã fix:** tolerance = 0.0002 (~20m)
- Vẫn chặt đủ để tránh match nhầm
- Nhưng linh hoạt hơn với GPS drift

## 🎯 AR Navigation

**Câu hỏi:** "Khi xuống cầu thang thì chức năng AR chỉ dẫn có hoạt động đúng không?"

**Trả lời:**

### AR hoạt động dựa trên:
1. **GPS position** - xác định vị trí hiện tại
2. **Compass bearing** - xác định hướng nhìn
3. **Route geometry** - điểm cần đến tiếp theo

### Vấn đề với cầu thang:
- **GPS không hoạt động trong nhà** (đặc biệt cầu thang kín)
- **Compass bị nhiễu** do cấu trúc kim loại
- **VIO (Visual-Inertial Odometry)** cần để tracking trong nhà

### Giải pháp:
1. **Dùng VIO thay GPS** khi trong nhà
   - File: `core/vio_fusion.py`
   - Tracking bằng camera + IMU
   
2. **Floor detection** tự động
   - File: `core/floor_detector.py`
   - Detect tầng bằng barometer + step counter

3. **AR markers** tại cầu thang
   - QR codes hoặc visual markers
   - Calibrate position khi scan

### Code hiện tại:
```python
# core/route_projection.py - build_ar_path()
# Tạo AR waypoints từ route geometry
# NHƯNG: chỉ dùng GPS coords, không có VIO fallback
```

**Kết luận:** AR sẽ **KHÔNG hoạt động tốt** trong cầu thang nếu chỉ dùng GPS. Cần tích hợp VIO.

## 📋 Action Items

### P0 - Critical (fix ngay)
1. ✅ **Xóa loc_1 (phòng 303 duplicate)**
   ```sql
   DELETE FROM locations WHERE id=1;
   ```

2. ✅ **Xóa edge 1 (self-loop không cần thiết)**
   ```sql
   DELETE FROM custom_edges WHERE id=1;
   ```

### P1 - High (cần sớm)
3. **Tích hợp VIO vào AR navigation**
   - Fallback từ GPS sang VIO khi indoor
   - Update `build_ar_path()` để dùng VIO coords

4. **Floor transition markers**
   - Thêm visual cues tại cầu thang
   - "Đang lên/xuống tầng X"

### P2 - Medium
5. **Improve edge matching**
   - Validate edge coords khi tạo
   - Warning nếu không match location nào

6. **Duplicate detection**
   - Check trùng tên + tầng khi thêm location
   - Suggest merge nếu tọa độ gần nhau

## 🧪 Verification

Sau khi xóa loc_1 và edge_1:

```bash
python test_indoor_graph.py
```

**Expected:**
- Nodes: 7 (giảm từ 8)
- Edges: 12 (giảm từ 14)
- Không có isolated nodes
- Không có self-loops
- Routing vẫn hoạt động đúng

## Kết luận

**Thuật toán routing ĐÚNG!** Vấn đề là:
1. ✅ Duplicate data (loc_1)
2. ✅ Bad edge data (edge_1)
3. ⚠️ AR cần VIO, không thể chỉ dùng GPS trong nhà

Bạn hoài nghi đúng về AR trong cầu thang - đó là vấn đề thực tế cần giải quyết bằng VIO.
