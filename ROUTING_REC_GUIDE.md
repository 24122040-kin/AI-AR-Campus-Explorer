# AI AR Campus — Core Engine Documentation

Dự án này là backend API được xây dựng bằng **FastAPI** nhằm cung cấp dịch vụ điều hướng và gợi ý địa điểm thông minh cho khuôn viên trường đại học (Campus) tích hợp AR. 

Hai thành phần lõi tạo nên sức mạnh của hệ thống là **Graph Pathfinding (Tìm đường tối ưu)** và **Semantic & Contextual Recommender (Hệ thống đề xuất ngữ nghĩa và ngữ cảnh)**.

---

## 🗺️ 1. Hệ thống Tìm đường Tối ưu (Graph Pathfinding)

Hệ thống điều hướng không chỉ đơn thuần tìm đường ngắn nhất, mà còn đánh giá tổng hợp các yếu tố về thời tiết, sự kiện, độ đông đúc và học máy (GNN).

### Cấu trúc Đồ thị (Graph Representation)
* **Thư viện sử dụng:** `NetworkX`.
* **Nodes (Đỉnh):** Đại diện cho các tòa nhà, phòng ban, tiện ích (ATM, Nhà xe). Chứa tọa độ GPS và các thuộc tính (điều hòa, độ ồn, giờ mở cửa).
* **Edges (Cạnh):** Đại diện cho các tuyến đường đi bộ. Trọng số cơ sở (Base Weight) là khoảng cách thực tế tính bằng mét (sử dụng công thức Haversine).

### Thuật toán Cốt lõi: A* Search
Hệ thống sử dụng thuật toán $A^*$ (`nx.astar_path`) để tối ưu hóa việc tìm đường.
* **Hàm Heuristic $h(n)$:** Khoảng cách Euclidean (`math.hypot`) từ node hiện tại đến đích, giúp thuật toán có "trực giác" định hướng thay vì duyệt mù.
* **Hàm Chi phí $g(n)$ (Custom Edge Cost):** Chi phí để đi qua một cạnh không chỉ là khoảng cách, mà được nội suy qua nhiều tham số động:

$$Cost(u, v) = \frac{Base\_Distance \times Weather\_Penalty \times Crowd\_Multiplier}{GAT\_Attention}$$

Trong đó:
1.  **Weather Penalty:** Nếu trời nắng gắt hoặc mưa (`weather in ["sunny", "rainy"]`) và đoạn đường không có mái che (`has_roof == False`), chi phí bị nhân lên gấp 5 lần, buộc thuật toán phải tìm đường đi vòng trong nhà hoặc có mái che.
2.  **Crowd Multiplier:** Trọng số tăng tuyến tính theo dự báo độ đông đúc tại 2 đầu mút của đoạn đường.
3.  **GAT Attention:** Trọng số ưu tiên học được từ Graph Neural Network (PyTorch Geometric). Tuyến đường "huyết mạch" sẽ có điểm Attention cao, làm giảm chi phí đi qua.
4.  **Trạng thái đóng/mở:** Nếu đoạn đường đang thi công (`status == "repairing"`), chi phí bị đẩy lên $999,999$ để thuật toán né tránh.

---

## 🧠 2. Hệ thống Đề xuất Thông minh (Recommendation System)

Hệ thống gợi ý được thiết kế nhiều lớp, từ xử lý ngôn ngữ tự nhiên (NLP) thuần túy đến gợi ý chủ động dựa trên ngữ cảnh không gian và thời gian.

### A. Đề xuất Ngữ nghĩa (Semantic AI)
Được module hóa trong class `CampusSemanticAI`, hệ thống xây dựng một không gian vector đại diện cho từng địa điểm mà không cần gọi API bên ngoài (OpenAI/Gemini).
* **Đầu vào:** Mô tả của người dùng (VD: *"tìm chỗ yên tĩnh có máy lạnh để học bài"*).
* **Phương pháp:** * Hệ thống gom tất cả metadata của một node (tên, bí danh, tag tiện ích, mô tả dịch vụ) thành một "tài liệu" (Document).
    * Tính toán ma trận **TF-IDF** cho toàn bộ đồ thị.
    * Khi có truy vấn, hệ thống tính toán **Độ tương đồng Cosine (Cosine Similarity)** giữa vector truy vấn ($Q$) và vector địa điểm ($D$):
    
    $$Similarity(Q, D) = \frac{Q \cdot D}{||Q|| \times ||D||}$$

### B. Luật Nhu cầu (Rule-based Intent Extraction)
Hệ thống quét các từ khóa trong câu hỏi để trích xuất `needs` (nhu cầu: máy lạnh, yên tĩnh, ăn uống, thể thao). Điểm số địa điểm sẽ được cộng/trừ gắt gao dựa trên các đặc trưng vật lý (features) của node đó. 

### C. Đề xuất Chủ động (Proactive & Context-Aware Recommender)
Hệ thống tự động đề xuất địa điểm ngay cả khi người dùng không hỏi, dựa trên "Ngữ cảnh Hiện tại". Hàm `get_smart_recommendations` đánh giá điểm số qua các khía cạnh:

1.  **Ngữ cảnh Thời gian (Time Context):** * *11:00 - 13:00:* Ưu tiên đề xuất căn tin.
    * *16:30 - 18:30:* Đề xuất nhà xe để chuẩn bị ra về.
2.  **Ngữ cảnh Lộ trình (Route Context):** * Hệ thống tính toán góc phương vị (Bearing) để biết một địa điểm có nằm thuận theo hướng đi tới đích hay không.
    * Ưu tiên những điểm cách vị trí hiện tại chỉ vài chục mét (Detour Bonus), giúp người dùng tiện đường ghé qua mà không bị mua đường.
3.  **Lọc Trạng thái (Open/Close Filters):** Các hàm gợi ý luôn kiểm tra `is_node_open(G, node, current_time)` để đảm bảo không bao giờ gợi ý một địa điểm đã đóng cửa.

---

## ⚙️ Các Module Quan trọng khác

* **Geofencing & Vùng Hạn chế (`optimizer.py`):** Liên tục kiểm tra tọa độ GPS của người dùng. Nếu bước vào phạm vi `< 30m` của một vùng `restricted` (khu hành chính, tòa nhà đang đóng cửa), hệ thống sẽ trả về cảnh báo nguy hiểm.
* **Inductive Learning (`gnn_engine.py`):** Cho phép thêm một địa điểm mới vào đồ thị (kèm vector embedding) ở runtime mà không cần train lại toàn bộ model GNN, giúp API mở rộng cực kỳ nhanh chóng.
