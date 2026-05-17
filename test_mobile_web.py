"""
test_mobile_web.py — Test cases cho mobile web frontend
Chạy: pytest test_mobile_web.py -v

Test này KHÔNG cần chạy app thật, chỉ test logic validation và phát hiện lỗi tiềm ẩn.
"""
import pytest
import re
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# TEST CASES — STATIC CODE ANALYSIS (không cần chạy app)
# ══════════════════════════════════════════════════════════════════════════════

def test_JS01_check_load_order_dependencies():
    """JS-01, JS-02, JS-03: Kiểm tra dependencies giữa các JS files."""
    web_dir = Path("web/static/js")
    if not web_dir.exists():
        pytest.skip("web/static/js không tồn tại")
    
    # Đọc gps.js
    gps_js = (web_dir / "gps.js").read_text(encoding="utf-8")
    # Kiểm tra _arOn được dùng nhưng không khai báo
    assert "_arOn" in gps_js, "gps.js dùng _arOn"
    assert "let _arOn" not in gps_js, "gps.js không khai báo _arOn → phụ thuộc ar.js"
    
    # Kiểm tra floorState
    assert "floorState" in gps_js, "gps.js dùng floorState"
    
    # Đọc ar.js
    ar_js = (web_dir / "ar.js").read_text(encoding="utf-8")
    assert "_lastArPath" in ar_js, "ar.js dùng _lastArPath"
    assert "let _lastArPath" not in ar_js, "ar.js không khai báo _lastArPath → phụ thuộc route.js"


def test_JS05_pending_bot_msg_not_reset_on_reconnect():
    """JS-05: pendingBotMsg không được reset khi WebSocket reconnect."""
    ws_js = Path("web/static/js/websocket.js").read_text(encoding="utf-8")
    # Tìm onclose handler
    assert "ws.onclose" in ws_js
    # Kiểm tra có reset pendingBotMsg không
    onclose_match = re.search(r"ws\.onclose\s*=\s*\(\)\s*=>\s*\{([^}]+)\}", ws_js, re.DOTALL)
    if onclose_match:
        onclose_body = onclose_match.group(1)
        assert "pendingBotMsg" not in onclose_body, "BUG: pendingBotMsg không được reset trong onclose"


def test_JS06_ws_send_without_null_check():
    """JS-06: ws.send() không check ws !== null."""
    chat_js = Path("web/static/js/chat.js").read_text(encoding="utf-8")
    # Tìm ws.send()
    assert "ws.send(" in chat_js
    # Kiểm tra có check ws && ws.readyState === 1 trước không
    lines = chat_js.split("\n")
    for i, line in enumerate(lines):
        if "ws.send(" in line:
            # Kiểm tra 5 dòng trước có check ws không
            context = "\n".join(lines[max(0, i-5):i])
            if "ws && ws.readyState" not in context and "if (ws" not in context:
                pytest.fail(f"BUG JS-06: ws.send() ở dòng {i+1} không check ws !== null")


def test_JS11_ws_send_when_connecting():
    """JS-11: ws.send() khi readyState=0 (CONNECTING) sẽ throw."""
    chat_js = Path("web/static/js/chat.js").read_text(encoding="utf-8")
    # Tìm ws.send()
    send_matches = re.findall(r"ws\.send\([^)]+\)", chat_js)
    assert len(send_matches) > 0, "Không tìm thấy ws.send()"
    
    # Kiểm tra có check readyState === 1 không
    for match in send_matches:
        # Lấy context xung quanh
        idx = chat_js.index(match)
        context = chat_js[max(0, idx-200):idx]
        if "readyState === 1" not in context and "readyState == 1" not in context:
            pytest.fail(f"BUG JS-11: {match} không check readyState === 1")


def test_MOB03_barometer_api_not_standard():
    """MOB-04: DevicePressureEvent không phải Web API chuẩn."""
    floor_js = Path("web/static/js/floor.js").read_text(encoding="utf-8")
    assert "DevicePressureEvent" in floor_js, "floor.js dùng DevicePressureEvent"
    # DevicePressureEvent không tồn tại trong Web API → sẽ throw
    # Cần có try-catch
    assert "try" in floor_js or "catch" in floor_js, "Cần có try-catch cho DevicePressureEvent"


def test_MOB07_video_play_autoplay_blocked():
    """MOB-07: video.play() bị block autoplay trên iOS."""
    camera_js = Path("web/static/js/camera.js").read_text(encoding="utf-8")
    # Tìm v.play()
    assert "v.play()" in camera_js or ".play()" in camera_js
    # Kiểm tra có try-catch không
    play_matches = re.findall(r"(await\s+)?v\.play\(\)", camera_js)
    for match in play_matches:
        idx = camera_js.index(match)
        context = camera_js[max(0, idx-100):idx+100]
        if "try" not in context and "catch" not in context:
            pytest.fail("BUG MOB-07: v.play() không có try-catch → autoplay bị block trên iOS")


def test_API02_gps_update_no_lat_lon_validation():
    """API-01: /api/gps không validate lat/lon range."""
    nav_py = Path("web/routes/navigation.py").read_text(encoding="utf-8")
    # Tìm GPSUpdateRequest
    assert "class GPSUpdateRequest" in nav_py
    # Kiểm tra có Field(ge=, le=) cho lat/lon không
    gps_req_match = re.search(r"class GPSUpdateRequest.*?(?=class|\Z)", nav_py, re.DOTALL)
    if gps_req_match:
        gps_req_body = gps_req_match.group(0)
        if "lat:" in gps_req_body:
            # Kiểm tra có ge=-90, le=90 không
            if "ge=-90" not in gps_req_body or "le=90" not in gps_req_body:
                pytest.fail("BUG API-01: GPSUpdateRequest.lat không validate range [-90, 90]")


def test_API10_startup_blocks_event_loop():
    """API-10: _build_vpr() trong startup không async → block event loop."""
    app_py = Path("web/app.py").read_text(encoding="utf-8")
    # Tìm startup function
    startup_match = re.search(r"@app\.on_event\(['\"]startup['\"]\)\s*async def startup\(\):(.*?)(?=\n@|\ndef\s|\Z)", app_py, re.DOTALL)
    if startup_match:
        startup_body = startup_match.group(1)
        # Kiểm tra có _build_vpr() không
        if "_build_vpr()" in startup_body:
            # Kiểm tra có await không
            if "await _build_vpr()" not in startup_body:
                pytest.fail("BUG API-10: _build_vpr() không async → block event loop khi load model")


def test_SEC05_websocket_uses_ws_not_wss():
    """SEC-05: WebSocket dùng ws:// thay wss:// khi không có HTTPS."""
    ws_js = Path("web/static/js/websocket.js").read_text(encoding="utf-8")
    # Tìm WebSocket URL construction
    assert "location.protocol" in ws_js
    assert "wss" in ws_js and "ws" in ws_js, "WebSocket URL phụ thuộc protocol"
    # Kiểm tra có dùng wss khi https không
    if "location.protocol === 'https:' ? 'wss' : 'ws'" not in ws_js:
        pytest.fail("BUG SEC-05: WebSocket không dùng wss:// khi HTTPS")


def test_UX01_toast_only_3_seconds():
    """UX-01: Toast chỉ hiện 3s → thông báo lỗi biến mất quá nhanh."""
    globals_js = Path("web/static/js/globals.js").read_text(encoding="utf-8")
    # Tìm toast function
    toast_match = re.search(r"function toast\([^)]+\)\s*\{([^}]+)\}", globals_js, re.DOTALL)
    if toast_match:
        toast_body = toast_match.group(1)
        # Kiểm tra timeout
        timeout_match = re.search(r"setTimeout\([^,]+,\s*(\d+)\)", toast_body)
        if timeout_match:
            timeout_ms = int(timeout_match.group(1))
            assert timeout_ms == 3000, f"Toast timeout = {timeout_ms}ms"
            # Với lỗi quan trọng, nên hiện lâu hơn
            pytest.fail("BUG UX-01: Toast chỉ hiện 3s → lỗi quan trọng biến mất quá nhanh")


def test_NET04_no_fetch_timeout():
    """NET-04: fetch() không có timeout → có thể hang mãi."""
    js_files = ["route.js", "camera.js", "chat.js"]
    for js_file in js_files:
        js_path = Path(f"web/static/js/{js_file}")
        if not js_path.exists():
            continue
        js_content = js_path.read_text(encoding="utf-8")
        if "fetch(" in js_content:
            # Kiểm tra có AbortController hoặc timeout không
            if "AbortController" not in js_content and "signal:" not in js_content:
                pytest.fail(f"BUG NET-04: {js_file} dùng fetch() không có timeout")


def test_PERF_captured_frames_limit_30():
    """Kiểm tra capturedFrames giới hạn 30 để tránh RAM crash."""
    camera_js = Path("web/static/js/camera.js").read_text(encoding="utf-8")
    # Tìm check giới hạn
    assert "capturedFrames.length >= 30" in camera_js, "Có giới hạn 30 frames"
    # Kiểm tra có toast warning không
    limit_match = re.search(r"if\s*\(capturedFrames\.length\s*>=\s*30\)\s*\{([^}]+)\}", camera_js, re.DOTALL)
    if limit_match:
        limit_body = limit_match.group(1)
        assert "toast" in limit_body, "Có thông báo khi đạt giới hạn"


# ══════════════════════════════════════════════════════════════════════════════
# TEST CASES — FILE EXISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def test_FILE01_all_js_files_exist():
    """Kiểm tra tất cả JS files được reference trong ui.html tồn tại."""
    ui_html = Path("web/ui.html")
    if not ui_html.exists():
        pytest.skip("ui.html không tồn tại")
    
    html_content = ui_html.read_text(encoding="utf-8")
    # Tìm tất cả <script src="...">
    script_matches = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', html_content)
    
    for script_src in script_matches:
        if script_src.startswith("http"):
            continue  # External CDN
        # Relative path
        script_path = Path("web") / script_src.lstrip("/")
        assert script_path.exists(), f"JS file không tồn tại: {script_path}"


def test_FILE02_all_css_files_exist():
    """Kiểm tra tất cả CSS files tồn tại."""
    ui_html = Path("web/ui.html")
    if not ui_html.exists():
        pytest.skip("ui.html không tồn tại")
    
    html_content = ui_html.read_text(encoding="utf-8")
    css_matches = re.findall(r'<link[^>]+href=["\']([^"\']+\.css)["\']', html_content)
    
    for css_href in css_matches:
        if css_href.startswith("http"):
            continue
        css_path = Path("web") / css_href.lstrip("/")
        assert css_path.exists(), f"CSS file không tồn tại: {css_path}"


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
