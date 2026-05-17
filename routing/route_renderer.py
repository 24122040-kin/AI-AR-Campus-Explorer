"""
routing/route_renderer.py — Rich route card generator
Produces:
  - Illustrated turn-by-turn HTML card (embeddable in bot responses)
  - Folium map with route polyline, step markers, and photo popups
  - Traffic timeline bar (best/worst times)
  - ETA with congestion breakdown
"""
from __future__ import annotations
import json
import math
import base64
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import io

from loguru import logger

from routing.router import Route, RouteStep
from core.traffic_analyzer import TrafficAnalyzer


# ─────────────────────────────────────────────────────────────────────────────
# Step icon map (emoji → SVG path equivalent for HTML)
# ─────────────────────────────────────────────────────────────────────────────

MANEUVER_ICONS = {
    "depart":            "🚀",
    "turn_left":         "⬅️",
    "turn_right":        "➡️",
    "slight_left":       "↖️",
    "slight_right":      "↗️",
    "sharp_left":        "↩️",
    "sharp_right":       "↪️",
    "straight":          "⬆️",
    "u_turn":            "🔄",
    "arrive":            "🏁",
    "arrive_left":       "🏁",
    "arrive_right":      "🏁",
    "merge":             "🔀",
    "ramp":              "🛣️",
    "roundabout_enter":  "🔁",
    "ferry":             "⛴️",
}

CONGESTION_COLOR = {
    "thông thoáng": "#22c55e",
    "bình thường":  "#84cc16",
    "hơi đông":     "#f59e0b",
    "tắc nghẽn":    "#ef4444",
    "kẹt xe nặng":  "#dc2626",
}


# ─────────────────────────────────────────────────────────────────────────────
# HTML card renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_route_html(
    route: Route,
    analyzer: TrafficAnalyzer,
    show_images: bool = True,
    compact: bool = False,
) -> str:
    """
    Returns a self-contained HTML fragment with the full illustrated route card.
    Designed to be injected into the web UI chat bubble.
    """
    dist_km = route.total_distance_m / 1000
    mins = int(route.total_duration_min)
    hour = route.depart_time.hour
    weekday = route.depart_time.weekday()
    cong = analyzer.congestion_at(hour, weekday, route.origin[0], route.origin[1])
    status = analyzer._status_label(cong)
    status_color = CONGESTION_COLOR.get(status, "#888")
    best = analyzer.best_departure_window(hour, 2, weekday)

    steps_html = ""
    for i, step in enumerate(route.steps):
        icon = MANEUVER_ICONS.get(step.maneuver, "•")
        dist_str = f"{int(step.distance_m)}m" if step.distance_m < 1000 else f"{step.distance_m/1000:.1f}km"
        dur_str  = f"{int(step.duration_s/60)}min" if step.duration_s >= 60 else f"{int(step.duration_s)}s"

        img_html = ""
        if show_images and step.image_paths:
            imgs = step.image_paths[:2]
            img_html = '<div style="display:flex;gap:6px;margin-top:6px;flex-wrap:wrap">'
            for p in imgs:
                if Path(p).exists():
                    try:
                        import base64 as b64m
                        from PIL import Image
                        img = Image.open(p)
                        img.thumbnail((200, 150))
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=75)
                        b64 = b64m.b64encode(buf.getvalue()).decode()
                        img_html += (
                            f'<img src="data:image/jpeg;base64,{b64}" '
                            f'style="width:140px;height:95px;object-fit:cover;border-radius:6px;'
                            f'border:1px solid #e0e0e0" />'
                        )
                    except Exception:
                        pass
            img_html += "</div>"

        is_last = i == len(route.steps) - 1
        step_bg = "#f0fdf4" if is_last else "white"

        steps_html += f"""
        <div style="display:flex;gap:10px;padding:10px 12px;background:{step_bg};
                    border-radius:8px;margin-bottom:6px;border:1px solid #f0f0f0">
          <div style="font-size:20px;flex-shrink:0;width:28px;text-align:center">{icon}</div>
          <div style="flex:1;min-width:0">
            <div style="font-size:14px;font-weight:500;color:#1a1a2e;line-height:1.4">
              {step.instruction}
            </div>
            {"" if is_last else f'<div style="font-size:12px;color:#888;margin-top:2px">{dist_str} · {dur_str}</div>'}
            {img_html}
          </div>
        </div>"""

    # Traffic advisory
    advisory = ""
    if cong > 0.5:
        save = best["save_minutes"]
        if save > 3:
            advisory = f"""
            <div style="background:#fef3c7;border:1px solid #fcd34d;border-radius:8px;
                        padding:10px 14px;margin-bottom:12px;font-size:13px;color:#92400e">
              ⚠️ <b>{status}</b> lúc {hour:02d}:00 — Nếu đi lúc <b>{best["recommended_hour"]:02d}:00</b>
              có thể tiết kiệm ~{save} phút.
            </div>"""

    why_bits = []
    if route.analysis.get("selected_profile"):
        why_bits.append(f"Profile: <b>{route.analysis['selected_profile']}</b>")
    if route.analysis.get("avg_congestion") is not None:
        why_bits.append(f"Congestion TB: <b>{route.analysis.get('avg_congestion', 0):.2f}</b>")
    if route.analysis.get("landmark_density"):
        why_bits.append(f"Landmark density: <b>{route.analysis.get('landmark_density', 0):.2f}</b>")
    if route.analysis.get("custom_edge_ratio"):
        why_bits.append(f"Duong local: <b>{route.analysis.get('custom_edge_ratio', 0)*100:.0f}%</b>")
    if route.analysis.get("turn_count") is not None:
        why_bits.append(f"So lan re: <b>{route.analysis.get('turn_count', 0)}</b>")

    candidate_html = ""
    candidates = route.analysis.get("candidate_profiles") or []
    if candidates:
        rows = "".join(
            f"<div style='display:flex;justify-content:space-between;gap:10px'>"
            f"<span>{item.get('profile','?')}</span>"
            f"<span>{item.get('duration_min',0):.1f} phut</span>"
            f"<span>score {item.get('score',0):.2f}</span>"
            f"</div>"
            for item in candidates[:3]
        )
        candidate_html = f"<div style=\"margin-top:8px;font-size:12px;color:#475569;display:grid;gap:4px\">{rows}</div>"

    why_html = ""
    if why_bits:
        why_html = (
            "<div style=\"background:#eef6ff;border:1px solid #bfdbfe;border-radius:8px;"
            "padding:10px 14px;margin-bottom:12px;font-size:13px;color:#1e3a8a\">"
            "<div style=\"font-weight:600;margin-bottom:6px\">Vi sao chon tuyen nay</div>"
            f"<div>{' · '.join(why_bits)}</div>{candidate_html}</div>"
        )

    return f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                max-width:520px;background:#fff;border-radius:12px;
                border:1px solid #e8e8e8;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.08)">

      <!-- Header -->
      <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                  color:white;padding:14px 18px;display:flex;justify-content:space-between;align-items:center">
        <div>
          <div style="font-size:16px;font-weight:600">Tuyến đường</div>
          <div style="font-size:22px;font-weight:700;margin-top:2px">{dist_km:.1f} km</div>
          <div style="font-size:14px;opacity:.85">~{mins} phút · Xuất phát {route.depart_time.strftime("%H:%M")}</div>
        </div>
        <div style="text-align:right">
          <div style="background:{status_color};color:white;padding:4px 10px;
                      border-radius:20px;font-size:12px;font-weight:500">{status}</div>
          <div style="font-size:11px;opacity:.7;margin-top:6px">
            {route.depart_time.strftime("%d/%m/%Y")}
          </div>
        </div>
      </div>

      <!-- Body -->
      <div style="padding:14px 14px 8px">
        {advisory}
        {why_html}
        <div style="font-size:12px;color:#888;margin-bottom:10px;font-weight:500;text-transform:uppercase;letter-spacing:.05em">
          Hướng dẫn từng bước
        </div>
        {steps_html}
      </div>

      <!-- Footer -->
      <div style="padding:10px 18px 14px;border-top:1px solid #f0f0f0;
                  font-size:12px;color:#888;display:flex;justify-content:space-between">
        <span>📏 {route.total_distance_m/1000:.1f} km tổng</span>
        <span>⏱ ~{mins} phút</span>
        <span style="color:{status_color}">● {status}</span>
      </div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
# Folium map with route + photos
# ─────────────────────────────────────────────────────────────────────────────

def render_route_map(route: Route, analyzer: TrafficAnalyzer) -> str:
    """
    Return Folium map HTML with:
    - Route polyline coloured by congestion segments
    - Start/end markers
    - Step markers with popup showing instruction + photo
    - Traffic heatmap overlay
    """
    try:
        import folium
        from folium.plugins import HeatMap, AntPath
    except ImportError:
        return "<p>folium not installed</p>"

    center_lat = (route.origin[0] + route.destination[0]) / 2
    center_lon = (route.origin[1] + route.destination[1]) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles="OpenStreetMap",
    )

    # Animated route line
    if len(route.geometry) >= 2:
        coords = [[lat, lon] for lat, lon in route.geometry]
        AntPath(
            coords,
            color="#4f46e5",
            weight=5,
            opacity=0.85,
            delay=800,
            dash_array=[10, 30],
        ).add_to(m)

    # Start marker
    folium.Marker(
        route.origin,
        popup="Điểm xuất phát",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(m)

    # End marker
    folium.Marker(
        route.destination,
        popup="Điểm đến",
        icon=folium.Icon(color="red", icon="flag", prefix="fa"),
    ).add_to(m)

    # Step markers with photos
    for i, step in enumerate(route.steps):
        if step.lat == 0 and step.lon == 0:
            continue
        icon_char = MANEUVER_ICONS.get(step.maneuver, "•")

        img_tag = ""
        if step.image_paths:
            p = step.image_paths[0]
            if Path(p).exists():
                try:
                    import base64 as b64m
                    from PIL import Image
                    img = Image.open(p)
                    img.thumbnail((180, 120))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=70)
                    b64 = b64m.b64encode(buf.getvalue()).decode()
                    img_tag = f'<br><img src="data:image/jpeg;base64,{b64}" width="180" style="border-radius:6px;margin-top:6px"/>'
                except Exception:
                    pass

        popup_html = f"""
        <div style="font-family:sans-serif;max-width:200px">
          <div style="font-weight:600;font-size:14px">{icon_char} Bước {i+1}</div>
          <div style="color:#555;font-size:13px;margin-top:4px">{step.instruction}</div>
          {img_tag}
        </div>"""

        folium.CircleMarker(
            [step.lat, step.lon],
            radius=7,
            color="#4f46e5",
            fill=True,
            fill_color="#fff",
            fill_opacity=0.9,
            weight=2,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"Bước {i+1}",
        ).add_to(m)

    # Traffic heatmap overlay
    heat_data = analyzer.heatmap_data()
    if heat_data:
        heat_pts = [[d["lat"], d["lon"], d["intensity"]] for d in heat_data]
        HeatMap(
            heat_pts,
            name="Tắc nghẽn",
            radius=20,
            blur=15,
            max_zoom=16,
            gradient={0.0: "blue", 0.4: "lime", 0.7: "orange", 1.0: "red"},
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m._repr_html_()


# ─────────────────────────────────────────────────────────────────────────────
# Traffic timeline bar (24-hour chart as HTML)
# ─────────────────────────────────────────────────────────────────────────────

def render_traffic_timeline(analyzer: TrafficAnalyzer, weekday: int | None = None) -> str:
    """Return a compact 24-hour traffic bar chart as HTML."""
    if weekday is None:
        weekday = datetime.now().weekday()

    curve = analyzer.full_day_curve(weekday)
    bars = ""
    current_h = datetime.now().hour

    for item in curve:
        h = item["hour"]
        c = item["congestion"]
        pct = int(c * 100)
        color = CONGESTION_COLOR.get(item["status"], "#84cc16")
        is_now = h == current_h
        outline = "box-shadow:0 0 0 2px #4f46e5;" if is_now else ""

        bars += f"""
        <div title="{h:02d}:00 — {item['status']}" style="
          display:flex;flex-direction:column;align-items:center;flex:1;gap:2px">
          <div style="height:{max(4, pct//2)}px;width:100%;background:{color};
                      border-radius:2px;{outline}transition:height .3s"></div>
          <div style="font-size:10px;color:#888;writing-mode:vertical-rl;
                      transform:rotate(180deg);height:22px;overflow:hidden">
            {h if h % 6 == 0 else ""}
          </div>
        </div>"""

    return f"""
    <div style="font-family:sans-serif;padding:12px;background:#fff;
                border-radius:8px;border:1px solid #e8e8e8">
      <div style="font-size:12px;color:#888;margin-bottom:8px;font-weight:500">
        Tình trạng giao thông — {["T2","T3","T4","T5","T6","T7","CN"][weekday]}
      </div>
      <div style="display:flex;gap:1px;align-items:flex-end;height:60px">
        {bars}
      </div>
      <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:8px">
        {"".join(f'<span style="font-size:11px;color:{c}">● {k}</span>' for k, c in CONGESTION_COLOR.items())}
      </div>
    </div>
    """
