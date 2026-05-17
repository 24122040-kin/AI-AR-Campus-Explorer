from __future__ import annotations


class RealtimeNavigator:
    def build_instruction(self, scene_state: dict) -> dict:
        route_progress = scene_state.get("route_progress", {})
        visual = scene_state.get("visual", {})
        next_maneuver = route_progress.get("next_maneuver")
        distance_m = route_progress.get("distance_to_next_turn_m")
        off_route = route_progress.get("off_route")

        reasons: list[str] = []
        if off_route:
            headline = "Ban dang lech khoi tuyen, he thong se tim lai duong."
            urgency = "high"
            reasons.append("GPS/map matching cho thay khoang cach toi tuyen tang cao.")
        elif next_maneuver:
            headline = next_maneuver
            if distance_m is not None and distance_m > 0:
                headline = f"{next_maneuver} sau khoang {int(distance_m)} m."
            urgency = "high" if distance_m is not None and distance_m <= 50 else "normal"
        else:
            headline = "Tiep tuc di thang theo tuyen hien tai."
            urgency = "low"

        landmarks = [item.get("label") for item in visual.get("landmarks", []) if item.get("label")]
        if landmarks:
            reasons.append("Landmark thay duoc: " + ", ".join(landmarks[:3]))
        texts = [item.get("text") for item in visual.get("ocr_blocks", []) if item.get("text")]
        if texts:
            reasons.append("Text doc duoc: " + " | ".join(texts[:2]))
        vpr_hint = visual.get("vpr_hint")
        if vpr_hint and vpr_hint.get("location_name"):
            reasons.append(f"VPR goi y khu vuc {vpr_hint['location_name']}.")

        return {
            "instruction": headline,
            "short_instruction": headline,
            "reason": " ".join(reasons) if reasons else "Dang bam sat route va du lieu sensor hien tai.",
            "urgency": urgency,
        }
