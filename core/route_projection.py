from __future__ import annotations

import math

from core.geo_ar import route_to_local_frame
from routing.router import Route


def _distance_enu(a: dict, b: dict) -> float:
    dx = a["east_m"] - b["east_m"]
    dy = a["north_m"] - b["north_m"]
    dz = a["up_m"] - b["up_m"]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def should_use_vio(gps_accuracy_m: float, indoor: bool, floor: int) -> bool:
    """
    Decide whether to use VIO instead of GPS for AR navigation.
    
    VIO is preferred when:
    - Indoors (GPS doesn't work)
    - On upper floors (GPS accuracy poor)
    - GPS accuracy > 20m (too unreliable)
    """
    if indoor or floor > 1:
        return True  # Always use VIO indoors or on upper floors
    if gps_accuracy_m > 20:
        return True  # GPS too poor
    return False


def build_ar_path(
    route: Route,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float = 0.0,
    min_spacing_m: float = 8.0,
    vio_pose=None,  # VIOPose | None
    use_vio: bool = False,
    current_floor: int = 1,
) -> dict:
    """
    Build AR path with VIO fallback for indoor navigation.
    
    When use_vio=True and vio_pose is provided:
    - Use VIO position as reference instead of GPS
    - Project route from VIO coords
    - Mark as "vio_mode" for client to use VIO updates
    
    When current_floor is provided:
    - Filter waypoints to show only current floor + next floor
    - Add floor transition markers
    """
    # Use VIO position as reference if available
    if use_vio and vio_pose and hasattr(vio_pose, 'to_latlon'):
        latlon = vio_pose.to_latlon()
        if latlon:
            ref_lat, ref_lon = latlon
    
    local_points = route_to_local_frame(route.geometry, ref_lat, ref_lon, ref_alt)
    if not local_points:
        return {
            "points": [],
            "point_count": 0,
            "vio_mode": use_vio,
            "current_floor": current_floor,
        }

    # Sample points with minimum spacing
    sampled = [local_points[0]]
    for point in local_points[1:]:
        if _distance_enu(sampled[-1], point) >= min_spacing_m:
            sampled.append(point)
    if sampled[-1]["index"] != local_points[-1]["index"]:
        sampled.append(local_points[-1])
    
    # Add floor info to points if route has floor data
    if hasattr(route, 'steps') and route.steps:
        for point in sampled:
            # Find corresponding step
            idx = point.get("index", 0)
            if idx < len(route.steps):
                step = route.steps[idx]
                point["floor"] = getattr(step, 'from_floor', current_floor)
                point["edge_type"] = getattr(step, 'edge_type', 'corridor')

    return {
        "reference": {"lat": ref_lat, "lon": ref_lon, "alt": ref_alt},
        "points": sampled,
        "point_count": len(sampled),
        "source_geometry_points": len(local_points),
        "vio_mode": use_vio,
        "vio_drift_m": vio_pose.drift_m if vio_pose and hasattr(vio_pose, 'drift_m') else 0.0,
        "current_floor": current_floor,
    }


def build_ar_path_floor_aware(
    route: Route,
    current_floor: int,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float = 0.0,
    vio_pose=None,
) -> dict:
    """
    Build AR path showing only waypoints on current floor + preview of next floor.
    
    When on stairs:
    - Show waypoints on current floor (full opacity)
    - Show floor transition marker (stairs/elevator)
    - Show preview of next floor waypoints (faded)
    """
    if not hasattr(route, 'steps') or not route.steps:
        # Fallback to regular AR path
        return build_ar_path(route, ref_lat, ref_lon, ref_alt, vio_pose=vio_pose, current_floor=current_floor)
    
    steps = route.steps
    
    # Find steps on current floor and floor transitions
    current_floor_steps = []
    next_floor_steps = []
    transition_step = None
    
    for i, step in enumerate(steps):
        step_from_floor = getattr(step, 'from_floor', 1)
        step_to_floor = getattr(step, 'to_floor', 1)
        
        if step_from_floor == current_floor:
            current_floor_steps.append(step)
            
            # Check if this is a floor transition
            if step_to_floor != step_from_floor:
                transition_step = step
                # Get steps on next floor
                for j in range(i+1, min(i+4, len(steps))):  # Next 3 steps
                    if getattr(steps[j], 'from_floor', 1) == step_to_floor:
                        next_floor_steps.append(steps[j])
                break
    
    # Build AR points
    points = []
    
    # Current floor points (full opacity)
    for step in current_floor_steps:
        points.append({
            "lat": step.lat,
            "lon": step.lon,
            "floor": getattr(step, 'from_floor', current_floor),
            "opacity": 1.0,
            "type": "waypoint",
            "instruction": step.instruction,
        })
    
    # Transition marker (if exists)
    if transition_step:
        edge_type = getattr(transition_step, 'edge_type', getattr(transition_step, 'maneuver', 'corridor'))
        target_floor = getattr(transition_step, 'to_floor', current_floor)
        direction = 'up' if target_floor > current_floor else 'down'
        
        points.append({
            "lat": transition_step.lat,
            "lon": transition_step.lon,
            "floor": getattr(transition_step, 'from_floor', current_floor),
            "opacity": 1.0,
            "type": "stairs" if 'stair' in edge_type.lower() else "elevator" if 'elev' in edge_type.lower() else "transition",
            "target_floor": target_floor,
            "direction": direction,
            "instruction": f"{'Lên' if direction == 'up' else 'Xuống'} tầng {target_floor}",
        })
    
    # Next floor points (reduced opacity - preview)
    for step in next_floor_steps:
        points.append({
            "lat": step.lat,
            "lon": step.lon,
            "floor": getattr(step, 'from_floor', current_floor + 1),
            "opacity": 0.3,  # Faded
            "type": "preview",
            "instruction": step.instruction,
        })
    
    return {
        "reference": {"lat": ref_lat, "lon": ref_lon, "alt": ref_alt},
        "points": points,
        "point_count": len(points),
        "current_floor": current_floor,
        "has_transition": transition_step is not None,
        "transition_type": getattr(transition_step, 'edge_type', None) if transition_step else None,
        "target_floor": getattr(transition_step, 'to_floor', None) if transition_step else None,
        "vio_mode": vio_pose is not None,
        "vio_drift_m": vio_pose.drift_m if vio_pose and hasattr(vio_pose, 'drift_m') else 0.0,
    }
