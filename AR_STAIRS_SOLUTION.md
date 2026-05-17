# 🎯 Giải pháp AR Navigation cho Cầu thang

## Vấn đề hiện tại

### 1. GPS không hoạt động trong nhà/cầu thang
- **Accuracy:** 5-50m outdoor → 100-500m indoor → **không có tín hiệu** trong cầu thang kín
- **AR hiện tại:** Chỉ dùng GPS coords → **FAIL hoàn toàn** trong cầu thang

### 2. VIO có nhưng không được dùng
- **File:** `core/vio_fusion.py` - VIO engine hoàn chỉnh với EKF
- **Vấn đề:** Không được tích hợp vào `build_ar_path()` và AR rendering
- **Kết quả:** VIO track position nhưng AR vẫn dùng GPS → mất đồng bộ

### 3. Floor detection không feed vào AR
- **File:** `core/floor_detector.py` - detect tầng bằng barometer + step pattern
- **Vấn đề:** AR không biết user đang ở tầng nào
- **Kết quả:** Khi xuống cầu thang, AR vẫn hiển thị waypoints tầng cũ

### 4. Không có transition markers
- **Vấn đề:** Khi vào cầu thang, không có visual cue rõ ràng
- **Kết quả:** User bối rối, không biết đi đúng hướng

## Giải pháp

### Phase 1: VIO Integration (P0 - CRITICAL)

#### 1.1. Enhance `build_ar_path()` với VIO fallback

**File:** `core/route_projection.py`

```python
def build_ar_path(
    route: Route,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float = 0.0,
    min_spacing_m: float = 8.0,
    vio_pose: VIOPose | None = None,  # NEW
    use_vio: bool = False,             # NEW
) -> dict:
    """
    Build AR path with VIO fallback for indoor navigation.
    
    When use_vio=True and vio_pose is provided:
    - Use VIO position instead of GPS
    - Project route from VIO coords
    - Mark as "vio_mode" for client
    """
    if use_vio and vio_pose and vio_pose.origin_lat:
        # Use VIO position as reference
        ref_lat, ref_lon = vio_pose.to_latlon() or (ref_lat, ref_lon)
        
    local_points = route_to_local_frame(route.geometry, ref_lat, ref_lon, ref_alt)
    
    # ... existing sampling logic ...
    
    return {
        "reference": {"lat": ref_lat, "lon": ref_lon, "alt": ref_alt},
        "points": sampled,
        "point_count": len(sampled),
        "source_geometry_points": len(local_points),
        "vio_mode": use_vio,  # NEW - tell client to use VIO
        "vio_drift_m": vio_pose.drift_m if vio_pose else 0.0,  # NEW
    }
```

#### 1.2. Auto-switch GPS ↔ VIO

**Logic:**
```python
def should_use_vio(gps_accuracy_m: float, indoor: bool, floor: int) -> bool:
    """Decide whether to use VIO instead of GPS."""
    if indoor or floor > 1:
        return True  # Always use VIO indoors
    if gps_accuracy_m > 20:
        return True  # GPS too poor
    return False
```

#### 1.3. Update AR rendering to use VIO

**File:** `web/static/js/ar.js`

```javascript
function _arUpdatePosition(lat, lon, heading, floor, vioMode) {
    if (!window.AREnhanced) return;
    
    // If VIO mode, get VIO pose from server
    if (vioMode && window._vioState) {
        const vioPose = window._vioState;
        // Convert VIO ENU to lat/lon
        lat = vioPose.lat;
        lon = vioPose.lon;
        heading = vioPose.heading_deg;
    }
    
    AREnhanced.setUserPose(lat, lon, heading, floor);
}
```

### Phase 2: Floor-Aware AR (P0 - CRITICAL)

#### 2.1. Filter waypoints by floor

**File:** `core/route_projection.py`

```python
def build_ar_path_floor_aware(
    route: Route,
    current_floor: int,
    ref_lat: float,
    ref_lon: float,
    ...
) -> dict:
    """
    Build AR path showing only waypoints on current floor + next floor.
    
    When on stairs:
    - Show waypoints on current floor (fading out)
    - Show waypoints on target floor (fading in)
    - Show "floor transition" marker
    """
    # Get route steps
    steps = route.steps
    
    # Filter by floor
    current_floor_steps = []
    next_floor_steps = []
    transition_step = None
    
    for i, step in enumerate(steps):
        if step.from_floor == current_floor:
            current_floor_steps.append(step)
            
            # Check if this is a floor transition
            if step.to_floor != step.from_floor:
                transition_step = step
                # Get steps on next floor
                for j in range(i+1, len(steps)):
                    if steps[j].from_floor == step.to_floor:
                        next_floor_steps.append(steps[j])
                break
    
    # Build AR points
    points = []
    
    # Current floor points (full opacity)
    for step in current_floor_steps:
        points.append({
            "lat": step.lat,
            "lon": step.lon,
            "floor": step.from_floor,
            "opacity": 1.0,
            "type": "waypoint",
        })
    
    # Transition marker (if exists)
    if transition_step:
        points.append({
            "lat": transition_step.lat,
            "lon": transition_step.lon,
            "floor": transition_step.from_floor,
            "opacity": 1.0,
            "type": "stairs" if transition_step.edge_type == "stairs" else "elevator",
            "target_floor": transition_step.to_floor,
            "instruction": f"{'Lên' if transition_step.to_floor > transition_step.from_floor else 'Xuống'} tầng {transition_step.to_floor}",
        })
    
    # Next floor points (reduced opacity - preview)
    for step in next_floor_steps[:3]:  # Only show first 3
        points.append({
            "lat": step.lat,
            "lon": step.lon,
            "floor": step.from_floor,
            "opacity": 0.3,  # Faded
            "type": "preview",
        })
    
    return {
        "points": points,
        "current_floor": current_floor,
        "has_transition": transition_step is not None,
        "transition_type": transition_step.edge_type if transition_step else None,
    }
```

#### 2.2. Floor transition UI

**File:** `web/static/js/ar.js`

```javascript
function _arShowFloorTransition(transitionType, targetFloor, currentFloor) {
    const overlay = document.getElementById('ar-floor-transition');
    if (!overlay) return;
    
    const direction = targetFloor > currentFloor ? 'LÊN' : 'XUỐNG';
    const icon = transitionType === 'stairs' ? '🪜' : '🛗';
    const floorDelta = Math.abs(targetFloor - currentFloor);
    
    overlay.innerHTML = `
        <div class="floor-transition-card">
            <div class="floor-transition-icon">${icon}</div>
            <div class="floor-transition-text">
                <div class="floor-transition-action">${direction} ${floorDelta} TẦNG</div>
                <div class="floor-transition-target">→ Tầng ${targetFloor}</div>
            </div>
            <div class="floor-transition-arrow">
                ${direction === 'LÊN' ? '⬆️' : '⬇️'}
            </div>
        </div>
    `;
    
    overlay.classList.add('show');
    
    // Auto-hide after 5s
    setTimeout(() => overlay.classList.remove('show'), 5000);
}
```

### Phase 3: Stair-Specific Features (P1 - HIGH)

#### 3.1. Step counter for stairs

**Use barometer + accelerometer:**
```python
# In floor_detector.py - already has step detection!
def get_stair_progress(self) -> dict:
    """Get progress through current staircase."""
    if not self._is_on_stairs():
        return {"on_stairs": False}
    
    # Estimate floors climbed from steps
    # Typical: 15-20 steps per floor
    steps_per_floor = 17
    floors_climbed = self._step_count / steps_per_floor
    
    return {
        "on_stairs": True,
        "steps": self._step_count,
        "floors_climbed": round(floors_climbed, 2),
        "confidence": self._confidence,
    }
```

#### 3.2. AR visual cues for stairs

**3D arrow pointing up/down:**
```javascript
// In ar_enhanced.js
function _renderStairArrow(direction, distance) {
    // Create 3D arrow mesh
    const geometry = new THREE.ConeGeometry(0.3, 1.0, 8);
    const material = new THREE.MeshBasicMaterial({
        color: direction === 'up' ? 0x00ff00 : 0xff0000,
        opacity: 0.8,
        transparent: true,
    });
    const arrow = new THREE.Mesh(geometry, material);
    
    // Position at stair entrance
    arrow.position.set(0, 0, -distance);
    
    // Rotate based on direction
    if (direction === 'up') {
        arrow.rotation.x = -Math.PI / 2;  // Point up
    } else {
        arrow.rotation.x = Math.PI / 2;   // Point down
    }
    
    // Animate (bob up/down)
    arrow.userData.animate = (time) => {
        arrow.position.y = Math.sin(time * 2) * 0.2 + 1.5;
    };
    
    return arrow;
}
```

#### 3.3. Haptic feedback

**Vibrate when approaching stairs:**
```javascript
function _arVibrateStairWarning() {
    if ('vibrate' in navigator) {
        // Pattern: short-long-short
        navigator.vibrate([100, 50, 200, 50, 100]);
    }
}
```

### Phase 4: VPR Relocalization (P2 - MEDIUM)

#### 4.1. Auto-trigger VPR at floor transitions

**When VIO drift > 2m at stair exit:**
```python
async def _ar_check_relocalization(vio: VIOFusion, camera_frame: bytes):
    """Auto-trigger VPR when VIO drifts too much."""
    if vio.drift_m > 2.0:
        # Capture frame and run VPR
        from core.vpr_engine import vpr_engine
        
        match = await vpr_engine.query_image(camera_frame)
        if match and match.confidence > 0.7:
            # Relocalize VIO
            vio.relocalize(
                match.lat,
                match.lon,
                heading_deg=None,  # Keep current heading
                accuracy_m=match.accuracy_m,
            )
            return {"relocated": True, "location": match.name}
    
    return {"relocated": False}
```

#### 4.2. Visual markers at stair exits

**QR codes or AR markers:**
- Place QR codes at each stair exit
- Scan → instant relocalization
- Format: `navbot://floor/3/exit/north`

## Implementation Plan

### Week 1: VIO Integration (P0)
- [ ] Update `build_ar_path()` with VIO support
- [ ] Add auto-switch logic GPS ↔ VIO
- [ ] Update AR.js to use VIO position
- [ ] Test indoor tracking accuracy

### Week 2: Floor-Aware AR (P0)
- [ ] Implement `build_ar_path_floor_aware()`
- [ ] Add floor transition UI
- [ ] Filter waypoints by floor
- [ ] Test multi-floor navigation

### Week 3: Stair Features (P1)
- [ ] Add step counter integration
- [ ] Implement 3D stair arrows
- [ ] Add haptic feedback
- [ ] Test on real stairs

### Week 4: VPR Relocalization (P2)
- [ ] Auto-trigger VPR at transitions
- [ ] Add QR code support
- [ ] Test drift correction
- [ ] Performance optimization

## Testing Checklist

### Indoor Navigation
- [ ] VIO tracks position accurately (< 1m drift per 50m)
- [ ] AR waypoints visible and stable
- [ ] No jitter or jumping

### Stair Navigation
- [ ] Floor transition detected correctly
- [ ] UI shows clear "up/down" instruction
- [ ] Waypoints switch to new floor smoothly
- [ ] VIO maintains tracking through stairs

### Edge Cases
- [ ] GPS → VIO switch is seamless
- [ ] VIO → GPS switch when exiting building
- [ ] Handles elevator (no step pattern)
- [ ] Recovers from VIO drift via VPR

## Expected Results

### Before Fix
- ❌ AR fails completely in stairs (no GPS)
- ❌ User gets lost between floors
- ❌ Waypoints show wrong floor

### After Fix
- ✅ AR works smoothly in stairs (VIO)
- ✅ Clear floor transition UI
- ✅ Waypoints filtered by floor
- ✅ Auto-relocalization when needed
- ✅ < 1m accuracy through multi-floor route

## Files to Modify

1. ✅ `core/route_projection.py` - VIO integration
2. ✅ `web/static/js/ar.js` - VIO position update
3. ✅ `web/static/js/ar_enhanced.js` - floor-aware rendering
4. ✅ `web/routes/navigation.py` - pass VIO state to AR
5. ✅ `core/floor_detector.py` - expose stair progress
6. ⚠️ `web/ui.html` - add floor transition overlay

## Conclusion

**Bạn hoài nghi đúng!** AR hiện tại sẽ **FAIL hoàn toàn** trong cầu thang vì:
1. GPS không hoạt động
2. VIO có nhưng không được dùng
3. Không có floor awareness

Giải pháp trên sẽ fix tất cả vấn đề này bằng cách:
1. Tích hợp VIO vào AR pipeline
2. Filter waypoints theo tầng
3. Thêm UI rõ ràng cho floor transitions
4. Auto-relocalization khi cần
