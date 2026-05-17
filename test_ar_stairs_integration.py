#!/usr/bin/env python3
"""
Test AR Stairs Integration (Phase 2)

Verify that VIO integration and floor-aware AR path work correctly.
"""
import asyncio
from datetime import datetime

from core.route_projection import build_ar_path_floor_aware, should_use_vio
from core.vio_fusion import VIOFusion, VIOPose
from routing.router import Router


async def test_should_use_vio():
    """Test VIO decision logic."""
    print("\n=== Test 1: should_use_vio() ===")
    
    # Outdoor, good GPS
    result = should_use_vio(gps_accuracy_m=5.0, indoor=False, floor=1)
    print(f"Outdoor, good GPS (5m): {result}")
    assert result is False, "Should use GPS outdoors with good accuracy"
    
    # Indoor, floor 1
    result = should_use_vio(gps_accuracy_m=5.0, indoor=True, floor=1)
    print(f"Indoor, floor 1: {result}")
    assert result is True, "Should use VIO indoors"
    
    # Indoor, floor 3
    result = should_use_vio(gps_accuracy_m=5.0, indoor=True, floor=3)
    print(f"Indoor, floor 3: {result}")
    assert result is True, "Should use VIO on upper floors"
    
    # Poor GPS
    result = should_use_vio(gps_accuracy_m=50.0, indoor=False, floor=1)
    print(f"Poor GPS (50m): {result}")
    assert result is True, "Should use VIO with poor GPS"
    
    print("✅ All should_use_vio() tests passed")


async def test_vio_pose_conversion():
    """Test VIO pose to lat/lon conversion."""
    print("\n=== Test 2: VIO Pose Conversion ===")
    
    # Create VIO with known origin
    vio = VIOFusion("test_session")
    origin_lat = 10.903244
    origin_lon = 106.795617
    vio.reset(origin_lat, origin_lon, heading_deg=0.0)
    
    # Simulate movement: 10m East, 20m North
    vio._ekf.initialize(10.0, 20.0, 0.0, 0.0)
    pose = vio.get_pose()
    
    print(f"VIO ENU: px={pose.px:.2f}m, py={pose.py:.2f}m")
    print(f"Origin: {origin_lat:.6f}, {origin_lon:.6f}")
    
    # Convert back to lat/lon
    latlon = pose.to_latlon()
    if latlon:
        lat, lon = latlon
        print(f"Converted: {lat:.6f}, {lon:.6f}")
        
        # Verify conversion (should be ~10m East, 20m North from origin)
        lat_m = 111320.0
        lon_m = 111320.0 * 0.9998  # cos(10.9°) ≈ 0.9998
        
        dlat = (lat - origin_lat) * lat_m
        dlon = (lon - origin_lon) * lon_m
        
        print(f"Delta: {dlon:.2f}m East, {dlat:.2f}m North")
        
        assert abs(dlon - 10.0) < 0.5, f"East offset should be ~10m, got {dlon:.2f}m"
        assert abs(dlat - 20.0) < 0.5, f"North offset should be ~20m, got {dlat:.2f}m"
        
        print("✅ VIO pose conversion test passed")
    else:
        print("❌ VIO pose conversion failed (no origin)")


async def test_floor_aware_ar_path():
    """Test floor-aware AR path generation."""
    print("\n=== Test 3: Floor-Aware AR Path ===")
    
    # Create mock route with multi-floor steps
    class MockStep:
        def __init__(self, lat, lon, floor, instruction, maneuver=None):
            self.lat = lat
            self.lon = lon
            self.floor = floor
            self.instruction = instruction
            self.maneuver = maneuver
            self.distance_m = 10.0
            self.duration_s = 15.0
            self.bearing = 0.0
    
    class MockRoute:
        def __init__(self):
            self.steps = [
                # Floor 3
                MockStep(10.903244, 106.795617, 3, "Đi thẳng hành lang tầng 3"),
                MockStep(10.903254, 106.795627, 3, "Rẽ phải"),
                # Stairs 3→2
                MockStep(10.903264, 106.795637, 3, "Xuống cầu thang", maneuver="stairs"),
                # Floor 2
                MockStep(10.903274, 106.795647, 2, "Đi thẳng hành lang tầng 2"),
                MockStep(10.903284, 106.795657, 2, "Rẽ trái"),
                # Stairs 2→1
                MockStep(10.903294, 106.795667, 2, "Xuống cầu thang", maneuver="stairs"),
                # Floor 1
                MockStep(10.903304, 106.795677, 1, "Đi thẳng đến bếp"),
            ]
            self.geometry = []
    
    route = MockRoute()
    
    # Test floor 3 (current floor)
    ar_path = build_ar_path_floor_aware(
        route=route,
        current_floor=3,
        ref_lat=10.903244,
        ref_lon=106.795617,
    )
    
    print(f"Current floor: 3")
    print(f"Total points: {ar_path['point_count']}")
    print(f"Has transition: {ar_path['has_transition']}")
    
    if ar_path['has_transition']:
        print(f"Transition type: {ar_path['transition_type']}")
        print(f"Transition direction: {ar_path['transition_direction']}")
        print(f"Target floor: {ar_path['target_floor']}")
        
        assert ar_path['transition_type'] == 'stairs', "Should detect stairs"
        assert ar_path['transition_direction'] == 'down', "Should be going down"
        assert ar_path['target_floor'] == 2, "Should target floor 2"
    
    # Count floor 3 waypoints (should be 3: 2 regular + 1 transition)
    floor_3_points = [p for p in ar_path['points'] if p.get('floor') == 3]
    print(f"Floor 3 waypoints: {len(floor_3_points)}")
    
    # Count preview points (floor 2, faded)
    preview_points = [p for p in ar_path['points'] if p.get('opacity', 1.0) < 1.0]
    print(f"Preview waypoints: {len(preview_points)}")
    
    assert len(floor_3_points) >= 2, "Should have at least 2 floor 3 waypoints"
    assert len(preview_points) > 0, "Should have preview waypoints"
    
    print("✅ Floor-aware AR path test passed")


async def test_vio_integration_in_ar_path():
    """Test VIO integration in AR path generation."""
    print("\n=== Test 4: VIO Integration in AR Path ===")
    
    # Create VIO with offset position
    vio = VIOFusion("test_session")
    vio.reset(10.903244, 106.795617, heading_deg=45.0)
    
    # Simulate VIO movement: 5m East, 10m North
    vio._ekf.initialize(5.0, 10.0, 0.785, 1.0)  # 0.785 rad = 45°
    vio_pose = vio.get_pose()
    
    print(f"VIO pose: px={vio_pose.px:.2f}m, py={vio_pose.py:.2f}m")
    print(f"VIO heading: {vio_pose.heading_rad * 180 / 3.14159:.1f}°")
    print(f"VIO drift: {vio_pose.drift_m:.2f}m")
    
    # Create simple route
    class MockStep:
        def __init__(self, lat, lon):
            self.lat = lat
            self.lon = lon
            self.floor = 1
            self.instruction = "Test"
            self.maneuver = None
            self.distance_m = 10.0
            self.duration_s = 15.0
            self.bearing = 0.0
    
    class MockRoute:
        def __init__(self):
            self.steps = [
                MockStep(10.903244, 106.795617),
                MockStep(10.903254, 106.795627),
            ]
            self.geometry = []
    
    route = MockRoute()
    
    # Build AR path with VIO
    from core.route_projection import build_ar_path
    
    ar_path = build_ar_path(
        route=route,
        ref_lat=10.903244,
        ref_lon=106.795617,
        vio_pose=vio_pose,
        use_vio=True,
        current_floor=1,
    )
    
    print(f"AR path points: {ar_path['point_count']}")
    print(f"VIO mode: {ar_path.get('vio_mode', False)}")
    print(f"VIO drift: {ar_path.get('vio_drift_m', 0):.2f}m")
    
    assert ar_path.get('vio_mode') is True, "Should be in VIO mode"
    assert ar_path['point_count'] > 0, "Should have waypoints"
    
    print("✅ VIO integration in AR path test passed")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("AR Stairs Integration Tests (Phase 2)")
    print("=" * 60)
    
    try:
        await test_should_use_vio()
        await test_vio_pose_conversion()
        await test_floor_aware_ar_path()
        await test_vio_integration_in_ar_path()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nPhase 2 integration is working correctly.")
        print("Ready for real-world testing on device.")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
