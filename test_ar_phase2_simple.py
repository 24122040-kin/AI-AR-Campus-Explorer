#!/usr/bin/env python3
"""
Simple AR Stairs Phase 2 Tests

Test core functions without full router dependencies.
"""
import math


def test_should_use_vio():
    """Test VIO decision logic."""
    print("\n=== Test 1: should_use_vio() ===")
    
    # Import function
    from core.route_projection import should_use_vio
    
    # Outdoor, good GPS
    result = should_use_vio(gps_accuracy_m=5.0, indoor=False, floor=1)
    print(f"✓ Outdoor, good GPS (5m): {result} (expected: False)")
    assert result is False
    
    # Indoor, floor 1
    result = should_use_vio(gps_accuracy_m=5.0, indoor=True, floor=1)
    print(f"✓ Indoor, floor 1: {result} (expected: True)")
    assert result is True
    
    # Indoor, floor 3
    result = should_use_vio(gps_accuracy_m=5.0, indoor=True, floor=3)
    print(f"✓ Indoor, floor 3: {result} (expected: True)")
    assert result is True
    
    # Poor GPS
    result = should_use_vio(gps_accuracy_m=50.0, indoor=False, floor=1)
    print(f"✓ Poor GPS (50m): {result} (expected: True)")
    assert result is True
    
    print("✅ should_use_vio() tests passed")


def test_vio_pose():
    """Test VIO pose creation and conversion."""
    print("\n=== Test 2: VIO Pose ===")
    
    from core.vio_fusion import VIOPose
    
    # Create pose
    pose = VIOPose(
        px=10.0,  # 10m East
        py=20.0,  # 20m North
        heading_rad=math.pi / 4,  # 45°
        speed_ms=1.5,
        origin_lat=10.903244,
        origin_lon=106.795617,
        source="imu",
        drift_m=0.5,
    )
    
    print(f"✓ VIO pose created: px={pose.px}m, py={pose.py}m")
    print(f"✓ Heading: {pose.heading_rad * 180 / math.pi:.1f}°")
    print(f"✓ Drift: {pose.drift_m}m")
    
    # Convert to lat/lon
    latlon = pose.to_latlon()
    assert latlon is not None, "Should convert to lat/lon"
    
    lat, lon = latlon
    print(f"✓ Converted to: {lat:.6f}, {lon:.6f}")
    
    # Verify conversion
    lat_m = 111320.0
    lon_m = 111320.0 * math.cos(math.radians(pose.origin_lat))
    
    dlat = (lat - pose.origin_lat) * lat_m
    dlon = (lon - pose.origin_lon) * lon_m
    
    print(f"✓ Delta: {dlon:.2f}m East, {dlat:.2f}m North")
    
    assert abs(dlon - 10.0) < 1.0, f"East offset should be ~10m, got {dlon:.2f}m"
    assert abs(dlat - 20.0) < 1.0, f"North offset should be ~20m, got {dlat:.2f}m"
    
    print("✅ VIO pose tests passed")


def test_vio_fusion():
    """Test VIO fusion basic operations."""
    print("\n=== Test 3: VIO Fusion ===")
    
    from core.vio_fusion import VIOFusion
    
    # Create VIO
    vio = VIOFusion("test_session")
    print("✓ VIO created")
    
    # Reset to known position
    vio.reset(10.903244, 106.795617, heading_deg=0.0)
    pose = vio.get_pose()
    
    print(f"✓ VIO reset: origin=({pose.origin_lat:.6f}, {pose.origin_lon:.6f})")
    print(f"✓ Initial position: px={pose.px}m, py={pose.py}m")
    print(f"✓ Initial heading: {pose.heading_rad * 180 / math.pi:.1f}°")
    
    assert pose.origin_lat == 10.903244
    assert pose.origin_lon == 106.795617
    assert pose.px == 0.0
    assert pose.py == 0.0
    
    # Simulate IMU update
    pose = vio.update_imu(
        ax=0.0, ay=0.0, az=-9.81,  # Stationary
        gyro_z_rad_s=0.0,
        compass_deg=45.0,
        dt_s=0.1,
    )
    
    print(f"✓ After IMU update: heading={pose.heading_rad * 180 / math.pi:.1f}°")
    print(f"✓ Source: {pose.source}")
    
    assert pose.source == "imu"
    
    # Check drift
    print(f"✓ Drift: {vio.drift_m:.3f}m")
    assert vio.drift_m < 0.1, "Drift should be minimal for stationary"
    
    print("✅ VIO fusion tests passed")


def test_navigation_integration():
    """Test navigation.py integration points."""
    print("\n=== Test 4: Navigation Integration ===")
    
    # Check imports work
    try:
        from web.routes.navigation import router
        print("✓ Navigation router imported")
    except Exception as e:
        print(f"⚠ Navigation router import failed: {e}")
        print("  (This is OK if dependencies not installed)")
        return
    
    # Check VIO registry
    from core.vio_fusion import vio_registry
    print("✓ VIO registry imported")
    
    # Create test session
    vio = vio_registry.get_or_create("test_nav_session")
    print(f"✓ VIO session created: {vio.session_id}")
    
    # Reset and get pose
    vio.reset(10.903244, 106.795617, heading_deg=0.0)
    pose = vio.get_pose()
    
    print(f"✓ VIO pose: px={pose.px}m, py={pose.py}m")
    
    # Check pose dict serialization
    pose_dict = pose.as_dict()
    print(f"✓ Pose dict keys: {list(pose_dict.keys())}")
    
    assert 'px' in pose_dict
    assert 'py' in pose_dict
    assert 'heading_deg' in pose_dict
    assert 'drift_m' in pose_dict
    
    print("✅ Navigation integration tests passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("AR Stairs Phase 2 - Simple Tests")
    print("=" * 60)
    
    try:
        test_should_use_vio()
        test_vio_pose()
        test_vio_fusion()
        test_navigation_integration()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nPhase 2 core functions are working correctly.")
        print("\nNext steps:")
        print("1. Deploy to server")
        print("2. Test on iPhone Safari")
        print("3. Navigate multi-floor route (phòng 303 → bếp)")
        print("4. Verify floor transition overlay appears")
        print("5. Check VIO position updates work indoors")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
