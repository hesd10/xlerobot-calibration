#!/usr/bin/env python3
"""Bake calibration results into a fitted XML model.

Reads touch_zero.json (arm mount corrections T_B_A and wrist camera mounts
T_wrist_cam), head_zero.json (head camera extrinsics T_tilt_cam), and
intrinsics_*.json (camera intrinsics), then generates a new XML where:
  - Arm root bodies (Rotation_Pitch, Rotation_Pitch_2) have updated pos/quat
  - Head camera has updated pos/quat (converted to MuJoCo frame) and fovy
  - Wrist cameras have updated pos/quat (converted to MuJoCo frame) and fovy

The output XML is saved to xlerobot_calib_fitted.xml (original preserved).

Usage:
  python calibration/bake_xml.py [--output path/to/fitted.xml] [--symmetrize-camera]

Options:
  --output: Output XML path (default: model/xlerobot_calib_fitted.xml)
  --symmetrize-camera: Force camera Y=0 (symmetric) instead of using measured offset
"""

from __future__ import annotations
import sys
import argparse
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).parent))
from core import storage, se3, head_model
import model_map


def R_to_quat_mujoco(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to MuJoCo quaternion (w x y z)."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def opencv_to_mujoco_camera(T_opencv: np.ndarray) -> np.ndarray:
    """Convert OpenCV camera frame to MuJoCo camera frame.
    
    OpenCV: +X right, +Y down, +Z forward (optical axis)
    MuJoCo: +X right, +Y up, -Z forward (optical axis)
    
    Rotation: 180° about X axis.
    """
    R_conv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    T_mj = T_opencv.copy()
    T_mj[:3, :3] = T_opencv[:3, :3] @ R_conv
    return T_mj


def compute_fovy(fy: float, height: int) -> float:
    """Compute vertical field of view in degrees from focal length."""
    return float(np.rad2deg(2 * np.arctan(height / (2 * fy))))


def main():
    parser = argparse.ArgumentParser(description="Bake calibration into XML")
    parser.add_argument("--output", default="model/xlerobot_calib_fitted.xml",
                        help="Output XML path")
    parser.add_argument("--symmetrize-camera", action="store_true",
                        help="Force camera Y=0 (symmetric) instead of measured offset")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    input_xml = base_dir / "model" / "xlerobot_calib.xml"
    output_xml = base_dir / args.output
    
    # Load calibration results
    touch = storage.load_result("touch_zero")
    head = storage.load_result("head_zero")
    intr_head = storage.load_result("intrinsics_head")
    
    if not touch or not head or not intr_head:
        print("ERROR: Missing paired Stage 5b results. Run through Stage 5b first.")
        return 1
    if head.get("body_frame_id") != touch.get("body_frame_id"):
        print("ERROR: head_zero and touch_zero use different body frames. Re-run Stage 5b.")
        return 1
    
    print("=" * 70)
    print("Baking calibration into XML")
    print("=" * 70)
    print(f"Input:  {input_xml}")
    print(f"Output: {output_xml}")
    print()
    
    # Load sim model to get nominal arm root poses
    sim = model_map.SimModel()
    sim.set_joints({})
    
    # ===== ARM ROOTS =====
    print("### ARM ROOTS ###\n")
    arm_updates = {}
    
    for arm, body_name in [("left_arm", "Rotation_Pitch"),
                           ("right_arm", "Rotation_Pitch_2")]:
        if arm not in touch["arms"]:
            print(f"  {arm}: NOT CALIBRATED, skipping")
            continue
        
        T_B_A = np.array(touch["arms"][arm]["T_B_A"])
        p_nom_B, R_nom_B = sim.body_pose_in_chassis(model_map.ROOT_BODIES[arm])
        T_nom_B = se3.make_transform(R_nom_B, p_nom_B)

        # T_B_A and the nominal root pose are both expressed in chassis frame B.
        T_new_B = T_B_A @ T_nom_B
        p_new = T_new_B[:3, 3]
        q_new = R_to_quat_mujoco(T_new_B[:3, :3])
        
        arm_updates[body_name] = {"pos": p_new, "quat": q_new}
        
        print(f"  {body_name} ({arm}):")
        print(f"    pos:  {p_new[0]:.5f} {p_new[1]:.5f} {p_new[2]:.5f}")
        print(f"    quat: {q_new[0]:.6f} {q_new[1]:.6f} {q_new[2]:.6f} {q_new[3]:.6f}")
    
    baseline = np.linalg.norm(arm_updates["Rotation_Pitch"]["pos"] -
                              arm_updates["Rotation_Pitch_2"]["pos"]) * 1000
    print(f"\n  >>> New arm spacing: {baseline:.2f} mm\n")
    
    # ===== HEAD CAMERA =====
    print("### HEAD CAMERA ###\n")
    
    # head_model expresses T_tilt_cam in the pan-link coordinates, NOT in MuJoCo's
    # head_tilt_link body frame (which carries an extra quat). So we can't drop
    # T_tilt_cam straight into the camera tag. Instead: compute the camera's
    # absolute pose in the base (chassis) frame at zero head angles, then express
    # it relative to MuJoCo's actual head_tilt_link body.
    T_tilt_cam_opencv = np.array(head["T_tilt_cam"])
    senses = tuple(head["senses"])
    T_base_cam_ocv = head_model.T_base_cam(0.0, 0.0, T_tilt_cam_opencv,
                                           senses=senses)  # chassis frame, OpenCV

    # Optionally symmetrize: zero the camera's Y offset from the pan axis (Y=0 in
    # the base frame is the pan axis, so this centres the camera between the arms).
    if args.symmetrize_camera:
        print("  --symmetrize-camera: forcing base-frame Y=0")
        T_base_cam_ocv[1, 3] = 0.0

    # OpenCV optical frame -> MuJoCo camera frame (looks along -Z, +Y up).
    T_base_cam_mj = opencv_to_mujoco_camera(T_base_cam_ocv)

    # Express relative to MuJoCo's head_tilt_link. Load the ORIGINAL model so the
    # tilt-link pose is unaffected by anything we are about to write.
    import mujoco
    m0 = mujoco.MjModel.from_xml_path(str(input_xml))
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    tilt_id = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, "head_tilt_link")
    chassis_id = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    T_w_tilt = se3.make_transform(d0.xmat[tilt_id].reshape(3, 3), d0.xpos[tilt_id])
    T_w_chassis = se3.make_transform(d0.xmat[chassis_id].reshape(3, 3),
                                     d0.xpos[chassis_id])
    # base frame == chassis frame, so camera-in-chassis -> camera-in-world
    T_w_cam_mj = T_w_chassis @ T_base_cam_mj
    # relative to tilt link (what the camera tag needs)
    T_tilt_cam_mj = se3.invert(T_w_tilt) @ T_w_cam_mj

    cam_pos = T_tilt_cam_mj[:3, 3]
    cam_quat = R_to_quat_mujoco(T_tilt_cam_mj[:3, :3])
    
    # Compute fovy from intrinsics
    K = np.array(intr_head["K"])
    fy = K[1, 1]
    height = intr_head["height"]
    fovy = compute_fovy(fy, height)
    
    print(f"  head_camera:")
    print(f"    pos:  {cam_pos[0]:.5f} {cam_pos[1]:.5f} {cam_pos[2]:.5f}")
    print(f"    quat: {cam_quat[0]:.6f} {cam_quat[1]:.6f} {cam_quat[2]:.6f} {cam_quat[3]:.6f}")
    print(f"    fovy: {fovy:.2f}°  (was 58° in XML)")
    print()
    
    # ===== WRIST CAMERAS =====
    # Stage 6 solves T_wrist_cam as the camera mount relative to the Fixed_Jaw
    # body (see core/wrist_model.py), which is the same body the XML hangs the
    # wrist camera off, so the transform maps straight across. Only the optical
    # convention differs: the solve is in OpenCV axes, MuJoCo cameras are not.
    print("### WRIST CAMERAS ###\n")
    wrist_layout = [
        ("left_arm", "left_wrist_camera", "intrinsics_left_wrist"),
        ("right_arm", "right_wrist_camera", "intrinsics_right_wrist"),
    ]
    wrist_updates = {}
    for arm, camera_name, intrinsics_name in wrist_layout:
        arm_result = (touch.get("arms") or {}).get(arm) or {}
        mount = arm_result.get("T_wrist_cam")
        if mount is None:
            print(f"  {camera_name}: NOT CALIBRATED, skipping")
            continue
        T_mj = opencv_to_mujoco_camera(np.asarray(mount, float))
        update = {"pos": T_mj[:3, 3], "quat": R_to_quat_mujoco(T_mj[:3, :3])}
        intrinsics = storage.load_result(intrinsics_name)
        if intrinsics:
            update["fovy"] = compute_fovy(
                np.array(intrinsics["K"])[1, 1], intrinsics["height"])
        else:
            # Without intrinsics the pose is still worth baking; leaving the
            # nominal fovy is honest, inventing one is not.
            print(f"  {camera_name}: no intrinsics, keeping the XML fovy")
        wrist_updates[camera_name] = update
        p, q = update["pos"], update["quat"]
        print(f"  {camera_name}:")
        print(f"    pos:  {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}")
        print(f"    quat: {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
        if "fovy" in update:
            print(f"    fovy: {update['fovy']:.2f}deg  (was 58deg in XML)")
    print()

    # ===== MODIFY XML =====
    print("### WRITING XML ###\n")
    
    tree = ET.parse(input_xml)
    root = tree.getroot()
    
    # Update arm root bodies
    for body in root.iter("body"):
        name = body.get("name")
        if name in arm_updates:
            p = arm_updates[name]["pos"]
            q = arm_updates[name]["quat"]
            body.set("pos", f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
            body.set("quat", f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
            print(f"  ✓ Updated {name}")
    
    # Update head camera
    for camera in root.iter("camera"):
        if camera.get("name") == "head_camera":
            camera.set("pos", f"{cam_pos[0]:.6f} {cam_pos[1]:.6f} {cam_pos[2]:.6f}")
            camera.set("quat", f"{cam_quat[0]:.6f} {cam_quat[1]:.6f} "
                              f"{cam_quat[2]:.6f} {cam_quat[3]:.6f}")
            camera.set("fovy", f"{fovy:.4f}")
            print(f"  ✓ Updated head_camera")

    # Update wrist cameras
    for camera in root.iter("camera"):
        update = wrist_updates.get(camera.get("name"))
        if update is None:
            continue
        p, q = update["pos"], update["quat"]
        camera.set("pos", f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
        camera.set("quat", f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
        if "fovy" in update:
            camera.set("fovy", f"{update['fovy']:.4f}")
        print(f"  ✓ Updated {camera.get('name')}")
    
    # Remove occluding head shell geom (tophead6) so camera can see through
    removed_occluders = []
    for body in root.iter("body"):
        for geom in list(body.findall("geom")):
            if geom.get("mesh") == "tophead6":
                body.remove(geom)
                removed_occluders.append("tophead6")
    if removed_occluders:
        print(f"  ✓ Removed occluding head shell: {', '.join(set(removed_occluders))}")
    
    # Write output
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    print(f"\n  Saved to: {output_xml}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Load the fitted XML in MuJoCo to verify geometry")
    print("  2. For real2sim: use zeros.json and senses.json in model_map.py")
    print("  3. Undistort real images using intrinsics_*.json distortion coefficients")
    print("  4. Note: MuJoCo doesn't handle lens distortion or principal point offset")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
