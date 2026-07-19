import json
import os
import math
import numpy as np
from PIL import Image
import open3d as o3d
from sklearn.linear_model import RANSACRegressor
from scipy.ndimage import binary_erosion

def process_dataset(dataset_dir):
    json_path = os.path.join(dataset_dir, 'transforms.json')
    if not os.path.exists(json_path):
        print(f"Dataset not found at {json_path}")
        return

    with open(json_path, 'r') as f:
        transforms = json.load(f)

    frames = transforms.get('frames', [])
    if not frames:
        print("No frames found in transforms.json")
        return

    voxel_grid = {}

    def get_chunk_and_voxel(x_w, y_w, z_w):
        gx = math.floor(x_w / 0.00625)
        gy = math.floor(y_w / 0.00625)
        gz = math.floor(z_w / 0.00625)
        return (gx, gy, gz)

    smoothed_s = 1.0
    smoothed_t = 0.0
    is_first_frame = True
    calibration_frames = 0
    s_final = 1.0
    t_final = 0.0
    
    ransac = RANSACRegressor(residual_threshold=0.05, max_trials=500)

    for i, frame in enumerate(frames):
        print(f"Processing frame {i}...")
        
        file_path = frame['file_path']
        img_path = os.path.join(dataset_dir, file_path)
        base_path = os.path.join(dataset_dir, os.path.splitext(file_path)[0])
        
        ai_depth_path = base_path + "_ai_depth.bin"
        tof_depth_path = base_path + "_tof_depth.bin"
        tof_conf_path = base_path + "_tof_conf.bin"
        slam_path = base_path + "_slam.bin"
        
        if not all(os.path.exists(p) for p in [ai_depth_path, tof_depth_path, tof_conf_path, slam_path]):
            print(f"Missing raw data files for {file_path}. Did you export the raw dataset properly?")
            continue
            
        try:
            img = np.array(Image.open(img_path).convert('RGB'))
        except Exception:
            img = None
            
        ai_depth = np.fromfile(ai_depth_path, dtype=np.float32).reshape((896, 504))
        
        depth_w = frame.get('depth_w', 160)
        depth_h = frame.get('depth_h', 120)
        
        tof_depth = np.fromfile(tof_depth_path, dtype=np.uint16).reshape((depth_h, depth_w))
        tof_conf = np.fromfile(tof_conf_path, dtype=np.uint8).reshape((depth_h, depth_w))
        slam_data = np.fromfile(slam_path, dtype=np.float32)
        
        fx, fy = frame['fl_x'], frame['fl_y']
        cx, cy = frame['cx'], frame['cy']
        pose_matrix = np.array(frame['transform_matrix']) # 4x4
        inv_pose = np.linalg.inv(pose_matrix)
        
        # 1. EXTRACT TOF ANCHORS (Vectorized)
        aspect = depth_w / float(depth_h)
        target = 16.0 / 9.0
        if aspect >= target:
            depth_w_cropped = depth_h * target
            depth_h_cropped = depth_h
            offset_x = (depth_w - depth_w_cropped) / 2.0
            offset_y = 0.0
        else:
            depth_w_cropped = depth_w
            depth_h_cropped = depth_w / target
            offset_x = 0.0
            offset_y = (depth_h - depth_h_cropped) / 2.0

        scale_x = 896.0 / depth_w_cropped
        scale_y = 504.0 / depth_h_cropped
        
        u, v = np.meshgrid(np.arange(depth_w), np.arange(depth_h))
        
        cropped_u = (u + 0.5) - offset_x
        cropped_v = (v + 0.5) - offset_y
        
        mask = (tof_conf >= 150) & (cropped_u >= 0) & (cropped_u < depth_w_cropped) & \
               (cropped_v >= 0) & (cropped_v < depth_h_cropped)
                
        valid_u = cropped_u[mask]
        valid_v = cropped_v[mask]
        z_metric = tof_depth[mask] / 1000.0
        
        u_land = valid_u * scale_x
        v_land = valid_v * scale_y
        
        u_ai = 504.0 - 1.0 - v_land
        v_ai = u_land
        
        tx = np.round(u_ai).astype(int)
        ty = np.round(v_ai).astype(int)
        
        bounds_mask = (tx >= 0) & (tx < 504) & (ty >= 0) & (ty < 896)
        tx = tx[bounds_mask]
        ty = ty[bounds_mask]
        u_ai = u_ai[bounds_mask]
        v_ai = v_ai[bounds_mask]
        z_metric = z_metric[bounds_mask]
        
        z_ai_lin = ai_depth[ty, tx]
        tof_anchors = np.column_stack((u_ai, v_ai, z_ai_lin, z_metric)) if len(z_metric) > 0 else np.empty((0, 4))
                        
        # 2. EXTRACT SLAM ANCHORS (Vectorized)
        slam_anchors = np.empty((0, 4))
        if len(slam_data) > 0:
            slam_points_w = slam_data.reshape(-1, 4).copy()
            slam_points_w[:, 3] = 1.0
            
            pt_c = (inv_pose @ slam_points_w.T).T
            
            x_p, y_p, z_p = pt_c[:, 0], pt_c[:, 1], pt_c[:, 2]
            z_true = -z_p
            
            valid_mask = z_true > 0.1
            x_p = x_p[valid_mask]
            y_p = y_p[valid_mask]
            z_true = z_true[valid_mask]
            
            if len(z_true) > 0:
                u_ai = (x_p * fx / z_true) + cx
                v_ai = cy - (y_p * fy / z_true)
                
                tx = np.round(u_ai).astype(int)
                ty = np.round(v_ai).astype(int)
                
                bounds_mask = (tx >= 0) & (tx < 504) & (ty >= 0) & (ty < 896)
                tx = tx[bounds_mask]
                ty = ty[bounds_mask]
                u_ai = u_ai[bounds_mask]
                v_ai = v_ai[bounds_mask]
                z_true = z_true[bounds_mask]
                
                if len(z_true) > 0:
                    z_ai_lin = ai_depth[ty, tx]
                    slam_anchors = np.column_stack((u_ai, v_ai, z_ai_lin, z_true))

        # 3. EXTRACT MODEL ANCHORS (Vectorized Ray-Marching)
        model_anchors = []
        if calibration_frames >= 5 and len(voxel_grid) > 0:
            ty_march = np.arange(0, 896, 16)
            tx_march = np.arange(0, 504, 16)
            tx_grid, ty_grid = np.meshgrid(tx_march, ty_march)
            tx_flat = tx_grid.flatten()
            ty_flat = ty_grid.flatten()
            
            x_p = (tx_flat - cx) / fx
            y_p = -(ty_flat - cy) / fy
            z_p = np.full_like(x_p, -1.0)
            
            dx = pose_matrix[0, 0]*x_p + pose_matrix[0, 1]*y_p + pose_matrix[0, 2]*z_p
            dy = pose_matrix[1, 0]*x_p + pose_matrix[1, 1]*y_p + pose_matrix[1, 2]*z_p
            dz = pose_matrix[2, 0]*x_p + pose_matrix[2, 1]*y_p + pose_matrix[2, 2]*z_p
            
            ox, oy, oz = pose_matrix[0, 3], pose_matrix[1, 3], pose_matrix[2, 3]
            
            z_steps = np.arange(0.1, 3.5, 0.005)
            x_w_all = ox + np.outer(dx, z_steps)
            y_w_all = oy + np.outer(dy, z_steps)
            z_w_all = oz + np.outer(dz, z_steps)
            
            gx_all = np.floor(x_w_all / 0.00625).astype(np.int64)
            gy_all = np.floor(y_w_all / 0.00625).astype(np.int64)
            gz_all = np.floor(z_w_all / 0.00625).astype(np.int64)
            
            for idx in range(len(tx_flat)):
                tx_val = tx_flat[idx]
                ty_val = ty_flat[idx]
                
                for step_idx in range(len(z_steps)):
                    v_key = (gx_all[idx, step_idx], gy_all[idx, step_idx], gz_all[idx, step_idx])
                    if v_key in voxel_grid:
                        v = voxel_grid[v_key]
                        if v['occupied'] and v['weight'] >= 3.0:
                            # Exact sub-voxel depth calculation
                            v_world = np.array([v['x'], v['y'], v['z'], 1.0])
                            v_cam = inv_pose @ v_world
                            true_z = -v_cam[2]
                            model_anchors.append((tx_val, ty_val, ai_depth[ty_val, tx_val], true_z))
                            break
                            
        model_anchors = np.array(model_anchors) if len(model_anchors) > 0 else np.empty((0, 4))
                                
        # 4. RANSAC + LSQ Optimization
        if calibration_frames < 5:
            active_anchors1 = tof_anchors if len(tof_anchors) > 0 else np.empty((0, 4))
            if len(slam_anchors) > 0:
                active_anchors1 = np.vstack((active_anchors1, slam_anchors))
        else:
            arrays_to_stack = []
            if len(tof_anchors) > 0: arrays_to_stack.append(tof_anchors)
            if len(slam_anchors) > 0: arrays_to_stack.append(slam_anchors)
            if len(model_anchors) > 0: arrays_to_stack.append(model_anchors)
            active_anchors1 = np.vstack(arrays_to_stack) if arrays_to_stack else np.empty((0, 4))
            
        success = False
        if len(active_anchors1) >= 2:
            X = active_anchors1[:, 2].reshape(-1, 1)
            y = active_anchors1[:, 3]
            try:
                ransac.fit(X, y)
                r_s = ransac.estimator_.coef_[0]
                r_t = ransac.estimator_.intercept_
                z_pred = r_s * active_anchors1[:, 2] + r_t
                tol = np.maximum(0.05, active_anchors1[:, 3] * 0.05)
                inlier_mask = np.abs(z_pred - active_anchors1[:, 3]) < tol
                inliers_count = np.sum(inlier_mask)
                success = inliers_count > max(10, len(active_anchors1) * 0.05)
            except ValueError:
                success = False

        if not success:
            if calibration_frames >= 5:
                print("Model ICP failed, falling back to frozen state.")
                s_final, t_final = smoothed_s, smoothed_t
            else:
                print("RANSAC failed, discarding frame.")
                continue
        else:
            z_ai_inliers = active_anchors1[inlier_mask, 2]
            z_metric_inliers = active_anchors1[inlier_mask, 3]
            # Agent Logic: Replace standard LSQ with Inverse-Depth Weighted LSQ
            # Prioritize foreground alignment to prevent object ghosting
            w = 1.0 / (z_metric_inliers + 0.5)**2
            
            sum_w = np.sum(w)
            sum_ai = np.sum(w * z_ai_inliers)
            sum_metric = np.sum(w * z_metric_inliers)
            
            # Weighted Covariance components
            s_xx = np.sum(w * (z_ai_inliers ** 2)) - (sum_ai ** 2) / sum_w
            s_xy = np.sum(w * z_ai_inliers * z_metric_inliers) - (sum_ai * sum_metric) / sum_w
            
            if s_xx > 1e-5:
                s_refit = s_xy / s_xx
                t_refit = (sum_metric - s_refit * sum_ai) / sum_w
            else:
                s_refit, t_refit = r_s, r_t
                
            if calibration_frames < 5:
                if is_first_frame:
                    smoothed_s, smoothed_t = s_refit, t_refit
                    is_first_frame = False
                else:
                    if abs(s_refit - smoothed_s) < 0.1 and abs(t_refit - smoothed_t) < 0.1:
                        alpha = 0.7
                        smoothed_s = alpha * s_refit + (1.0 - alpha) * smoothed_s
                        smoothed_t = alpha * t_refit + (1.0 - alpha) * smoothed_t
                    else:
                        smoothed_s, smoothed_t = s_refit, t_refit
                calibration_frames += 1
                s_final, t_final = smoothed_s, smoothed_t
            else:
                if abs(s_refit - smoothed_s) < 0.1 and abs(t_refit - smoothed_t) < 0.1:
                    alpha = 0.7
                    smoothed_s = alpha * s_refit + (1.0 - alpha) * smoothed_s
                    smoothed_t = alpha * t_refit + (1.0 - alpha) * smoothed_t
                else:
                    smoothed_s, smoothed_t = s_refit, t_refit
                s_final, t_final = smoothed_s, smoothed_t
                    
        print(f"Frame {i} S: {s_final:.4f} T: {t_final:.4f} (Anchors: {len(active_anchors1)})")
                    
        # 5. SPATIAL HASHING & POINT FUSION
        dy_ai_full, dx_ai_full = np.gradient(ai_depth)
        sub_ai_depth = ai_depth[0:896:4, 0:504:4]
        dx_ai = dx_ai_full[0:896:4, 0:504:4].flatten()
        dy_ai = dy_ai_full[0:896:4, 0:504:4].flatten()
        z_lin = sub_ai_depth.flatten()
        ty_sub, tx_sub = np.meshgrid(np.arange(0, 896, 4), np.arange(0, 504, 4), indexing='ij')
        tx_flat = tx_sub.flatten()
        ty_flat = ty_sub.flatten()
        # Agent Logic: Restored pure affine prediction. DO NOT apply local warping.
        z_true = s_final * z_lin + t_final
        grad_tol = 0.04 * z_lin
        valid_fusion = (z_true >= 0.1) & (z_true <= 3.5) & (np.abs(dx_ai) <= grad_tol) & (np.abs(dy_ai) <= grad_tol)
        nx = (tx_flat - cx) / cx
        ny = (ty_flat - cy) / cy
        r2 = nx**2 + ny**2
        
        # Agent Logic: Implement Hyper-Radial Attenuation
        # Aggressive boundary penalty (Cubic falloff)
        blend_weight = 1.0 - (r2 ** 1.5)
        
        # Strict Frustum Cropping: Discard outer 15% completely
        valid_fusion &= (r2 < 0.85)
        valid_fusion &= (blend_weight > 0.0)
        
        # Phase 2: Morphological Boundary Erosion
        valid_fusion_2d = valid_fusion.reshape((224, 126))
        valid_fusion_2d = binary_erosion(valid_fusion_2d, iterations=2)
        valid_fusion = valid_fusion_2d.flatten()
        
        tx_val, ty_val = tx_flat[valid_fusion], ty_flat[valid_fusion]
        z_true_val = z_true[valid_fusion]
        bw_val = blend_weight[valid_fusion]
        
        if img is not None:
            r_val, g_val, b_val = img[ty_val, tx_val, 0].astype(float), img[ty_val, tx_val, 1].astype(float), img[ty_val, tx_val, 2].astype(float)
        else:
            r_val, g_val, b_val = np.full(len(tx_val), 255.0), np.full(len(tx_val), 255.0), np.full(len(tx_val), 255.0)
            
        x_p = (tx_val - cx) / fx
        y_p = -(ty_val - cy) / fy
        z_p = np.full_like(x_p, -1.0)
        
        # Valid Surface Points
        x_w = pose_matrix[0, 0]*x_p*z_true_val + pose_matrix[0, 1]*y_p*z_true_val + pose_matrix[0, 2]*z_p*z_true_val + pose_matrix[0, 3]
        y_w = pose_matrix[1, 0]*x_p*z_true_val + pose_matrix[1, 1]*y_p*z_true_val + pose_matrix[1, 2]*z_p*z_true_val + pose_matrix[1, 3]
        z_w = pose_matrix[2, 0]*x_p*z_true_val + pose_matrix[2, 1]*y_p*z_true_val + pose_matrix[2, 2]*z_p*z_true_val + pose_matrix[2, 3]
        gx_arr, gy_arr, gz_arr = np.floor(x_w / 0.00625).astype(np.int64), np.floor(y_w / 0.00625).astype(np.int64), np.floor(z_w / 0.00625).astype(np.int64)
        
        bw_inc = bw_val * 16.0
        forced_keys = np.array([None] * len(tx_val), dtype=object)
        
        # Agent Logic: Implement Virtual Z-Buffer for Occlusion and Space Carving
        if len(voxel_grid) > 0:
            # 1. Extract existing world points
            keys = list(voxel_grid.keys())
            world_pts = np.array([[voxel_grid[k]['x'], voxel_grid[k]['y'], voxel_grid[k]['z']] for k in keys])
            world_pts_h = np.hstack((world_pts, np.ones((len(world_pts), 1))))
            
            # 2. Project world into current camera POV
            cam_pts = (inv_pose @ world_pts_h.T).T
            z_c = -cam_pts[:, 2]
            
            # Keep points in front of the camera
            valid_cam = z_c > 0.1
            cam_pts = cam_pts[valid_cam]
            z_c = z_c[valid_cam]
            active_keys = [keys[idx] for idx, val in enumerate(valid_cam) if val]
            
            # 3. Project to pixels
            u_proj = np.round((cam_pts[:, 0] * fx / z_c) + cx).astype(int)
            v_proj = np.round(cy - (cam_pts[:, 1] * fy / z_c)).astype(int)
            
            # Subsample to match the fusion grid (step of 4)
            u_sub = u_proj // 4
            v_sub = v_proj // 4
            
            # Bounds check for the subsampled 224x126 AI depth grid
            in_bounds = (u_sub >= 0) & (u_sub < 126) & (v_sub >= 0) & (v_sub < 224)
            
            u_sub = u_sub[in_bounds]
            v_sub = v_sub[in_bounds]
            z_c = z_c[in_bounds]
            final_keys = [active_keys[idx] for idx, val in enumerate(in_bounds) if val]

            # 4. Build the Z-Buffer (Using a dictionary for sparse mapping of pixels to world keys)
            z_buffer = {}
            for idx in range(len(z_c)):
                v_center, u_center = v_sub[idx], u_sub[idx]
                depth = z_c[idx]
                key = final_keys[idx]

                # Splat into adjacent pixels
                for dv in [-1, 0, 1]:
                    for du in [-1, 0, 1]:
                        p = (v_center + dv, u_center + du)
                        # Keep the closest surface geometry
                        if p not in z_buffer or depth < z_buffer[p][0]:
                            z_buffer[p] = (depth, key)

            # 5. Execute Cull & Carve logic against new DAv3 points
            valid_fusion_cull = np.ones(len(tx_val), dtype=bool) 

            for idx in range(len(tx_val)):
                p_v, p_u = ty_val[idx] // 4, tx_val[idx] // 4
                new_z = z_true_val[idx]
                pixel = (p_v, p_u)

                if pixel in z_buffer:
                    old_z, old_key = z_buffer[pixel]

                    # Dynamic tolerance: Depth uncertainty increases with distance (min 6cm)
                    tol = max(0.06, old_z * 0.04)

                    # Condition A: Strictly Occluded (Behind the integration band)
                    if new_z > old_z + tol:
                        valid_fusion_cull[idx] = False

                    # Condition B: Free-Space Violation (In front of the integration band)
                    elif new_z < old_z - tol:
                        # Penalize the ghost voxel blocking the true surface
                        if old_key in voxel_grid:
                            voxel_grid[old_key]['weight'] -= (bw_inc[idx] * 2.0)
                            if voxel_grid[old_key]['weight'] <= 0.0:
                                del voxel_grid[old_key]
                                z_buffer[pixel] = (0.0, None) # Nullify to prevent KeyError
                                
                    # Condition C: Surface Match (Within Tolerance)
                    else:
                        forced_keys[idx] = old_key
                        
                        # SEAM SMOOTHING: Taper the fusion weight based on Z-delta
                        delta_z = abs(new_z - old_z)
                        
                        # Quadratic falloff: 100% weight at 0m delta, approaching 1% weight at tolerance limit
                        taper = 1.0 - (delta_z / tol)**2
                        
                        # Apply the smoothing taper directly to the incoming voxel weight
                        bw_inc[idx] *= max(0.01, taper)
                        
            # 6. Apply the occlusion mask to the incoming fusion arrays
            forced_keys = forced_keys[valid_fusion_cull]
            tx_val = tx_val[valid_fusion_cull]
            ty_val = ty_val[valid_fusion_cull]
            z_true_val = z_true_val[valid_fusion_cull]
            bw_inc = bw_inc[valid_fusion_cull]
            x_w = x_w[valid_fusion_cull]
            y_w = y_w[valid_fusion_cull]
            z_w = z_w[valid_fusion_cull]
            r_val = r_val[valid_fusion_cull]
            g_val = g_val[valid_fusion_cull]
            b_val = b_val[valid_fusion_cull]
            gx_arr = gx_arr[valid_fusion_cull]
            gy_arr = gy_arr[valid_fusion_cull]
            gz_arr = gz_arr[valid_fusion_cull]
            
        for idx in range(len(x_w)):
            # Accumulate Valid Surface Voxels
            if forced_keys[idx] is not None:
                v_key = forced_keys[idx] # Bypass math.floor hash
            else:
                v_key = (gx_arr[idx], gy_arr[idx], gz_arr[idx])
                
            w_i = bw_inc[idx]
            if v_key not in voxel_grid:
                voxel_grid[v_key] = {'x': x_w[idx], 'y': y_w[idx], 'z': z_w[idx], 'r': r_val[idx], 'g': g_val[idx], 'b': b_val[idx], 'weight': w_i, 'occupied': True}
            else:
                v = voxel_grid[v_key]
                w = v['weight']
                new_w = w + w_i
                v['x'] = (v['x'] * w + x_w[idx] * w_i) / new_w
                v['y'] = (v['y'] * w + y_w[idx] * w_i) / new_w
                v['z'] = (v['z'] * w + z_w[idx] * w_i) / new_w
                v['r'] = (v['r'] * w + r_val[idx] * w_i) / new_w
                v['g'] = (v['g'] * w + g_val[idx] * w_i) / new_w
                v['b'] = (v['b'] * w + b_val[idx] * w_i) / new_w
                v['weight'] = new_w

    print(f"Total processed voxels: {len(voxel_grid)}")
    ply_path = os.path.join(dataset_dir, 'prototype_points.ply')
    print(f"Exporting point cloud to {ply_path}...")
    export_voxels = [v for v in voxel_grid.values() if v['weight'] >= 3.0]
    if not export_voxels:
        print("Exported 0 valid voxels. Complete.")
        return
    pts = np.array([[v['x'], v['y'], v['z']] for v in export_voxels], dtype=np.float64)
    colors = np.array([[v['r']/255.0, v['g']/255.0, v['b']/255.0] for v in export_voxels], dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.voxel_down_sample(voxel_size=0.015)
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Exported {len(export_voxels)} valid voxels. Complete.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        process_dataset(sys.argv[1])
    else:
        print("Usage: python prototype.py <path_to_unzipped_dataset>")
