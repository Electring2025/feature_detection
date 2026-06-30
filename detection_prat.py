import cv2
import numpy as np
import os
import sys
import torch
import time
import argparse
import yaml
from transformers import AutoImageProcessor, AutoModel

# =============================================================================
#  HYPERPARAMETERS — edit everything here
# =============================================================================
MATCH_THRESHOLD    = 0.37       # Cosine-similarity threshold for patch matching
MODEL_NAME         = "facebook/dinov2-base"

# Mapping of reference filenames to descriptive nouns (Feature IDs)
FEATURE_NAMES = {
    "red rock.jpeg": "red_rock",
    "silver soil.jpeg": "silver_soil",
    "red soil.jpeg": "red_soil",
    # Add other reference filenames and their desired nouns here
}

# Visualization toggle
SHOW_DEBUG         = True        # Set to True to visualize matches
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
processor = AutoImageProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
model     = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager", local_files_only=True).to(device).eval()

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def _pad_to_224(img_rgb_128: np.ndarray) -> np.ndarray:
    """
    Place a 128×128 RGB image onto a 224×224 canvas filled with the image's mean colour.
    Offset: 42 px = 3 × 14 px DINOv2 patch.
    """
    assert img_rgb_128.shape == (128, 128, 3), f"Expected (128,128,3), got {img_rgb_128.shape}"
    mean_color = img_rgb_128.mean(axis=(0, 1)).astype(np.uint8)
    canvas = np.full((224, 224, 3), mean_color, dtype=np.uint8)
    canvas[42:170, 42:170] = img_rgb_128
    return canvas

def get_template_embedding(img_rgb_128: np.ndarray):
    """
    Compute template normalized embeddings and CLS attention weights for a 128×128 RGB reference image.
    Returns:
      normalized_patches: Tensor of shape [10, 10, 768] (unit normalized along channel dimension)
      roi_attn: Tensor of shape [10, 10]
    """
    canvas = _pad_to_224(img_rgb_128)
    inputs = processor(images=canvas, return_tensors="pt").to(device)
    model.config.output_attentions = True
    with torch.no_grad():
        outputs = model(**inputs)
    model.config.output_attentions = False
    patch_tokens = outputs.last_hidden_state[:, 1:, :]          # drop CLS [1, 256, 768]
    patch_grid   = patch_tokens.view(16, 16, 768)
    roi_patches  = patch_grid[3:13, 3:13, :]                     # 10×10 ROI [10, 10, 768]
    
    normalized_patches = torch.nn.functional.normalize(roi_patches, p=2, dim=-1)
    
    # Extract and average CLS attention maps from the last 3 layers
    last_layers_att = [outputs.attentions[i] for i in [-1, -2, -3]]
    all_cls_to_patches = []
    for att in last_layers_att:
        cls_to_patches = att[0, :, 0, 1:]
        all_cls_to_patches.append(cls_to_patches.mean(dim=0))
    mean_attention = torch.stack(all_cls_to_patches).mean(dim=0)
    attn_grid = mean_attention.view(16, 16)
    roi_attn = attn_grid[3:13, 3:13]
    roi_attn = (roi_attn - roi_attn.min()) / (roi_attn.max() - roi_attn.min() + 1e-8)
    
    return normalized_patches, roi_attn

def preprocess_image_tensor(img_bgr: np.ndarray, target_w: int, target_h: int, device):
    """
    Manually resize and normalize query image using ImageNet statistics.
    Returns a tensor of shape [1, 3, target_h, target_w] ready for DINOv2.
    """
    img_resized = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Convert to float tensor and normalize
    tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device)

def compute_dense_similarity(query_features, template_features, weights):
    """
    Perform attention-weighted dense cosine similarity matching in the deep feature space.
    query_features: [1, 768, H_q, W_q]
    template_features: [10, 10, 768]
    weights: [10, 10]
    """
    device_loc = query_features.device
    kh, kw, C = template_features.shape
    
    T = template_features.permute(2, 0, 1).unsqueeze(0) # [1, 768, 10, 10]
    W2 = (weights ** 2).unsqueeze(0).unsqueeze(0) # [1, 1, 10, 10]
    
    T_weighted = T * W2
    
    numerator = torch.nn.functional.conv2d(query_features, T_weighted, padding=0)
    denominator = W2.sum()
    
    similarity = numerator / (denominator + 1e-8)
    return similarity.squeeze()

# ---------------------------------------------------------------------------
# YAML load helpers
# ---------------------------------------------------------------------------
def load_yaml_coordinates(yaml_path: str) -> dict | None:
    if not os.path.exists(yaml_path): return None
    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        pos = data.get("position", {})
        return {"x": pos.get("x", 0), "y": pos.get("y", 0), "z": pos.get("z", 0)}
    except Exception as e:
        print(f"[WARN] Could not parse {yaml_path}: {e}", file=sys.stderr)
        return None

def load_camera_calibration(yaml_path: str):
    """
    Load camera calibration matrices (Camera Matrix, Dist Coeffs) using PyYAML.
    """
    if not yaml_path or not os.path.exists(yaml_path):
        return None, None
    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        cam_data = data.get("camera_matrix", {})
        dist_data = data.get("distortion_coefficients", {})
        
        camera_matrix = None
        dist_coeffs = None
        
        if "data" in cam_data:
            camera_matrix = np.array(cam_data["data"], dtype=np.float32).reshape((3, 3))
        if "data" in dist_data:
            dist_coeffs = np.array(dist_data["data"], dtype=np.float32)
            
        return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"[WARN] Failed to load calibration via PyYAML: {e}", file=sys.stderr)
        return None, None

# ---------------------------------------------------------------------------
# Reference-image loading
# ---------------------------------------------------------------------------
def load_reference_images(ref_dir: str) -> list:
    refs = []
    valid_exts = (".jpeg", ".jpg", ".png")
    if not os.path.exists(ref_dir):
        print(f"[ERROR] Reference directory does not exist: {ref_dir}", file=sys.stderr)
        return refs
    files = sorted([f for f in os.listdir(ref_dir) if f.lower().endswith(valid_exts)])
    for name in files:
        path = os.path.join(ref_dir, name)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue

        # Force references to 128x128
        if img_bgr.shape[0] != 128 or img_bgr.shape[1] != 128:
            img_128_bgr = cv2.resize(img_bgr, (128, 128), interpolation=cv2.INTER_AREA)
        else:
            img_128_bgr = img_bgr
            
        img_128_rgb = cv2.cvtColor(img_128_bgr, cv2.COLOR_BGR2RGB)
        
        normalized_patches, roi_attn = get_template_embedding(img_128_rgb)
        ref_noun = FEATURE_NAMES.get(name, name)
        refs.append({
            "name": name,
            "noun": ref_noun,
            "emb": normalized_patches,
            "attn": roi_attn,
        })
    return refs

# ---------------------------------------------------------------------------
# Comparison & Localization
# ---------------------------------------------------------------------------
def compare_and_localize(img_bgr: np.ndarray, ref: dict, threshold: float) -> dict | None:
    orig_h, orig_w = img_bgr.shape[:2]
    
    best_score = -1.0
    best_bbox = None
    best_centroid = None
    
    # Search across multiple query resolutions to handle scale variations
    for target_h in [336, 448, 560]:
        target_w = int(round((target_h * (orig_w / orig_h)) / 14)) * 14
        
        query_tensor = preprocess_image_tensor(img_bgr, target_w, target_h, device)
        with torch.no_grad():
            outputs = model(pixel_values=query_tensor)
            
        patch_tokens = outputs.last_hidden_state[:, 1:, :] # [1, H_patches * W_patches, 768]
        H_patches = target_h // 14
        W_patches = target_w // 14
        
        patch_grid = patch_tokens.view(1, H_patches, W_patches, 768).permute(0, 3, 1, 2)
        query_features = torch.nn.functional.normalize(patch_grid, p=2, dim=1)
        
        sim_map = compute_dense_similarity(query_features, ref["emb"], ref["attn"])
        
        max_val, max_idx = torch.max(sim_map.view(-1), dim=0)
        max_val = max_val.item()
        
        if max_val > best_score:
            best_score = max_val
            
            H_map, W_map = sim_map.shape
            py = (max_idx // W_map).item()
            px = (max_idx % W_map).item()
            
            # Map patch coordinates to resized image
            x1, y1 = px * 14, py * 14
            x2, y2 = (px + 10) * 14, (py + 10) * 14
            
            # Scale to original image
            scale_x = orig_w / target_w
            scale_y = orig_h / target_h
            
            orig_x1 = max(0, min(orig_w - 1, int(round(x1 * scale_x))))
            orig_y1 = max(0, min(orig_h - 1, int(round(y1 * scale_y))))
            orig_x2 = max(0, min(orig_w - 1, int(round(x2 * scale_x))))
            orig_y2 = max(0, min(orig_h - 1, int(round(y2 * scale_y))))
            
            best_bbox = (orig_x1, orig_y1, orig_x2, orig_y2)
            best_centroid = ((orig_x1 + orig_x2) // 2, (orig_y1 + orig_y2) // 2)
            
    if best_score < threshold:
        return None
        
    x1, y1, x2, y2 = best_bbox
    if x2 <= x1 or y2 <= y1:
        return None
        
    return {
        "centroid": best_centroid,
        "bbox": best_bbox,
        "score": best_score
    }

# ---------------------------------------------------------------------------
# Main Processing Loop
# ---------------------------------------------------------------------------
def process_images(images_dir: str, ref_images_dir: str, threshold: float, calib_yaml: str = None):
    # Default camera parameters if no calibration file is found
    fx = 1651.36327
    fy = 1649.05851
    cx = 989.33579
    cy = 533.58065
    
    if calib_yaml:
        cam_mat, dist_coeff = load_camera_calibration(calib_yaml)
        if cam_mat is not None:
            fx = float(cam_mat[0, 0])
            fy = float(cam_mat[1, 1])
            cx = float(cam_mat[0, 2])
            cy = float(cam_mat[1, 2])

    refs = load_reference_images(ref_images_dir)
    if not refs:
        print("[ERROR] No reference images found. Exiting.", file=sys.stderr)
        return []

    query_images = []
    valid_exts = (".jpeg", ".jpg", ".png")
    if os.path.exists(images_dir):
        import re
        files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)]
        def get_file_num(f):
            match = re.search(r'\d+', f)
            return int(match.group()) if match else 0
        files.sort(key=get_file_num)

        for name in files:
            img_path = os.path.join(images_dir, name)
            base_name, _ = os.path.splitext(name)
            yaml_path = os.path.join(images_dir, base_name + ".yaml")
            if not os.path.exists(yaml_path):
                yaml_path = os.path.join(images_dir, base_name + ".yml")
                if not os.path.exists(yaml_path):
                    yaml_path = None
            
            img_idx = get_file_num(name)
            query_images.append((img_idx, img_path, yaml_path))

    # Group images by stop and select the top 4 sharpest (least blurred) frames using Laplacian Variance
    stops = {}
    for (img_idx, img_path, yaml_path) in query_images:
        group_id = (img_idx // 10) * 10
        if group_id not in stops:
            stops[group_id] = []
        stops[group_id].append((img_idx, img_path, yaml_path))

    selected_query_images = []
    for group_id, group_list in sorted(stops.items()):
        scored_images = []
        for (img_idx, img_path, yaml_path) in group_list:
            img = cv2.imread(img_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                scored_images.append((blur_score, (img_idx, img_path, yaml_path)))
        
        # Sort by blur score descending (highest variance = sharpest first)
        scored_images.sort(key=lambda x: x[0], reverse=True)
        # Select top 4
        top_4 = [item[1] for item in scored_images[:4]]
        
        selected_names = [os.path.basename(x[1]) for x in top_4]
        print(f"[INFO] Group {group_id}: Selected 4 sharpest frames: {selected_names}", flush=True)
        selected_query_images.extend(top_4)

    # Group raw detections by stop group_id (10 images per stop)
    group_detections = {}

    for count, (img_idx, img_path, yaml_path) in enumerate(selected_query_images, 1):
        if count % 4 == 0 or count == len(selected_query_images):
            print(f"-> Active progress: processed {count}/{len(selected_query_images)} frames...", flush=True)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue

        coords = load_yaml_coordinates(yaml_path)
        
        any_match = False
        # Collect matches for all reference templates in current frame
        frame_candidates = []
        for ref in refs:
            match_data = compare_and_localize(img_bgr, ref, threshold)
            if match_data:
                frame_candidates.append((ref, match_data))
        
        # Select the single best template match (highest confidence score)
        if frame_candidates:
            best_ref, best_match = max(frame_candidates, key=lambda x: x[1]["score"])
            score = best_match["score"]
            u, v = best_match["centroid"]
            x1, y1, x2, y2 = best_match["bbox"]

            # Format: only image_name and confidence_percent to stdout
            print(f"{os.path.basename(img_path)} {score * 100:.2f}%")

            any_match = True
            matched_bbox = (x1, y1, x2, y2)
            matched_centroid = (int(u), int(v))
            matched_noun = best_ref["noun"]
            matched_score = score

            if coords:
                z = coords.get("z", 1.8)
                dx = ((u - cx) * z) / fx
                dy = -((v - cy) * z) / fy  # Negative Y offset projection
                
                global_x = coords.get("x", 0.0) + dx
                global_y = coords.get("y", 0.0) + dy
                global_z = 0.0  # Ground plane z-coordinate
                
                group_id = (img_idx // 10) * 10
                if group_id not in group_detections:
                    group_detections[group_id] = []
                group_detections[group_id].append({
                    "ref_name": best_ref["name"],
                    "x": global_x,
                    "y": global_y,
                    "z": global_z,
                    "score": score
                })

        if SHOW_DEBUG:
            debug_frame = img_bgr.copy()
            if any_match and matched_bbox:
                bx1, by1, bx2, by2 = matched_bbox
                cv2.rectangle(debug_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.circle(debug_frame, matched_centroid, 5, (0, 0, 255), -1)
                text = f"{matched_noun} - {matched_score * 100:.1f}%"
                cv2.putText(debug_frame, text, (bx1, max(by1 - 10, 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(debug_frame, "No Match Detected", (30, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            
            cv2.namedWindow("xyz", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("xyz", 800, 600)
            cv2.imshow("xyz", debug_frame)
            cv2.waitKey(1)

    # Apply stop-wise majority voting & confidence filtering
    validated_by_group = {} 
    for group_id, detections in group_detections.items():
        counts = {}
        max_scores = {}
        for d in detections:
            r = d["ref_name"]
            counts[r] = counts.get(r, 0) + 1
            max_scores[r] = max(max_scores.get(r, 0.0), d["score"])
        
        if counts:
            # Winner has highest occurrence count, resolved by highest confidence score
            winning_ref = max(counts.keys(), key=lambda r: (counts[r], max_scores[r]))
            validated_by_group[group_id] = [d for d in detections if d["ref_name"] == winning_ref]

    # Find the single 10-frame stop group with the most detections for each reference feature
    ref_group_counts = {}
    ref_group_max_score = {}
    ref_group_detections = {}

    for group_id, detections in validated_by_group.items():
        for d in detections:
            r = d["ref_name"]
            if r not in ref_group_counts:
                ref_group_counts[r] = {}
                ref_group_max_score[r] = {}
                ref_group_detections[r] = {}
            
            ref_group_detections[r][group_id] = ref_group_detections[r].get(group_id, [])
            ref_group_detections[r][group_id].append(d)
            ref_group_counts[r][group_id] = len(ref_group_detections[r][group_id])
            ref_group_max_score[r][group_id] = max(ref_group_max_score[r].get(group_id, 0.0), d["score"])

    # Resolve coordinates ONLY from the single best stop group for each unique feature
    matched_coords = []
    for ref in refs:
        ref_name = ref["name"]
        if ref_name in ref_group_counts and ref_group_counts[ref_name]:
            # Select the winning 10-frame group: most occurrences, resolved by max confidence score
            best_group_id = max(
                ref_group_counts[ref_name].keys(),
                key=lambda gid: (ref_group_counts[ref_name][gid], ref_group_max_score[ref_name][gid])
            )
            best_pts = ref_group_detections[ref_name][best_group_id]
            pts = np.array([[d["x"], d["y"], d["z"]] for d in best_pts])
            mean_pt = np.mean(pts, axis=0)
            
            matched_coords.append({
                "ref_name": ref_name,
                "x": float(mean_pt[0]),
                "y": float(mean_pt[1]),
                "z": float(mean_pt[2]),
                "source_group": f"image_{best_group_id}.jpg to image_{best_group_id+9}.jpg"
            })

    print(f"Processing complete. Evaluated {len(query_images)} images.", flush=True)
    return matched_coords


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dense Feature Localization using DINOv2.")
    parser.add_argument("images_dir", help="Path to the folder containing query images.")
    parser.add_argument("ref_images_dir", help="Path to the folder containing reference images.")
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD, help="Heatmap threshold.")
    parser.add_argument("--calib", type=str, default="zebronics_HD_new.yaml", help="Path to camera calibration YAML.")
    
    args = parser.parse_args()

    coords_list = process_images(
        images_dir     = args.images_dir,
        ref_images_dir = args.ref_images_dir,
        threshold      = args.threshold,
        calib_yaml     = args.calib,
    )

    # Format the coordinates into a YAML string with Feature ID comments
    yaml_lines = []
    for item in coords_list:
        ref_noun = FEATURE_NAMES.get(item["ref_name"], item["ref_name"])
        yaml_lines.append(f"# Feature ID: {ref_noun}")
        yaml_lines.append(f"# Position obtained from image set: {item['source_group']}")
        yaml_lines.append(f"- ref_name: \"{item['ref_name']}\"")
        yaml_lines.append(f"  x: {item['x']}")
        yaml_lines.append(f"  y: {item['y']}")
        yaml_lines.append(f"  z: {item['z']}")
    
    yaml_str = "\n".join(yaml_lines) + "\n" if yaml_lines else "[]\n"
    with open("matched_coordinates.yml", "w") as f:
        f.write(yaml_str)

    # Clean up resources to prevent VS Code terminal window/display server crashes
    cv2.destroyAllWindows()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[INFO] Processing complete. Matched coordinates saved to 'matched_coordinates.yml'.")

