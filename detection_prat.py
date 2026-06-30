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
MATCH_THRESHOLD    = 0.32       # Cosine-similarity threshold for patch matching
MODEL_NAME         = "facebook/dinov2-base"

# Visualization toggle
SHOW_DEBUG         = True        # Set to True to visualize matches (press any key to continue)
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager").to(device).eval()

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
    
    # Extract CLS attention map from last layer
    last_layer_att = outputs.attentions[-1]
    cls_to_patches = last_layer_att[0, :, 0, 1:]
    mean_attention = cls_to_patches.mean(dim=0)
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
    Load OpenCV camera calibration matrices (Camera Matrix, Dist Coeffs).
    """
    if not yaml_path or not os.path.exists(yaml_path):
        return None, None
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        print(f"[WARN] Failed to open calibration YAML: {yaml_path}", file=sys.stderr)
        return None, None
    
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat() 
    fs.release()
    return camera_matrix, dist_coeffs

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
        refs.append({
            "name": name,
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
    # Camera Calib setup (Task 4)
    if calib_yaml:
        cam_mat, dist_coeff = load_camera_calibration(calib_yaml)

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

    matched_coords = []
    for count, (img_idx, img_path, yaml_path) in enumerate(query_images, 1):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue

        coords = load_yaml_coordinates(yaml_path)
        matched_for_query = False
        for ref in refs:
            match_data = compare_and_localize(img_bgr, ref, threshold)
            
            if match_data:
                score = match_data["score"]
                cx, cy = match_data["centroid"]
                x1, y1, x2, y2 = match_data["bbox"]
                
                # Format: only image_name and confidence_percent to stdout
                print(f"{os.path.basename(img_path)} {score * 100:.2f}%")

                if coords and not matched_for_query:
                    matched_coords.append(coords)
                    matched_for_query = True

                if SHOW_DEBUG:
                    debug_frame = img_bgr.copy()
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(debug_frame, (cx, cy), 5, (0, 0, 255), -1)
                    text = f"{ref['name']} - {score * 100:.1f}%"
                    cv2.putText(debug_frame, text, (x1, max(y1 - 10, 20)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.namedWindow("xyz", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("xyz", 800, 600)
                    cv2.imshow("xyz", debug_frame)
                    cv2.waitKey(0)

    return matched_coords

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dense Feature Localization using DINOv2.")
    parser.add_argument("images_dir", help="Path to the folder containing query images.")
    parser.add_argument("ref_images_dir", help="Path to the folder containing reference images.")
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD, help="Heatmap threshold.")
    parser.add_argument("--calib", type=str, default=None, help="Path to camera calibration YAML.")
    
    args = parser.parse_args()

    coords_list = process_images(
        images_dir     = args.images_dir,
        ref_images_dir = args.ref_images_dir,
        threshold      = args.threshold,
    )

    yaml_str = yaml.safe_dump(coords_list, default_flow_style=False)
    with open("matched_coordinates.yml", "w") as f:
        f.write(yaml_str)
