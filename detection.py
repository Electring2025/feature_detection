import cv2
import numpy as np
import os
import torch
import time
import argparse
import yaml
from transformers import AutoImageProcessor, AutoModel

# =============================================================================
#  HYPERPARAMETERS — edit everything here
# =============================================================================
MATCH_THRESHOLD    = 0.95       # Cosine-similarity threshold for patch matching
MODEL_NAME         = "facebook/dinov2-base"

# Color-gating tolerance (multiples of std-dev; set very high to disable)
H_GATE_SIGMA       = 2.0
S_GATE_SIGMA       = 2.0

# Visualization toggle
SHOW_DEBUG         = False
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
cosine_similarity = torch.nn.functional.cosine_similarity

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

def get_embedding(img_rgb_128: np.ndarray) :
    """
    Compute a normalised DINOv2 embedding for a 128×128 RGB numpy array.
    Returns: Tensor of shape [100, 768] (Spatial information preserved).
    """
    canvas = _pad_to_224(img_rgb_128)
    inputs = processor(images=canvas, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    patch_tokens = outputs.last_hidden_state[:, 1:, :]          # drop CLS
    patch_grid   = patch_tokens.view(-1, 16, 16, 768)           # spatial grid
    roi_patches  = patch_grid[:, 3:13, 3:13, :]                 # 10×10 ROI
    
    # Reshape to 100 patches, keeping spatial distinctness (No mean pooling)
    patch_embeddings = roi_patches.reshape(100, 768)
    patch_embeddings_mean = roi_patches.reshape(1,100, 768).mean(dim=1)
    # print(patch_embeddings_mean.shape)
    return torch.nn.functional.normalize(patch_embeddings, p=2, dim=1),torch.nn.functional.normalize(patch_embeddings_mean, p=2, dim=1)

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
        print(f"[WARN] Could not parse {yaml_path}: {e}")
        return None

def load_camera_calibration(yaml_path: str):
    """
    Load OpenCV camera calibration matrices (Camera Matrix, Dist Coeffs).
    """
    if not yaml_path or not os.path.exists(yaml_path):
        return None, None
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        print(f"[WARN] Failed to open calibration YAML: {yaml_path}")
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
        print(f"[ERROR] Reference directory does not exist: {ref_dir}")
        return refs
    files = sorted([f for f in os.listdir(ref_dir) if f.lower().endswith(valid_exts)])
    for name in files:
        path = os.path.join(ref_dir, name)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue

        img_128_bgr = cv2.resize(img_bgr, (128, 128), interpolation=cv2.INTER_AREA)
        img_128_rgb = cv2.cvtColor(img_128_bgr, cv2.COLOR_BGR2RGB)

        img_hsv = cv2.cvtColor(img_128_bgr, cv2.COLOR_BGR2HSV)
        h, s, _ = cv2.split(img_hsv)
        vari = get_embedding(img_128_rgb)
        refs.append({
            "name": name,
            "emb": vari[0], # Shape: [100, 768]
            "emb_mean": vari[1], # Shape: [100, 768]
            
            "h_stats": (float(np.mean(h)), float(np.std(h))),
            "s_stats": (float(np.mean(s)), float(np.std(s))),
        })
        print(f"  Loaded reference: {name}")
    return refs

# ---------------------------------------------------------------------------
# Per-image Comparison & Localization
# ---------------------------------------------------------------------------
def compare_and_localize(img_bgr: np.ndarray, ref: dict, threshold: float) -> dict | None:
    orig_h, orig_w = img_bgr.shape[:2]
    img_128_bgr = cv2.resize(img_bgr, (128, 128), interpolation=cv2.INTER_AREA)

    # --- Colour gate ---
    frame_hsv = cv2.cvtColor(img_128_bgr, cv2.COLOR_BGR2HSV)
    h_f, s_f, _ = cv2.split(frame_hsv)
    h_mean, h_std = ref["h_stats"]
    s_mean, s_std = ref["s_stats"]

    if abs(np.mean(h_f) - h_mean) > H_GATE_SIGMA * h_std or \
       abs(np.mean(s_f) - s_mean) > S_GATE_SIGMA * s_std:
        return None

    # --- Get Embeddings ---
    img_128_rgb = cv2.cvtColor(img_128_bgr, cv2.COLOR_BGR2RGB)
    query_emb,mean_emb = get_embedding(img_128_rgb)  # Shape: [100, 768]
    mean_emb_score = cosine_similarity(ref["emb_mean"],mean_emb).item()
    if(mean_emb_score>0.7):
        # --- Matrix Multiplication for Dense Similarity ---
        # S shape: [100 (query), 100 (ref)]. 
        S = torch.mm(query_emb, ref["emb"].t()) 
        
        # Max similarity across all ref patches for each query patch
        max_sims, _ = torch.max(S, dim=1) 
        heatmap = max_sims.reshape(10, 10).cpu().numpy()

        # --- Thresholding & Connected Components ---
        mask = (heatmap > threshold).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num_labels <= 1:
            return None  # Only background label (0) exists

        # Find the largest connected component (ignoring background)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        # Extract patch-level bounding box and centroid
        x_patch = stats[largest_label, cv2.CC_STAT_LEFT]
        y_patch = stats[largest_label, cv2.CC_STAT_TOP]
        w_patch = stats[largest_label, cv2.CC_STAT_WIDTH]
        h_patch = stats[largest_label, cv2.CC_STAT_HEIGHT]
        cx_patch, cy_patch = centroids[largest_label]
        
        max_score = float(np.max(heatmap[labels == largest_label]))

        # --- Coordinate Conversion ---
        # Map patch coordinates to 128x128 image (patch size 14, offset 7 for center)
        cx_128 = cx_patch * 14 + 7
        cy_128 = cy_patch * 14 + 7
        x1_128 = x_patch * 14
        y1_128 = y_patch * 14
        x2_128 = x1_128 + (w_patch * 14)
        y2_128 = y1_128 + (h_patch * 14)

        # Scale to original image dimensions
        scale_x = orig_w / 128.0
        scale_y = orig_h / 128.0

        return {
            "centroid": (int(cx_128 * scale_x), int(cy_128 * scale_y)),
            "bbox": (int(x1_128 * scale_x), int(y1_128 * scale_y), int(x2_128 * scale_x), int(y2_128 * scale_y)),
            "score": max_score
        }

# ---------------------------------------------------------------------------
# Main Processing Loop
# ---------------------------------------------------------------------------
def process_images(images_dir: str, ref_images_dir: str, threshold: float, calib_yaml: str = None):
    print(f"\n{'='*60}")
    print(f" Images dir : {images_dir}")
    print(f" Refs dir   : {ref_images_dir}")
    print(f" Threshold  : {threshold}")
    print(f"{'='*60}\n")

    # Camera Calib setup (Task 4)
    if calib_yaml:
        cam_mat, dist_coeff = load_camera_calibration(calib_yaml)
        if cam_mat is not None:
            print("[INFO] Camera Calibration loaded successfully.\n")

    print("Loading reference images …")
    refs = load_reference_images(ref_images_dir)
    if not refs:
        print("[ERROR] No reference images found. Exiting.")
        return []
    print(f"  {len(refs)} reference(s) loaded.\n")

    query_images = []
    valid_exts = (".jpeg", ".jpg", ".png")
    if os.path.exists(images_dir):
        import re
        files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)]
        # Parse numerical index from file name for sorting
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

    print(f"Query images found: {len(query_images)}\n")

    matched_coords = []
    a = time.time()
    for count, (img_idx, img_path, yaml_path) in enumerate(query_images, 1):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue

        coords = load_yaml_coordinates(yaml_path)
        coord_str = f"x={coords['x']}, y={coords['y']}, z={coords['z']}" if coords else "No YAML"

        matched_for_query = False
        for ref in refs:
            match_data = compare_and_localize(img_bgr, ref, threshold)
            
            if match_data:
                score = match_data["score"]
                cx, cy = match_data["centroid"]
                x1, y1, x2, y2 = match_data["bbox"]
                
                print(
                    f"[MATCH] query: image{img_idx}.jpeg | ref: {ref['name']} | "
                    f"score: {score * 100:.2f}% | pic_num: {img_idx} | "
                    f"Feature Centroid: ({cx}, {cy})"
                )

                if coords and not matched_for_query:
                    matched_coords.append(coords)
                    matched_for_query = True

                if SHOW_DEBUG:
                    debug_frame = img_bgr.copy()
                    # Draw BBox
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw Centroid
                    cv2.circle(debug_frame, (cx, cy), 5, (0, 0, 255), -1)
                    # Draw text
                    text = f"{ref['name']} - {score * 100:.1f}%"
                    cv2.putText(debug_frame, text, (x1, max(y1 - 10, 20)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.namedWindow("xyz", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("xyz", 800,600)
                    
                    cv2.imshow("xyz", debug_frame)
                    print("  [DEBUG] Press any key to continue to the next frame...")
                    cv2.waitKey(0)

        if count % 10 == 0:
            pct = 100 * count / len(query_images)
            print(f"  … {count}/{len(query_images)} images checked ({pct:.1f}%)")

    b = time.time()
    print(f"\nDone. Time taken: {b - a:.2f}s")
    if SHOW_DEBUG:
        cv2.destroyAllWindows()
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
        # calib_yaml     = args.calib
    )

    # Output matched coordinates in YAML list format
    print("\nMatched coordinates (YAML list format):")
    yaml_str = yaml.safe_dump(coords_list, default_flow_style=False)
    print(yaml_str)

    # Save to file
    with open("matched_coordinates.yml", "w") as f:
        f.write(yaml_str)
    print("\n[INFO] Saved coordinates to matched_coordinates.yml")