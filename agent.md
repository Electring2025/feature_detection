# Feature Detection Project Guide

This project implements a dense feature detection and localization pipeline using **DINOv2** (`facebook/dinov2-base`). It is designed to find a small reference template image within a sequence of query images, calculate their projected global 3D coordinates, and output them clubbed by camera position stops.

---

## 📂 Project Structure

- **[detection_prat.py](file:///home/prath/Downloads/feature_detection/detection_prat.py)**: The main, improved detection pipeline. It uses multi-scale dense similarity mapping to match lower-resolution reference templates against high-resolution webcam/HD query images.
- **[detection.py](file:///home/prath/Downloads/feature_detection/detection.py)**: The baseline detection pipeline. It resizes both the reference and query frames to 128×128, leading to aspect ratio squashing and localization accuracy loss on HD inputs.
- **[ref_images/](file:///home/prath/Downloads/feature_detection/ref_images)**: Directory containing reference image templates (e.g., photos from a phone).
- **[images_dir/](file:///home/prath/Downloads/feature_detection/images_dir)**: Directory containing high-resolution query frames (`image_*.jpg`) and their corresponding metadata YAML files (`image_*.yaml`) containing ground truth 3D spatial positions.
- **[zebronics_HD_new.yaml](file:///home/prath/Downloads/feature_detection/zebronics_HD_new.yaml)**: Camera calibration matrix and distortion coefficients.
- **[matched_coordinates.yml](file:///home/prath/Downloads/feature_detection/matched_coordinates.yml)**: Output file containing the clubbed and averaged global coordinates of detected features.

---

## ⚙️ Ground Projection & Clubbing

When a feature match is detected in a query image:
1. **Centroid Extraction**: The centroid $(u, v)$ in pixels is calculated from the best-matched bounding box.
2. **Camera Coordinate Projection**: We compute the physical ground displacement offsets $(\Delta x, \Delta y)$ relative to the camera center:
   $$\Delta x = \frac{(u - c_x) \cdot z_c}{f_x}$$
   $$\Delta y = -\frac{(v - c_y) \cdot z_c}{f_y}$$
   Where:
   - $f_x, f_y$ and $c_x, c_y$ are loaded from the camera calibration matrix in [zebronics_HD_new.yaml](file:///home/prath/Downloads/feature_detection/zebronics_HD_new.yaml).
   - $z_c$ is the camera altitude/height loaded from the image's YAML metadata.
3. **Global Translation**: The offsets are added to the camera position $(x_c, y_c)$ to compute the global 3D coordinate of the feature:
   $$x_{\text{global}} = x_c + \Delta x$$
   $$y_{\text{global}} = y_c + \Delta y$$
   $$z_{\text{global}} = 0.0 \text{ (ground plane)}$$
4. **Stop-wise Clubbing**: Multiple query images taken at the same flight position (typically in blocks of 10, e.g., `image_30` to `image_39`) are grouped together. The coordinates calculated across the matching frames are averaged to output a single, robust position.

---

## 🚀 Execution & Usage

### Running the Pipeline
To run the main detection script:
```bash
python detection_prat.py images_dir ref_images/ --calib zebronics_HD_new.yaml
```

### Hyperparameters
You can adjust the following parameters inside the script:
- `MATCH_THRESHOLD` (default: `0.32`): The minimum attention-weighted cosine similarity score required to declare a match.
- `SHOW_DEBUG` (default: `False`): Set to `False` for headless execution compatibility.
