# Feature Detection Project Guide

This project implements a dense feature detection and localization pipeline using **DINOv2** (`facebook/dinov2-base`). It is designed to find a small reference template image within a sequence of query images and output the matched image names along with their 3D spatial coordinates.

---

## 📂 Project Structure

- **[detection_prat.py](file:///home/prath/Downloads/feature_detection/detection_prat.py)**: The main, improved detection pipeline. It uses multi-scale dense similarity mapping to match lower-resolution reference templates against high-resolution webcam/HD query images.
- **[detection.py](file:///home/prath/Downloads/feature_detection/detection.py)**: The baseline detection pipeline. It resizes both the reference and query frames to 128×128, leading to aspect ratio squashing and localization accuracy loss on HD inputs.
- **[ref_images/](file:///home/prath/Downloads/feature_detection/ref_images)**: Directory containing reference image templates (e.g., photos from a phone).
- **[images_dir/](file:///home/prath/Downloads/feature_detection/images_dir)**: Directory containing high-resolution query frames (`image_*.jpg`) and their corresponding metadata YAML files (`image_*.yaml`) containing ground truth 3D spatial positions.
- **[matched_coordinates.yml](file:///home/prath/Downloads/feature_detection/matched_coordinates.yml)**: Output file containing a list of the 3D coordinates of all query frames where a valid match was detected.

---

## ⚙️ Pipelines Comparison

### 1. Baseline Pipeline (`detection.py`)
The baseline pipeline operates by downsampling all inputs to 128×128 pixels:
- **Resizing**: Resizes both the reference and query images to $128 \times 128$ (squashing any non-1:1 aspect ratio inputs).
- **Embedding**: Computes DINOv2 embeddings on the $128 \times 128$ canvas padded to $224 \times 224$ (via [_pad_to_224](file:///home/prath/Downloads/feature_detection/detection.py#L36-L45)), producing a spatial representation of $10 \times 10$ patches ($100$ tokens of $768$ dimensions).
- **Matching**: Computes cosine similarity between all reference patches and all query patches ([compare_and_localize](file:///home/prath/Downloads/feature_detection/detection.py#L134-L198)).
- **Localization**: Uses thresholding and connected components on the $10 \times 10$ similarity grid to identify the target region and upscales the coordinate back to the original image size.

---

### 2. Improved Pipeline (`detection_prat.py`)
To match a small, possibly lower-resolution reference patch (e.g., $128 \times 128$ from a phone) inside a high-resolution query image without loss of detail, `detection_prat.py` implements **Dense Deep Feature Template Matching with Unsupervised Attention Gating**:

```mermaid
graph TD
    Ref[Reference 128x128] -->|Pad & Embed| DINO_Ref[DINOv2 Encoder]
    DINO_Ref -->|Saliency Map| Attention[CLS Self-Attention weights: 10x10]
    DINO_Ref -->|ROI Extraction| Template[Template Features: 10x10x768]
    
    Query[HD Query 1080p] -->|Resize keeping Aspect Ratio| QueryScale[Multi-Scale Query: e.g. H=448]
    QueryScale -->|Embed| DINO_Query[DINOv2 Encoder]
    DINO_Query -->|Feature Map| FeatMap[Query Feature Map: H_q x W_q x 768]
    
    Template -->|Attention-Weighted Convolution| CrossCorr[Weighted Cosine Similarity conv2d]
    Attention -->|Attention-Weighted Convolution| CrossCorr
    FeatMap -->|Attention-Weighted Convolution| CrossCorr
    
    CrossCorr --> Heatmap[Dense Heatmap]
    Heatmap -->|Argmax Peak Score| Success[Save Coordinates & Log Output]
```

- **Attention Extraction**: The reference is forced to $128 \times 128$, padded to $224 \times 224$ (via [_pad_to_224](file:///home/prath/Downloads/feature_detection/detection_prat.py#L32-L41)), and encoded using `attn_implementation="eager"`. The last-layer self-attentions from the `CLS` token to the patch tokens are extracted ([get_template_embedding](file:///home/prath/Downloads/feature_detection/detection_prat.py#L43-L70)) to produce an unsupervised $10 \times 10$ saliency mask.
- **Weighted Cross-Correlation**: We multiply each template patch by its squared self-attention score ($W^2$) and run a weighted 2D convolution over the query feature map ([compute_dense_similarity](file:///home/prath/Downloads/feature_detection/detection_prat.py#L78-L98)). This automatically forces DINOv2 to focus only on the stones, ignoring background tiles (regardless of whether they are grey or brown).
- **Scale-Space Search**: The query image is processed at multiple heights (`[336, 448, 560]`) while **preserving its aspect ratio** ([compare_and_localize](file:///home/prath/Downloads/feature_detection/detection_prat.py#L172-L238)). Bounding boxes are scaled back to the original HD coordinates.

---

## 🚀 Execution & Usage

### Running the Improved Pipeline
To run the main detection script:
```bash
python detection_prat.py images_dir ref_images/
```

### Hyperparameters
You can adjust the following parameters inside the script:
- `MATCH_THRESHOLD` (default: `0.32`): The minimum attention-weighted cosine similarity score required to declare a match.
- `SHOW_DEBUG` (default: `False`): Set to `False` for headless execution compatibility.

### Output
1. **Terminal logs**: Only the filename and confidence percentage are output to standard output:
   ```text
   image_45.jpg 52.51%
   ```
2. **Coordinate File**: The 3D coordinates from the matched frames' YAML files are saved in `matched_coordinates.yml`.
