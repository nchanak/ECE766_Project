import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    AutoModelForSemanticSegmentation,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)

# Outputs are in ./layered_output folder


####################################
# Change this to whatever input    #
####################################
image_path = "landscape.jpg"
####################################

image_stem = Path(image_path).stem 

# Models
dino_model_id = "IDEA-Research/grounding-dino-tiny"
sam_model_id = "facebook/sam-vit-base"
semantic_model_id = "nvidia/segformer-b5-finetuned-ade-640-640"
depth_model_id = "depth-anything/Depth-Anything-V2-Small-hf"

# DINO prompt
text_prompt = (
    "building. skyscraper. facade. wall. doorway. entrance. pole. sign. "
    "parked car. car. truck. bus. van. tree. fence. pole. rock. boulder. cliff."
)

dino_threshold = 0.30
dino_text_threshold = 0.25

# resize image for speed
MAX_SIZE = 800

# optional dino filter out small objects
MIN_BOX_AREA = 0
MIN_BOX_WIDTH = 0
MIN_BOX_HEIGHT = 0

# Semantic classes
SEMANTIC_CLASS_QUERIES = [
    "sky",
    "building",
    "ground",
    "mountain",
    "tree",
    "water",
    "road",
    "rock",
    "earth",
    "grass",
    "plant",
    "vegetation",
]

SORT_ASCENDING_DEPTH = True

OUTPUT_DIR = Path("layered_output")
MASK_DIR = OUTPUT_DIR / "out_masks"

# Clean old outputs every run
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# helpers
def mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_label_position(mask: np.ndarray):
    bbox = mask_bbox(mask)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def resize_image_keep_aspect(image: Image.Image, max_size: int) -> Image.Image:
    w, h = image.size
    scale = max_size / max(w, h)
    if scale < 1:
        image = image.resize((int(w * scale), int(h * scale)))
    return image


def mask_to_u8(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255)


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask_to_u8(mask)).save(path)


def compute_mask_depth(depth_map: np.ndarray, mask: np.ndarray) -> float | None:
    vals = depth_map[mask]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def colorize_layer_map(layer_map: np.ndarray) -> np.ndarray:
    max_layer = int(layer_map.max())
    rng = np.random.default_rng(0)
    colors = rng.integers(0, 255, size=(max_layer + 1, 3), dtype=np.uint8)
    colors[0] = np.array([0, 0, 0], dtype=np.uint8)
    return colors[layer_map]


def find_matching_class_ids(id2label: dict, substring: str) -> list[int]:
    return [
        int(cid) for cid, name in id2label.items()
        if substring.lower() in str(name).lower()
    ]


# resize image
image = Image.open(image_path).convert("RGB")
image = resize_image_keep_aspect(image, MAX_SIZE)
image_np = np.array(image)
H, W = image_np.shape[:2]
print("Image size:", image.size)


# dino process
dino_processor = AutoProcessor.from_pretrained(dino_model_id)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(device)

dino_inputs = dino_processor(images=image, text=text_prompt, return_tensors="pt").to(device)

with torch.no_grad():
    dino_outputs = dino_model(**dino_inputs)

dino_results = dino_processor.post_process_grounded_object_detection(
    dino_outputs,
    dino_inputs.input_ids,
    threshold=dino_threshold,
    text_threshold=dino_text_threshold,
    target_sizes=[image.size[::-1]],
)

dino_result = dino_results[0]
dino_boxes = dino_result["boxes"].cpu()
dino_scores = dino_result["scores"].cpu()
dino_labels = dino_result["labels"]

filtered_boxes = []
filtered_labels = []
filtered_scores = []

for label, score, box in zip(dino_labels, dino_scores, dino_boxes):
    x1, y1, x2, y2 = box.tolist()
    bw = x2 - x1
    bh = y2 - y1
    area = bw * bh
    if area >= MIN_BOX_AREA and bw >= MIN_BOX_WIDTH and bh >= MIN_BOX_HEIGHT:
        filtered_boxes.append([x1, y1, x2, y2])
        filtered_labels.append(str(label))
        filtered_scores.append(float(score))

print("\nFiltered DINO detections:")
for label, score, box in zip(filtered_labels, filtered_scores, filtered_boxes):
    print(label, score, box)


# Sam masks
instance_entries = []

if filtered_boxes:
    sam_processor = SamProcessor.from_pretrained(sam_model_id)
    sam_model = SamModel.from_pretrained(sam_model_id).to(device)

    sam_inputs = sam_processor(
        images=image,
        input_boxes=[filtered_boxes],
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)

    sam_masks = sam_processor.image_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(),
        sam_inputs["original_sizes"].cpu(),
        sam_inputs["reshaped_input_sizes"].cpu()
    )[0]

    for i in range(sam_masks.shape[0]):
        mask = sam_masks[i, 0].numpy() > 0
        if mask.sum() == 0:
            continue

        instance_entries.append({
            "name": filtered_labels[i],
            "source": "sam",
            "score": filtered_scores[i],
            "mask": mask,
        })

        save_mask(mask, MASK_DIR / f"{image_stem}_sam_{i:02d}_{filtered_labels[i]}.png")


# semantic masks
semantic_processor = AutoImageProcessor.from_pretrained(semantic_model_id)
semantic_model = AutoModelForSemanticSegmentation.from_pretrained(semantic_model_id).to(device)

semantic_inputs = semantic_processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    semantic_outputs = semantic_model(**semantic_inputs)

seg = semantic_processor.post_process_semantic_segmentation(
    semantic_outputs,
    target_sizes=[image.size[::-1]]
)[0].cpu().numpy()

id2label = semantic_model.config.id2label

print("\nSemantic classes present:")
for cid in sorted(np.unique(seg).tolist()):
    print(cid, id2label.get(cid, f"unknown_{cid}"))

semantic_entries = []

for query in SEMANTIC_CLASS_QUERIES:
    class_ids = find_matching_class_ids(id2label, query)
    if not class_ids:
        continue

    mask = np.isin(seg, class_ids)
    if mask.sum() == 0:
        continue

    semantic_entries.append({
        "name": query,
        "source": "semantic",
        "score": 1.0,
        "mask": mask,
    })

    save_mask(mask, MASK_DIR / f"{image_stem}_semantic_{query}.png")


# depth map generation
depth_processor = AutoImageProcessor.from_pretrained(depth_model_id)
depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_id).to(device)

depth_inputs = depth_processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    depth_outputs = depth_model(**depth_inputs)

predicted_depth = depth_outputs.predicted_depth
predicted_depth = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=(H, W),
    mode="bicubic",
    align_corners=False,
).squeeze()

depth_map = predicted_depth.cpu().numpy()

depth_norm = depth_map - depth_map.min()
if depth_norm.max() > 0:
    depth_norm = depth_norm / depth_norm.max()
depth_u8 = (depth_norm * 255).astype(np.uint8)
Image.fromarray(depth_u8).save(OUTPUT_DIR / f"{image_stem}_depth_map.png")


# semantic overlap handling
instance_union = np.zeros((H, W), dtype=bool)
for entry in instance_entries:
    instance_union |= entry["mask"]

for entry in semantic_entries:
    entry["mask"] = entry["mask"] & (~instance_union)


# depth score per mask
all_entries = []

for entry in instance_entries + semantic_entries:
    if entry["mask"].sum() == 0:
        continue

    depth_score = compute_mask_depth(depth_map, entry["mask"])
    if depth_score is None:
        continue

    entry["depth_score"] = depth_score
    all_entries.append(entry)


# probably don't have sky in front of building, so just in case
CLASS_DEPTH_PRIOR = {
    "sky": -1e6,
    "mountain": -2e5,
    "building": -1e5,
    "water": -5e4,
}

for entry in all_entries:
    prior = CLASS_DEPTH_PRIOR.get(entry["name"], 0.0)
    entry["sort_key"] = entry["depth_score"] + prior


# sort depth and map
all_entries.sort(key=lambda e: e["sort_key"], reverse=not SORT_ASCENDING_DEPTH)

layer_map = np.zeros((H, W), dtype=np.uint8)

for layer_id, entry in enumerate(all_entries, start=1):
    entry["layer_id"] = layer_id
    layer_map[entry["mask"]] = layer_id

Image.fromarray(layer_map).save(OUTPUT_DIR / f"{image_stem}_layer_map.png")

layer_color = colorize_layer_map(layer_map)
Image.fromarray(layer_color).save(OUTPUT_DIR / f"{image_stem}_layer_color.png")

overlay = (
    0.55 * image_np.astype(np.float32) +
    0.45 * layer_color.astype(np.float32)
).astype(np.uint8)
Image.fromarray(overlay).save(OUTPUT_DIR / f"{image_stem}_layer_overlay.png")

# save annotated overlay
annotated_overlay_img = Image.fromarray(overlay.copy())
draw = ImageDraw.Draw(annotated_overlay_img)

try:
    font = ImageFont.truetype("arial.ttf", 14)
except Exception:
    font = ImageFont.load_default()

for entry in all_entries:
    pos = mask_label_position(entry["mask"])
    if pos is None:
        continue

    x, y = pos
    label_text = f"{entry['name']} | d={entry['depth_score']:.2f} | L={entry['layer_id']}"

    text_bbox = draw.textbbox((x, y), label_text, font=font)
    pad = 2
    bg_box = (
        text_bbox[0] - pad,
        text_bbox[1] - pad,
        text_bbox[2] + pad,
        text_bbox[3] + pad,
    )

    draw.rectangle(bg_box, fill=(0, 0, 0))
    draw.text((x, y), label_text, fill=(255, 255, 255), font=font)

annotated_overlay_img.save(OUTPUT_DIR / f"{image_stem}_layer_overlay_annotated.png")

# save final per-layer masks and json metadata
serializable_entries = []
for entry in all_entries:
    mask_rel = f"out_masks/{image_stem}_layer_{entry['layer_id']:02d}_{entry['name'].replace(' ', '_')}.png"
    save_mask(entry["mask"], OUTPUT_DIR / mask_rel)

    bbox = mask_bbox(entry["mask"])

    serializable_entries.append({
        "name": entry["name"],
        "source": entry["source"],
        "score": entry["score"],
        "depth_score": entry["depth_score"],
        "sort_key": entry["sort_key"],
        "layer_id": entry["layer_id"],
        "pixel_count": int(entry["mask"].sum()),
        "mask_file": mask_rel,
        "bbox": bbox,
    })

with open(OUTPUT_DIR / f"{image_stem}_layers.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "image_path": image_path,
            "image_stem": image_stem,
            "image_size": [W, H],
            "depth_model": depth_model_id,
            "semantic_model": semantic_model_id,
            "dino_model": dino_model_id,
            "sam_model": sam_model_id,
            "layers": serializable_entries,
        },
        f,
        indent=2,
    )

print("\nSaved:")
print(f" - layered_output/{image_stem}_depth_map.png")
print(f" - layered_output/{image_stem}_layer_map.png")
print(f" - layered_output/{image_stem}_layer_color.png")
print(f" - layered_output/{image_stem}_layer_overlay.png")
print(f" - layered_output/{image_stem}_layer_overlay_annotated.png")
print(f" - layered_output/{image_stem}_layers.json")
print(f" - layered_output/out_masks/{image_stem}_*.png")