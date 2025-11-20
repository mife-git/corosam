import os
import json
import shutil
import yaml
import numpy as np
import cv2
from tqdm import tqdm


def load_config(config_path="preprocess_config.yaml"):
    """Loads configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_binary_mask_from_annotations(image_info, annotations, width, height):
    """Creates a binary mask from COCO polygon annotations for a single image."""
    mask = np.zeros((height, width), dtype=np.uint8)
    image_id = image_info['id']

    # Filter annotations
    anns = [ann for ann in annotations if ann['image_id'] == image_id]

    for ann in anns:
        # Get segmentation data
        seg = ann.get('segmentation', None)
        if seg is None:
            continue

        if isinstance(seg, list):
            for polygon in seg:
                pts = np.array(polygon, dtype=np.int32).reshape((-1, 2))
                # Draw the polygon on the mask
                cv2.fillPoly(mask, [pts], color=1)
        else:
            print(f"Warning: Unsupported segmentation format for annotation {ann.get('id')}")

    return mask


def generate_masks(task_path, json_name, cfg):
    """Generates and saves binary mask images from the ORIGINAL COCO JSON file (before renaming)."""
    ann_folder = os.path.join(task_path, cfg["folders"]["annotations"])
    img_folder = os.path.join(task_path, cfg["folders"]["images"])

    with open(os.path.join(ann_folder, json_name), "r") as f:
        data = json.load(f)

    images_info = data["images"]
    annotations = data["annotations"]

    info_by_filename = {img["file_name"]: img for img in images_info}

    print(f"Generating masks for {task_path}...")

    # Iterate over all images in the folder
    for img_filename in tqdm(os.listdir(img_folder)):
        if img_filename not in info_by_filename:
            continue

        info = info_by_filename[img_filename]
        img_path = os.path.join(img_folder, img_filename)

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        # Create the binary mask from annotations
        mask = create_binary_mask_from_annotations(info, annotations, w, h)
        mask_uint8 = (mask * 255).astype(np.uint8)

        base = os.path.splitext(img_filename)[0]
        mask_filename = f"{base}_gt.png"

        cv2.imwrite(os.path.join(ann_folder, mask_filename), mask_uint8)


def renumber_dataset(task_path, task_name, dataset_name, original_json, output_json, start_index, start_ann_id, cfg):
    """
    Renumbers images, masks, and their corresponding annotations sequentially.
    This is useful for merging multiple datasets.
    It renames image files, mask files, and creates a new COCO JSON file.

    IMPORTANT: This should be called AFTER generate_masks() has created the mask files.
    """
    img_folder = os.path.join(task_path, cfg["folders"]["images"])
    ann_folder = os.path.join(task_path, cfg["folders"]["annotations"])

    with open(os.path.join(ann_folder, original_json), "r") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    # Use filesystem order (os.listdir) instead of sorted order to maintain reproducibility
    # with the original preprocessing. This causes non-intuitive numbering (e.g., 10.png -> 2,
    # 11.png -> 3) but ensures consistency with the preprocessed data used for the paper
    files_in_folder = [f for f in os.listdir(img_folder) if not f.endswith('_gt.png')]

    # Create a dictionary for quick access to image info
    images_dict = {img["file_name"]: img for img in images}

    # Use the order of files as they appear in the folder
    images_sorted = [images_dict[f] for f in files_in_folder if f in images_dict]

    new_images = []
    new_annotations = []
    new_id = start_index
    ann_id = start_ann_id

    id_map = {}

    for img in images_sorted:
        old_filename = img["file_name"]
        old_id = img["id"]
        old_base = os.path.splitext(old_filename)[0]

        new_filename = f"{dataset_name}_{new_id}.png"
        new_base = f"{dataset_name}_{new_id}"

        id_map[old_id] = new_id

        # Rename the image file
        old_img_path = os.path.join(img_folder, old_filename)
        new_img_path = os.path.join(img_folder, new_filename)
        if os.path.exists(old_img_path):
            shutil.move(old_img_path, new_img_path)

        # Rename the corresponding mask file
        old_mask_filename = f"{old_base}_gt.png"
        new_mask_filename = f"{new_base}_gt.png"
        old_mask_path = os.path.join(ann_folder, old_mask_filename)
        new_mask_path = os.path.join(ann_folder, new_mask_filename)
        if os.path.exists(old_mask_path):
            shutil.move(old_mask_path, new_mask_path)

        new_images.append({
            "id": new_id,
            "file_name": new_filename,
            "width": img["width"],
            "height": img["height"]
        })

        new_id += 1

    # Update the image_id and annotation id in each annotation to match the new IDs
    for ann in annotations:
        new_annotations.append({
            **ann,
            "id": ann_id,
            "image_id": id_map[ann["image_id"]]
        })
        ann_id += 1

    new_json_path = os.path.join(ann_folder, output_json)
    with open(new_json_path, "w") as f:
        json.dump({"images": new_images, "annotations": new_annotations}, f, indent=2)

    return new_id, ann_id


if __name__ == "__main__":
    cfg = load_config("preprocessing_config.yaml")
    dataset_name = cfg["dataset_name"]

    root = cfg["root_dir"]
    tasks = cfg["tasks"]
    json_names = cfg["json_names"]
    output_json_names = cfg["output_json_names"]

    # Generate masks from original JSON files (before renaming)
    print("\nGenerating masks")
    for task in tasks:
        task_path = os.path.join(root, task)
        generate_masks(task_path, json_names[task], cfg)

    # Rename images, masks, and update JSON
    print("\nRenaming")
    current_index = 1
    current_ann_id = 1

    for task in tasks:
        print(f"\nRenumbering {task}...")

        task_path = os.path.join(root, task)
        original_json = json_names[task]
        new_json = output_json_names[task]

        current_index, current_ann_id = renumber_dataset(
            task_path,
            task,
            dataset_name,
            original_json,
            new_json,
            current_index,
            current_ann_id,
            cfg
        )

    print("\n=== Processing complete! ===")