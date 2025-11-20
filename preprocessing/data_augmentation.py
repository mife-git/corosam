import os
import cv2
import numpy as np
import shutil
import yaml
from tqdm import tqdm

def load_config(config_path="preprocessing_config.yaml"):
    """Loads configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_datasets(root, source_tasks, target_task, cfg):
    """
    Merges multiple tasks (e.g., train and val) into a single target task (e.g., train_all).
    Copies all images and annotations from source tasks to the target task folder.
    """
    target_path = os.path.join(root, target_task)
    target_img_folder = os.path.join(target_path, cfg["folders"]["images"])
    target_ann_folder = os.path.join(target_path, cfg["folders"]["annotations"])

    # Create target folders
    os.makedirs(target_img_folder, exist_ok=True)
    os.makedirs(target_ann_folder, exist_ok=True)

    print(f"\n=== Merging {source_tasks} into {target_task} ===")

    for task in source_tasks:
        print(f"Copying from {task}...")
        task_path = os.path.join(root, task)
        source_img_folder = os.path.join(task_path, cfg["folders"]["images"])
        source_ann_folder = os.path.join(task_path, cfg["folders"]["annotations"])

        # Copy images
        if os.path.exists(source_img_folder):
            for filename in tqdm(os.listdir(source_img_folder)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src = os.path.join(source_img_folder, filename)
                    dst = os.path.join(target_img_folder, filename)
                    shutil.copy2(src, dst)

        # Copy annotations (ground truth masks)
        if os.path.exists(source_ann_folder):
            for filename in os.listdir(source_ann_folder):
                if filename.endswith('_gt.png'):
                    src = os.path.join(source_ann_folder, filename)
                    dst = os.path.join(target_ann_folder, filename)
                    shutil.copy2(src, dst)

    print(f"Merge complete! Created {target_task}")


def augment_images(input_folder, gt_folder, output_folder, output_gt_folder, aug_config):
    """
    Augment both images and their ground truth masks with identical transformations.
    Ground truth files are expected to have "_gt" suffix before the extension.
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_gt_folder, exist_ok=True)

    # Get augmentation parameters
    p = aug_config.get("probability", 0.5)
    rotation_range = aug_config.get("rotation_range", 180)
    zoom_range = aug_config.get("zoom_range", 0.5)
    brightness_range = aug_config.get("brightness_range", 0.5)
    seed = aug_config.get("seed", 42)

    # Set seeds
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Augmenting {len(image_files)} images from {input_folder}...")

    for filename in tqdm(image_files):
        input_path = os.path.join(input_folder, filename)
        base_name, ext = os.path.splitext(filename)
        gt_filename = f"{base_name}_gt{ext}"
        gt_path = os.path.join(gt_folder, gt_filename)

        if not os.path.exists(gt_path):
            print(f"Skipping {filename}: GT not found ({gt_filename})")
            continue

        # Copy originals
        shutil.copy2(input_path, os.path.join(output_folder, filename))
        shutil.copy2(gt_path, os.path.join(output_gt_folder, gt_filename))

        # Read images
        gray_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if gray_image is None or gt_image is None:
            print(f"Error reading {filename} or its GT")
            continue

        height, width = gray_image.shape

        try:
            # Decide which augmentations to apply
            do_rotate = np.random.random() > p
            do_zoom = np.random.random() > p
            do_flip = np.random.random() > p
            do_brightness = np.random.random() > p

            # If none selected, choose one randomly
            if not any([do_rotate, do_zoom, do_flip, do_brightness]):
                augmentation_choice = np.random.choice(['rotate', 'zoom', 'flip', 'brightness'])
                do_rotate = augmentation_choice == 'rotate'
                do_zoom = augmentation_choice == 'zoom'
                do_flip = augmentation_choice == 'flip'
                do_brightness = augmentation_choice == 'brightness'

            # Initialize with original images
            augmented_image = gray_image.copy()
            augmented_gt = gt_image.copy()

            # Apply selected augmentations
            if do_brightness:
                gamma = np.random.uniform(1 - brightness_range, 1 + brightness_range)
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
                augmented_image = cv2.LUT(augmented_image, table)

            if do_zoom:
                scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
                scale = max(0.1, scale)
                scaled_width = max(1, int(width * scale))
                scaled_height = max(1, int(height * scale))

                augmented_image = cv2.resize(augmented_image, (scaled_width, scaled_height))
                augmented_gt = cv2.resize(augmented_gt, (scaled_width, scaled_height))

                if scale > 1:
                    start_x = (scaled_width - width) // 2
                    start_y = (scaled_height - height) // 2
                    augmented_image = augmented_image[start_y:start_y + height, start_x:start_x + width]
                    augmented_gt = augmented_gt[start_y:start_y + height, start_x:start_x + width]
                else:
                    pad_x = (width - scaled_width) // 2
                    pad_y = (height - scaled_height) // 2
                    augmented_image = cv2.copyMakeBorder(augmented_image, pad_y, pad_y + (height - scaled_height) % 2,
                                                         pad_x, pad_x + (width - scaled_width) % 2,
                                                         cv2.BORDER_CONSTANT, value=0)
                    augmented_gt = cv2.copyMakeBorder(augmented_gt, pad_y, pad_y + (height - scaled_height) % 2,
                                                      pad_x, pad_x + (width - scaled_width) % 2,
                                                      cv2.BORDER_CONSTANT, value=0)

            if do_flip:
                augmented_image = cv2.flip(augmented_image, 1)
                augmented_gt = cv2.flip(augmented_gt, 1)

            if do_rotate:
                angle = np.random.uniform(-rotation_range, rotation_range)
                rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
                augmented_image = cv2.warpAffine(augmented_image, rotation_matrix, (width, height))
                augmented_gt = cv2.warpAffine(augmented_gt, rotation_matrix, (width, height))

            # Save augmented versions
            output_filename = f"{base_name}_aug{ext}"
            output_gt_filename = f"{base_name}_aug_gt{ext}"

            output_path = os.path.join(output_folder, output_filename)
            output_gt_path = os.path.join(output_gt_folder, output_gt_filename)

            cv2.imwrite(output_path, augmented_image)
            cv2.imwrite(output_gt_path, augmented_gt)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue


if __name__ == "__main__":
    cfg = load_config("preprocessing_config.yaml")

    root = cfg["root_dir"]

    # Merge train and val into train_all
    source_tasks = cfg.get("merge", {}).get("source_tasks", ["train", "val"])
    target_task = cfg.get("merge", {}).get("target_task", "train_all")

    merge_datasets(root, source_tasks, target_task, cfg)

    # Augment train_all
    print(f"\nAugmenting {target_task}")

    aug_config = cfg.get("augmentation", {})
    task_path = os.path.join(root, target_task)
    img_folder = os.path.join(task_path, cfg["folders"]["images"])
    ann_folder = os.path.join(task_path, cfg["folders"]["annotations"])

    output_img_folder = f"{img_folder}_augmented"
    output_ann_folder = f"{ann_folder}_augmented"

    augment_images(
        input_folder=img_folder,
        gt_folder=ann_folder,
        output_folder=output_img_folder,
        output_gt_folder=output_ann_folder,
        aug_config=aug_config
    )
