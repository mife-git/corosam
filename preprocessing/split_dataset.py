import os
import shutil
import yaml
import argparse
from sklearn.model_selection import KFold


def load_config(config_path="preprocessing_config.yaml"):
    """Loads configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def split_dataset(image_folder, anns_folder, split_folder_name, output_path, kf_splits, seed):
    """
    Split the augmented dataset for k-fold cross-validation.

    Args:
        image_folder: Path to folder containing input images
        anns_folder: Path to folder containing annotations (ground truth masks)
        split_folder_name: Name of output folder for k-fold splits
        output_path: Base path where to create the split folder
        kf_splits: Number of k-fold splits
        seed: Random seed for reproducibility
    """
    image_list = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    split_directory = os.path.join(output_path, split_folder_name)

    if not os.path.exists(split_directory):
        os.makedirs(split_directory)

    kf_cv = KFold(n_splits=kf_splits, shuffle=True, random_state=seed)

    # Filter only non-augmented images (original images without '_aug' suffix)
    original_images = [img for img in image_list if '_aug' not in img]

    print(f"\nFound {len(original_images)} original images")
    print(f"Found {len(image_list) - len(original_images)} augmented images")
    print(f"\nSplitting into {kf_splits} folds with seed {seed}\n")

    for split, (train_idx, val_idx) in enumerate(kf_cv.split(original_images)):
        print(f'=' * 50)
        print(f'SPLIT {split + 1}/{kf_splits}')
        print(f'TRAIN INDEXES: {train_idx}')
        print(f'VAL INDEXES: {val_idx}')
        print(f'TRAIN len: {len(train_idx)}, VAL len: {len(val_idx)}')
        print(f'=' * 50)

        # Create directory structure for this split
        directory = os.path.join(split_directory, f'set{split + 1}')
        if not os.path.exists(directory):
            os.makedirs(directory)

        train_images_dir = os.path.join(directory, 'train', 'images')
        train_annotations_dir = os.path.join(directory, 'train', 'annotations')
        val_images_dir = os.path.join(directory, 'val', 'images')
        val_annotations_dir = os.path.join(directory, 'val', 'annotations')

        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(train_annotations_dir, exist_ok=True)
        os.makedirs(val_images_dir, exist_ok=True)
        os.makedirs(val_annotations_dir, exist_ok=True)

        # Get train and validation sets
        train_set_cv = [original_images[i] for i in train_idx]
        val_set_cv = [original_images[i] for i in val_idx]

        # Copy training data (original + augmented)
        print(f"\nCopying training data for split {split + 1}...")
        for img in train_set_cv:
            # Copy original image
            shutil.copy(os.path.join(image_folder, img), train_images_dir)

            # Copy original ground truth
            gt_img = img.replace('.png', '_gt.png')
            gt_path = os.path.join(anns_folder, gt_img)
            if os.path.exists(gt_path):
                shutil.copy(gt_path, train_annotations_dir)
            else:
                print(f"Warning: Ground truth not found for {img}")

            # Copy augmented image (if exists)
            augmented_img = img.replace('.png', '_aug.png')
            augmented_img_path = os.path.join(image_folder, augmented_img)
            if os.path.exists(augmented_img_path):
                shutil.copy(augmented_img_path, train_images_dir)

                # Copy augmented ground truth
                augmented_gt_img = augmented_img.replace('.png', '_gt.png')
                augmented_gt_path = os.path.join(anns_folder, augmented_gt_img)
                if os.path.exists(augmented_gt_path):
                    shutil.copy(augmented_gt_path, train_annotations_dir)

        # Copy validation data (only original, no augmentation)
        print(f"Copying validation data for split {split + 1}...")
        for img in val_set_cv:
            # Copy original image
            shutil.copy(os.path.join(image_folder, img), val_images_dir)

            # Copy original ground truth
            gt_img = img.replace('.png', '_gt.png')
            gt_path = os.path.join(anns_folder, gt_img)
            if os.path.exists(gt_path):
                shutil.copy(gt_path, val_annotations_dir)
            else:
                print(f"Warning: Ground truth not found for {img}")

        # Save train and validation set lists to CSV files
        with open(os.path.join(directory, 'train_set.csv'), 'w') as f:
            f.write("\n".join(train_set_cv))

        with open(os.path.join(directory, 'val_set.csv'), 'w') as f:
            f.write("\n".join(val_set_cv))

        print(f"Split {split + 1} completed!\n")

    print("\n" + "=" * 50)
    print("K-fold split completed successfully!")
    print("=" * 50)

    return train_set_cv, val_set_cv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split dataset for k-fold cross-validation')
    parser.add_argument('--config', type=str, default='preprocessing_config.yaml',
                        help='Path to configuration YAML file')
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Get split configuration
    split_cfg = cfg.get('split', {})

    # Get paths - using train_all folders created by merge_and_augment.py
    root = cfg['root_dir']
    train_all_task = cfg.get('merge', {}).get('target_task', 'train_all')

    data_path = os.path.join(root, train_all_task, cfg['folders']['images'] + '_augmented')
    anns_path = os.path.join(root, train_all_task, cfg['folders']['annotations'] + '_augmented')
    output_path = root

    kf_splits = split_cfg.get('kf_splits', 5)
    k_fold_folder_name = split_cfg.get('k_fold_folder_name', 'kf_split')
    seed = split_cfg.get('seed', cfg.get('seed', 2025))

    print(f"\nDataset Split Configuration:")
    print(f"  Input images: {data_path}")
    print(f"  Input annotations: {anns_path}")
    print(f"  Output path: {output_path}")
    print(f"  K-fold splits: {kf_splits}")
    print(f"  Output folder: {k_fold_folder_name}")
    print(f"  Random seed: {seed}")

    # Perform the split
    train_set, val_set = split_dataset(
        image_folder=data_path,
        anns_folder=anns_path,
        split_folder_name=k_fold_folder_name,
        output_path=output_path,
        kf_splits=kf_splits,
        seed=seed
    )