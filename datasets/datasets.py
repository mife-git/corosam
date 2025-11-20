import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import cv2
import plantcv.plantcv as pcv
from skimage.morphology import binary_closing
from scipy.spatial import KDTree
import numpy as np


def _filter_close_points(points, min_distance):
    """
    Filters a list of points to ensure no two points are closer than `min_distance`.
    Uses a greedy approach with a KDTree for efficient neighbor search.
    """
    if min_distance <= 0 or len(points) <= 1:
        return points

    tree = KDTree(points)
    selected = []
    visited = set()

    for i, pt in enumerate(points):
        if i in visited:
            continue
        selected.append(pt)
        neighbors = tree.query_ball_point(pt, r=min_distance)
        visited.update(neighbors)

    return selected


def perturb_points(points, perturbation_std=0, gt_img=None):
    """
    Applies random Gaussian noise to a set of 2D points.

    Args:
        points (list or np.ndarray): A list of (x, y) coordinates.
        perturbation_std (float): The standard deviation of the Gaussian noise.
        gt_img (np.ndarray): The ground truth image, used to determine image boundaries
                             for clipping the perturbed points.

    Returns:
        np.ndarray: The perturbed points, clipped to stay within the image dimensions.
    """
    if perturbation_std > 0 and len(points) > 0:
        noise = np.random.normal(loc=0, scale=perturbation_std, size=(len(points), 2))
        points = np.array(points, dtype=np.float32) + noise
        points = np.clip(points, a_min=0, a_max=np.array(gt_img.shape[::-1]) - 1)  # stay in image bounds
    return np.array(points)


def get_point_prompts(gt_img,
                      perturbation=0,
                      pruning_size=20,
                      min_distance=5,
                      max_tips=None,
                      max_branches=None,
                      num_total_points=None):
    """
    Extracts salient points (tips and branches) from a vessel skeleton.

    Args:
        gt_img (ndarray): Binary image of the vessel ground truth.
        perturbation (float): Maximum amount of perturbation to apply.
        pruning_size (int): Size parameter for skeleton pruning.
        min_distance (int): Minimum distance between points for filtering.
        max_tips (int, optional): Maximum number of tip points to extract.
        max_branches (int, optional): Maximum number of branch points to extract.
        num_total_points (int, optional): Total number of points to extract, distributed between tips and branches.

    Returns:
        tips (ndarray): Array of [x, y] tip coordinates.
        branch_points (ndarray): Array of [x, y] branch point coordinates.
        skeleton (ndarray): Skeleton image.
    """
    # Disable debug mode in plantcv to avoid saving intermediate images
    pcv.params.debug = None

    # Pre-process the mask and extract the skeleton.
    # Closing is used to fill small holes in the vessel mask before skeletonization.
    gt_img = binary_closing(gt_img)
    skeleton = pcv.morphology.skeletonize(mask=gt_img)
    skeleton, _, _ = pcv.morphology.prune(skel_img=skeleton, size=pruning_size)

    # Identify endpoint pixels (tips) of the skeleton
    tips_mask = pcv.morphology.find_tips(skel_img=skeleton)
    tips = np.argwhere(tips_mask > 0)
    tips = sorted([tuple(pt[::-1]) for pt in tips])  # Convert (row, col) to (x, y)

    # Create a mask for the skeleton points at the border
    border_mask = np.zeros_like(skeleton)
    border_mask[0, :] = 1  # Top border
    border_mask[-1, :] = 1  # Bottom border
    border_mask[:, 0] = 1  # Left border
    border_mask[:, -1] = 1  # Right border

    # Find skeleton points that touch the border and treat them as tips.
    # This handles cases where vessels are cut off by the image frame.
    border_points = np.argwhere((skeleton > 0) & (border_mask > 0))
    border_tips = sorted([tuple(pt[::-1]) for pt in border_points])

    # Add border tips to regular tips
    if len(border_tips) > 0:
        tips.extend(border_tips)
        tips = sorted(list(set(tips)))  # Remove duplicates and sort

    # Identify branch points in the skeleton
    branch_mask = pcv.morphology.find_branch_pts(skel_img=skeleton)
    branch_points = np.argwhere(branch_mask > 0)
    branch_points = sorted([tuple(pt[::-1]) for pt in branch_points])

    # Fallback: If no branch points are found, use centroids of skeleton segments.
    # This ensures prompts can be generated even for simple, unbranched structures.
    if len(branch_points) == 0:
        segments, _ = pcv.morphology.segment_skeleton(skel_img=skeleton)
        pixels = segments.reshape(-1, 3)
        unique_colors = np.unique(pixels[np.any(pixels != 0, axis=1)], axis=0)

        skel_y, skel_x = np.where(skeleton > 0)
        skel_points = sorted(zip(skel_x, skel_y))
        skel_tree = KDTree(skel_points)

        centroids = []
        for color in unique_colors:
            mask = np.all(segments == color, axis=2)
            segment_points = np.argwhere(mask)
            segment_points = sorted(
                [tuple(pt[::-1]) for pt in segment_points if tuple(pt[::-1]) in skel_points]
            )

            if segment_points:
                centroid_x = np.mean([pt[0] for pt in segment_points])
                centroid_y = np.mean([pt[1] for pt in segment_points])
                centroid = (centroid_x, centroid_y)

                _, nearest_idx = skel_tree.query(centroid)
                nearest_skel_point = skel_points[nearest_idx]

                if nearest_skel_point not in tips and nearest_skel_point not in branch_points:
                    centroids.append(nearest_skel_point)

        # Add centroids deterministically
        centroids = sorted(centroids)
        if centroids:
            branch_points.extend(centroids)

    # Filter out points that are too close to each other to ensure good spatial distribution.
    tips = _filter_close_points(tips, min_distance)
    branch_points = _filter_close_points(branch_points, min_distance)

    # Apply sampling strategies based on the provided limits for the number of points.
    if num_total_points is not None:
        # Mode 1: Fixed total number of points, prioritize tips
        if len(tips) >= num_total_points:
            tips = tips[:num_total_points]
            branch_points = []
        else:
            remaining = num_total_points - len(tips)
            branch_points = branch_points[:remaining]

            # If tips and branches are not enough, supplement with random points from the skeleton.
            if len(branch_points) < remaining:
                skel_y, skel_x = np.where(skeleton > 0)
                skel_points = sorted(zip(skel_x, skel_y))
                tip_set = set(tips)
                branch_set = set(branch_points)

                additional_needed = remaining - len(branch_points)
                additional_candidates = [pt for pt in skel_points
                                         if pt not in tip_set and pt not in branch_set]

                if additional_candidates and additional_needed > 0:
                    additional_points = sorted(random.sample(additional_candidates,
                                                             min(additional_needed, len(additional_candidates))))
                    branch_points.extend(additional_points)

    elif max_tips is not None or max_branches is not None:
        # Mode 2: Independent limits on tips and branches
        if max_tips is not None and len(tips) > max_tips:
            tips = random.sample(tips, max_tips)

        if max_branches is not None and len(branch_points) > max_branches:
            branch_points = random.sample(branch_points, max_branches)

    # Apply random perturbation to the final point sets
    tips = perturb_points(tips, perturbation_std=perturbation, gt_img=gt_img)
    branch_points = perturb_points(branch_points, perturbation_std=perturbation, gt_img=gt_img)

    return np.array(tips), np.array(branch_points)


def create_point_channels(tips, branch_points, image_size):
    """
    Create two separate channels for points:
    - Channel 1: Encodes tip locations.
    - Channel 2: Encodes branch point locations.

    Each point is represented as a 4x4 square in its respective channel.
    This spatial encoding can be used as an input to a neural network.
    """
    tips_channel = np.zeros((image_size, image_size), dtype=np.float32)
    branch_channel = np.zeros((image_size, image_size), dtype=np.float32)

    # Draw 4x4 squares for each tip point
    for point in tips:
        x, y = int(point[0]), int(point[1])
        y_min, y_max = max(0, y - 2), min(image_size, y + 2)
        x_min, x_max = max(0, x - 2), min(image_size, x + 2)
        tips_channel[y_min:y_max, x_min:x_max] = 1.0

    # Add branch points to branch channel
    # Draw 4x4 squares for each branch point
    for point in branch_points:
        x, y = int(point[0]), int(point[1])
        y_min, y_max = max(0, y - 2), min(image_size, y + 2)
        x_min, x_max = max(0, x - 2), min(image_size, x + 2)
        branch_channel[y_min:y_max, x_min:x_max] = 1.0

    return tips_channel, branch_channel


def create_single_point_channel(points, image_size):
    """
    Creates a single channel encoding the locations of all provided points.

    Each point is represented as a 4x4 square. This is useful when not
    distinguishing between different types of points (e.g., tips vs. branches).
    """
    point_channel = np.zeros((image_size, image_size), dtype=np.float32)

    for point in points:
        x, y = int(point[0]), int(point[1])
        y_min, y_max = max(0, y - 2), min(image_size, y + 2)
        x_min, x_max = max(0, x - 2), min(image_size, x + 2)
        point_channel[y_min:y_max, x_min:x_max] = 1.0

    return point_channel


def get_expanded_bbox(gt, points, bbox_shift=5):
    """
    Calculates a bounding box that encompasses all provided points and then
    expands it by a random amount.

    Args:
        gt (np.ndarray): The ground truth image, used for shape information.
        points (np.ndarray): An array of [x, y] coordinates.
        bbox_shift (int): The maximum random value to add to each side of the bbox.

    Returns:
        np.ndarray: The expanded bounding box coordinates [x_min, y_min, x_max, y_max].
    """
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    H, W = gt.shape
    x_min = max(0, x_min - random.randint(0, bbox_shift))
    x_max = min(W, x_max + random.randint(0, bbox_shift))
    y_min = max(0, y_min - random.randint(0, bbox_shift))
    y_max = min(H, y_max + random.randint(0, bbox_shift))

    return np.array([x_min, y_min, x_max, y_max])


class CoroDataset(Dataset):
    """
    PyTorch Dataset for coronary artery data.

    This class handles loading images and ground truth masks, generating point/box
    prompts for segmentation models, and applying necessary pre-processing.

    It supports several modes for prompt generation:
    1.  `num_pos_pts`: A fixed total number of positive points.
    2.  `num_tips` & `num_other`: Separate limits for tip and branch points.
    3.  No limits: Use all detected points.

    Args:
        data_root (str): Path to the root directory of the dataset.
    """
    def __init__(self, data_root,
                 image_size=256,
                 use_augmented=False,
                 bbox_shift=10,
                 points_shift=0,
                 num_pos_pts=None,
                 num_tips=None,
                 num_other=None,
                 min_distance=5,
                 num_neg_pts=5,
                 divide_channels=True,
                 use_prompts=True):
        self.data_root = data_root
        # Determine whether to use original or augmented data
        self.use_augmented = use_augmented
        if self.use_augmented:
            self.img_dir = Path(data_root) / "images_augmented"
            self.gt_dir = Path(data_root) / "annotations_augmented"
        else:
            self.img_dir = Path(data_root) / "images"
            self.gt_dir = Path(data_root) / "annotations"

        # Configuration for prompt generation and data processing
        self.image_size = image_size
        self.bbox_shift = bbox_shift
        self.points_shift = points_shift
        self.num_pos_pts = num_pos_pts
        self.num_tips = num_tips
        self.num_other = num_other
        self.min_distance = min_distance
        self.num_neg_pts = num_neg_pts
        self.divide_channels = divide_channels
        self.use_prompts = use_prompts

        # Discover image files and derive corresponding ground truth file paths
        self.image_files = sorted(self.img_dir.glob("*.png"))
        if not self.image_files:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = self.image_files[index]
        gt_path = self.gt_dir / f"{img_path.stem}_gt.png"
        img_name = img_path.name

        # Load image and ground truth
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found for image {img_name} at {gt_path}")

        # Load image as grayscale
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Failed to load image: {img_path}")

        # Load ground truth as grayscale
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise IOError(f"Failed to load ground truth: {gt_path}")

        # Resize images and masks to a consistent size
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR).astype(
                np.uint8)
        if gt.shape[0] != self.image_size or gt.shape[1] != self.image_size:
            gt = cv2.resize(gt, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST).astype(
                np.uint8)

        # Binarize the ground truth mask and normalize the image
        gt = np.uint8(gt >= (np.max(gt) / 2))
        img = img / 255.0

        # Prompt generation
        if self.use_prompts:
            # Get tips and branch points using the unified function
            if self.num_pos_pts is not None:
                # Fixed total number of points mode
                tips, branch_points = get_point_prompts(
                    gt,
                    perturbation=self.points_shift,
                    min_distance=self.min_distance,
                    num_total_points=self.num_pos_pts
                )
            elif self.num_tips is not None or self.num_other is not None:
                # Limited tips and branches mode
                tips, branch_points = get_point_prompts(
                    gt,
                    perturbation=self.points_shift,
                    min_distance=self.min_distance,
                    max_tips=self.num_tips,
                    max_branches=self.num_other
                )
            else:
                # Original unlimited mode
                tips, branch_points = get_point_prompts(
                    gt,
                    perturbation=self.points_shift,
                    min_distance=self.min_distance
                )

            # Sample negative points from the background region
            neg_points = []
            if self.num_neg_pts is not None:
                y_indices, x_indices = np.where(gt == 0)
                if len(y_indices) > 0:  # Ensure there are negative regions
                    for _ in range(self.num_neg_pts):
                        random_idx = random.randint(0, len(x_indices) - 1)
                        neg_points.append([x_indices[random_idx], y_indices[random_idx]])

            # Combine all points and create corresponding labels (1 for positive, 0 for negative)
            all_points = []
            all_labels = []

            # Add positive points (tips and branches)
            if len(tips) > 0:
                all_points.extend(tips)
                all_labels.extend([1] * len(tips))

            if len(branch_points) > 0:
                all_points.extend(branch_points)
                all_labels.extend([1] * len(branch_points))

            # Add negative points
            if len(neg_points) > 0:
                all_points.extend(neg_points)
                all_labels.extend([0] * len(neg_points))

            if len(all_points) == 0:
                # Fallback: if no points were detected (e.g., empty mask),
                # create a single artificial positive point to avoid errors downstream
                print(f"Warning: No valid points found for {img_name}. Creating a fallback point.")
                y_indices, x_indices = np.where(gt > 0)
                if len(y_indices) > 0:
                    random_idx = random.randint(0, len(x_indices) - 1)
                    all_points.append([x_indices[random_idx], y_indices[random_idx]])
                    all_labels.append(1)
                else:
                    # If the mask is completely empty, we cannot generate a point
                    print(f"Error: Cannot create fallback point for {img_name} because mask is empty.")

            point_coords = np.array(all_points)
            point_labels = np.array(all_labels)

            # Generate a bounding box prompt around all points
            bboxes_coords = get_expanded_bbox(gt, point_coords, bbox_shift=self.bbox_shift)

            # Create input image with prompt channels
            if self.divide_channels:
                tips_channel, branch_channel = create_point_channels(tips, branch_points, self.image_size)
                num_points = len(branch_points) + len(tips)
                input_image = np.stack([img, tips_channel, branch_channel], axis=0)
            else:
                pos_points = np.array([p for p, l in zip(all_points, all_labels) if l == 1])
                num_points = len(pos_points)
                points_channel = create_single_point_channel(pos_points, self.image_size)
                input_image = np.stack([img, points_channel, points_channel], axis=0)

            # If num_pos_pts is specified, we are in a SAM-style inference mode
            # where prompts are passed separately, not as image channels
            if self.num_pos_pts is not None:
                input_image = np.stack([img, img, img], axis=0)
                return {
                    "image": torch.tensor(input_image).float(),  # Shape: [3, H, W]
                    "gt": torch.tensor(gt[None, :, :]).long(),
                    "image_name": img_name,
                    "index": index,
                    "num_points": num_points,
                    "bboxes_coords": torch.tensor(bboxes_coords[None, None, ...]).float(),
                    "point_coords": torch.tensor(point_coords).float(),
                    "point_labels": torch.tensor(point_labels).float(),
                }
            else:
                return {
                    "image": torch.tensor(input_image).float(),  # Shape: [3, H, W]
                    "gt": torch.tensor(gt[None, :, :]).long(),
                    "image_name": img_name,
                    "index": index,
                    "num_points": num_points,
                    "bboxes_coords": torch.tensor(bboxes_coords[None, None, ...]).float(),
                }

        else:
            # If not using prompts, return the image and ground truth only.
            input_image = np.stack([img, img, img], axis=0)
            return {
                "image": torch.tensor(input_image).float(),  # Shape: [3, H, W]
                "gt": torch.tensor(gt[None, :, :]).long(),
                "num_points": 0,
                "image_name": img_name,
                "index": index,
            }
