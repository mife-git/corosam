import torch
import os
import csv
import yaml
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets.datasets import CoroDataset
import numpy as np
import random
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.transforms import AsDiscrete, Activations, Compose
from monai.data import decollate_batch
import training.train_utils as utils
import time
from tqdm import tqdm
import cv2
from scipy import stats

execution_timestamp = time.strftime("%Y%m%d-%H%M%S")


def load_config(config_path="test_config.yaml"):
    """Loads configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seeds(seed):
    """Set all random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def write_to_csv(exp_name, dice, precision, recall, accuracy, f1, specificity, results_path):
    """Write aggregated results to CSV."""
    csv_file = os.path.join(results_path, f"results_{execution_timestamp}.csv")

    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(['exp_name', 'dice', 'precision', 'recall', 'accuracy', 'f1', 'specificity'])

        writer.writerow([exp_name, dice, precision, recall, accuracy, f1, specificity])


def save_prediction_fig(image, gt, pred, index, results_path):
    """Save prediction visualization comparing original, GT, and prediction."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[1].imshow(gt, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title("Predicted Output")
    axes[2].axis('off')

    # Save individual prediction mask
    cv2.imwrite(
        os.path.join(results_path, f"prediction_{index}.png"),
        np.array(pred * 255).astype(np.uint8)
    )

    # Save comparison figure
    plt.savefig(os.path.join(results_path, f"comparison_{index}.png"), bbox_inches='tight', dpi=150)
    plt.close()


def setup_model(config, device):
    """Initialize and load the model with trained weights."""
    model = utils.create_model(
        model_name=config['model_name'],
        device=device,
        pretrained=True,
        checkpoint=config['checkpoint'],
        use_adapter=config['use_adapters'],
        use_conv_adapter=config['use_conv_adapters'],
        use_sam_med_adapter=config['use_sam_med_adapters'],
        channel_reduction=config['channel_reduction']
    )
    return model


def process_batch(batch, model, device, config):
    """Process a single batch through the model and measure inference time."""
    image = batch["image"].to(device)
    gt = batch["gt"].to(device)

    points = (batch["point_coords"].to(device), batch["point_labels"].to(device)) if config['use_points'] else None
    boxes = batch["bboxes_coords"].to(device) if config['use_boxes'] else None

    start_time = time.time()

    if config['use_points'] or config['use_boxes']:
        outputs = model(image, points=points, boxes=boxes)
    else:
        outputs = model(image)

    end_time = time.time()

    return outputs, gt, end_time - start_time


def evaluate_model(model, dataloader, device, config, post_trans):
    """Evaluate model on test dataset and compute metrics."""
    print("Evaluation started")

    model.eval()

    inference_times = []
    image_names = []
    all_dice_scores = []
    all_metrics = []

    results_path = config['results_path']

    # Prepare CSV files
    per_pixel_score_file = os.path.join(results_path, f"scores_{execution_timestamp}.csv")
    per_image_file = os.path.join(results_path, f"per_image_metrics_{execution_timestamp}.csv")

    write_pixel_header = not os.path.exists(per_pixel_score_file)
    score_file = open(per_pixel_score_file, 'a', newline='')
    score_writer = csv.writer(score_file)
    if write_pixel_header:
        score_writer.writerow(["image_name", "y_true", "y_score"])

    if not os.path.exists(per_image_file):
        with open(per_image_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_name', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score', 'Dice'])

    start_total = time.time()

    for batch in tqdm(dataloader):
        with torch.no_grad():
            outputs, gt, inference_duration = process_batch(batch, model, device, config)

        inference_times.append(inference_duration)
        image_names.append(batch["image_name"][0])  # Assuming batch size = 1

        # Save per-pixel scores for ROC/AUC analysis
        raw_outputs = [i.detach().cpu().numpy().flatten() for i in decollate_batch(torch.sigmoid(outputs))]
        gts = [i.detach().cpu().numpy().flatten() for i in decollate_batch(gt)]

        for image_name, y_score, y_true in zip(batch["image_name"], raw_outputs, gts):
            for s, t in zip(y_score, y_true):
                score_writer.writerow([image_name, int(t), float(s)])

        # Post-processing: apply sigmoid and threshold
        outputs = [post_trans(i) for i in decollate_batch(outputs)]

        # Compute per-image metrics
        all_dice_scores.append(DiceMetric()(y_pred=outputs, y=gt).item())

        metrics_per_sample = ConfusionMatrixMetric(
            compute_sample=True,
            metric_name=["accuracy", "precision", "sensitivity", "specificity", "f1_score"]
        )
        metrics_per_sample(y_pred=outputs, y=gt)
        per_img_metrics = [m.detach().cpu().item() for m in metrics_per_sample.aggregate()]
        per_img_dice = all_dice_scores[-1]

        all_metrics.append(per_img_metrics)

        # Write per-image results
        with open(per_image_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                image_names[-1],
                f"{per_img_metrics[0]:.4f}",  # Accuracy
                f"{per_img_metrics[1]:.4f}",  # Precision
                f"{per_img_metrics[2]:.4f}",  # Recall
                f"{per_img_metrics[3]:.4f}",  # Specificity
                f"{per_img_metrics[4]:.4f}",  # F1
                f"{per_img_dice:.4f}"  # Dice
            ])

        # Save prediction visualizations
        if config['save_predictions']:
            image = batch["image"][0][0]
            gt_img = batch["gt"][0][0]
            image_name = batch["image_name"][0][:-4]
            save_prediction_fig(image, gt_img, outputs[0][0].detach().cpu(), image_name, results_path)

    score_file.close()
    end_total = time.time()
    total_time = end_total - start_total

    # Compute aggregated statistics
    all_metrics = np.array(all_metrics)
    mean_metrics = np.mean(all_metrics, axis=0)
    std_metrics = np.std(all_metrics, axis=0)
    ci95 = stats.t.interval(0.95, len(all_metrics) - 1, loc=mean_metrics, scale=stats.sem(all_metrics, axis=0))

    mean_dice = np.mean(all_dice_scores)
    std_dice = np.std(all_dice_scores)
    ci_dice = stats.t.interval(0.95, len(all_dice_scores) - 1, loc=mean_dice, scale=stats.sem(all_dice_scores))

    # Save inference times
    times_file = os.path.join(results_path, f"inference_times_{execution_timestamp}.csv")
    with open(times_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'inference_time_sec'])
        for name, t in zip(image_names, inference_times):
            writer.writerow([name, f"{t:.6f}"])

    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total inference time: {total_time:.2f} sec")
    print(f"Average inference time per image: {mean_time:.6f} sec (± {std_time:.6f})\n")

    print(f"Dice:       {mean_dice:.4f} ± {std_dice:.4f} (95% CI: [{ci_dice[0]:.4f}, {ci_dice[1]:.4f}])")
    print(f"Accuracy:   {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f} (95% CI: [{ci95[0][0]:.4f}, {ci95[1][0]:.4f}])")
    print(f"Precision:  {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f} (95% CI: [{ci95[0][1]:.4f}, {ci95[1][1]:.4f}])")
    print(f"Recall:     {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f} (95% CI: [{ci95[0][2]:.4f}, {ci95[1][2]:.4f}])")
    print(f"Specificity:{mean_metrics[3]:.4f} ± {std_metrics[3]:.4f} (95% CI: [{ci95[0][3]:.4f}, {ci95[1][3]:.4f}])")
    print(f"F1 Score:   {mean_metrics[4]:.4f} ± {std_metrics[4]:.4f} (95% CI: [{ci95[0][4]:.4f}, {ci95[1][4]:.4f}])")
    print("=" * 50 + "\n")

    return mean_dice, mean_metrics[1], mean_metrics[2], mean_metrics[0], mean_metrics[3], mean_metrics[4]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test model with YAML configuration')
    parser.add_argument('--config', type=str, default='test_config.yaml',
                        help='Path to configuration YAML file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set seeds for reproducibility
    set_seeds(config['seed'])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Create results directory
    os.makedirs(config['results_path'], exist_ok=True)

    # Setup model
    model = setup_model(config, device)

    # Post-processing transform
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # Prepare summary CSV for parameter sweep
    summary_csv = os.path.join(config['results_path'], f"test_summary_{execution_timestamp}.csv")
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['num_tips', 'num_other', 'points_shift', 'dice', 'precision', 'recall',
                         'accuracy', 'specificity', 'f1'])

    # Parameter sweep (if configured)
    num_tips_list = config.get('num_tips_list', [None])
    num_other_list = config.get('num_other_list', [0])

    for num_tips in num_tips_list:
        for num_other in num_other_list:
            print(f"\n{'=' * 60}")
            print(f"Testing with num_tips = {num_tips}, num_other = {num_other}")
            print(f"{'=' * 60}")

            # Create test dataset with current parameters
            test_dataset = CoroDataset(
                config['test_path'],
                image_size=config['img_size'],
                use_augmented=False,
                num_pos_pts=config.get('num_pos_pts'),
                num_neg_pts=config.get('num_neg_pts'),
                num_tips=num_tips,
                num_other=num_other,
                divide_channels=config['divide_channels'],
                points_shift=config['points_shift'],
                use_prompts=config['use_prompts']
            )

            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

            # Evaluate
            val_dice, val_precision, val_recall, val_accuracy, val_specificity, val_f1 = evaluate_model(
                model, test_dataloader, device, config, post_trans
            )

            # Save to summary CSV
            with open(summary_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    num_tips if num_tips is not None else "None",
                    num_other if num_other is not None else "None",
                    config['points_shift'],
                    f"{val_dice:.4f}", f"{val_precision:.4f}", f"{val_recall:.4f}",
                    f"{val_accuracy:.4f}", f"{val_specificity:.4f}", f"{val_f1:.4f}"
                ])
