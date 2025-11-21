import torch
import os
import csv
import yaml
from torch.utils.data import DataLoader
from datasets.datasets import CoroDataset
import numpy as np
import random
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.transforms import AsDiscrete, Activations, Compose
from monai.data import decollate_batch
import train_utils as utils
import wandb
import time
from tqdm import tqdm
import argparse

execution_timestamp = time.strftime("%Y%m%d-%H%M%S")


def load_config(config_path="train_config.yaml"):
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


def write_to_csv(exp_name, fold, dice, precision, recall, accuracy, f1, specificity, save_path):
    """Writes evaluation metrics to CSV file."""
    csv_file = os.path.join(save_path, f"results_{execution_timestamp}.csv")
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['exp_name', 'fold', 'dice', 'precision', 'recall', 'accuracy', 'f1', 'specificity'])
        writer.writerow([exp_name, fold, dice, precision, recall, accuracy, f1, specificity])


def setup_model(config, device):
    """Initializes and returns the model based on configuration."""
    return utils.create_model(
        model_name=config['model_name'],
        device=device,
        pretrained=config['pretrained'],
        checkpoint=config.get('initial_checkpoint'),
        dropout=config['dropout'],
        use_adapter=config['use_adapters'],
        use_conv_adapter=config['use_conv_adapters'],
        use_sam_med_adapter=config['use_sam_med_adapters'],
        update_adapter_only=config['train_only_adapters'],
        channel_reduction=config['channel_reduction']
    )


def process_batch(batch, model, device, use_amp, criterion, config):
    """Processes a single batch through the model and computes loss."""
    image = batch["image"].to(device)
    gt = batch["gt"].to(device)

    points = (batch["point_coords"].to(device), batch["point_labels"].to(device)) if config['use_points'] else None
    boxes = batch["bboxes_coords"].to(device) if config['use_boxes'] else None

    if use_amp:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            if config['use_points'] or config['use_boxes']:
                outputs = model(image, points=points, boxes=boxes)
            else:
                outputs = model(image)
            loss = criterion(outputs, gt)
    else:
        if config['use_points'] or config['use_boxes']:
            outputs = model(image, points=points, boxes=boxes)
        else:
            outputs = model(image)
        loss = criterion(outputs, gt)

    return outputs, loss


def train_or_validate(model, dataloader, optimizer, criterion, scaler, device, dice_metric, cm_metric, post_trans,
                      config, train=True):
    """Performs one epoch of training or validation."""
    phase = "Train" if train else "Validation"
    print(f"{phase} phase started")
    model.train() if train else model.eval()

    loss_list = []
    dice_metric.reset()
    cm_metric.reset()

    for batch in tqdm(dataloader):
        if train:
            optimizer.zero_grad()

        if train:
            outputs, loss = process_batch(batch, model, device, config['use_amp'], criterion, config)
        else:
            with torch.no_grad():
                outputs, loss = process_batch(batch, model, device, config['use_amp'], criterion, config)

        if train:
            if config['use_amp']:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        loss_list.append(loss.item())

        outputs = [post_trans(i) for i in decollate_batch(outputs)]
        dice_metric(y_pred=outputs, y=batch["gt"].to(device))
        cm_metric(y_pred=outputs, y=batch["gt"].to(device))

    avg_loss = sum(loss_list) / len(loss_list)
    avg_dice = dice_metric.aggregate().item()
    confusion_matrix_metrics = cm_metric.aggregate()
    avg_accuracy = confusion_matrix_metrics[0].item()
    avg_precision = confusion_matrix_metrics[1].item()
    avg_recall = confusion_matrix_metrics[2].item()
    avg_specificity = confusion_matrix_metrics[3].item()
    avg_f1 = confusion_matrix_metrics[4].item()

    return avg_loss, avg_dice, avg_precision, avg_recall, avg_accuracy, avg_specificity, avg_f1


def save_model(epoch, model, optimizer, path, name, suffix):
    """Saves model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, os.path.join(path, f'{name}_{suffix}.pt'))


def create_dataloaders(train_path, val_path, config):
    """Creates training and validation dataloaders."""

    use_aug = False
    if config.get('n_folds') == 1:
        use_aug = True  # Use augmentation only in single-fold training

    train_dataset = CoroDataset(
        train_path,
        image_size=config['img_size'],
        use_augmented=use_aug,
        num_pos_pts=config.get('num_pos_pts'),
        num_neg_pts=config.get('num_neg_pts'),
        num_tips=config.get('num_tips'),
        num_other=config.get('num_other'),
        divide_channels=config['divide_channels'],
        points_shift=config['points_shift'],
        use_prompts=config['use_prompts']
    )

    val_dataset = CoroDataset(
        val_path,
        image_size=config['img_size'],
        use_augmented=False,
        num_pos_pts=config.get('num_pos_pts'),
        num_neg_pts=config.get('num_neg_pts'),
        num_tips=config.get('num_tips'),
        num_other=config.get('num_other'),
        divide_channels=config['divide_channels'],
        points_shift=0,
        use_prompts=config['use_prompts']
    )

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(config['seed'])

    train_dataloader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        worker_init_fn=seed_worker, generator=g)
    val_dataloader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        worker_init_fn=seed_worker, generator=g)

    return train_dataloader, val_dataloader


def get_adapter_type_string(config):
    """Determines adapter type string for experiment naming."""
    if config['use_conv_adapters']:
        adapter_type = f"CoroSAM_Conv_HidDim={config['channel_reduction']}"
    elif config['use_sam_med_adapters']:
        adapter_type = 'SAMMed2d'
    elif config['use_adapters']:
        adapter_type = 'Original'
    else:
        adapter_type = 'NoAdapter'

    if config['use_prompts'] and config['use_points']:
        adapter_type += '_original_prompts'
    if not config['use_prompts']:
        adapter_type += '_no_prompts'

    return adapter_type


def run_training(config, fold=None, device="cuda"):
    """Main training loop for a single fold or entire training set."""
    # Setup model and training components
    model = setup_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if config['use_amp'] and device == 'cuda' else None

    # Setup losses and metrics
    loss = DiceFocalLoss(sigmoid=True, alpha=config.get('alpha_focal'))
    dice_metric = DiceMetric(reduction='mean')
    cm_metric = ConfusionMatrixMetric(
        reduction='mean',
        compute_sample=True,
        metric_name=["accuracy", "precision", "sensitivity", "specificity", "f1_score"]
    )
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # Setup paths
    if fold is not None:
        train_path = os.path.join(config['k_fold_path'], f"set{fold + 1}", "train")
        val_path = os.path.join(config['k_fold_path'], f"set{fold + 1}", "val")
        fold_suffix = f"_fold_{fold}"
    else:
        train_path = config['train_path']
        val_path = config['val_path']
        fold_suffix = ""

    train_dataloader, val_dataloader = create_dataloaders(train_path, val_path, config)

    # Setup logging
    ckpt_path = str(os.path.join(config['checkpoints_path'], config['exp_name']))
    if config['save_models'] and not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    timestamp = utils.get_timestamp()
    adapter_type = get_adapter_type_string(config)
    exp_name = f"{config['model_name']}_{config['exp_name']}_{adapter_type}_{timestamp}"

    if config['save_models']:
        with open(os.path.join(ckpt_path, f'{exp_name}.yaml'), 'w') as yaml_file:
            yaml.dump(config, yaml_file)

    # Setup WandB
    if config['use_wandb']:
        wandb.init(
            project=config['proj_name'],
            config={
                "exp_name": exp_name,
                "model": config['model_name'],
                "sammed_adapt": config['use_sam_med_adapters'],
                "conv_adapt": config['use_conv_adapters'],
                "adapt": config['use_adapters'],
                "learning_rate": config['lr'],
                "weight_decay": config['wd'],
                "batch_size": config['batch_size'],
                "dropout": config['dropout'],
                "alpha_focal": config.get('alpha_focal'),
                "seed": config['seed'],
                "epochs": config['epochs'],
                "use_pts": config['use_points'],
                "use_box": config['use_boxes'],
                "pos_pts": config.get('num_pos_pts'),
                "neg_pts": config.get('num_neg_pts'),
                "tips": config.get('num_tips'),
                "other": config.get('num_other'),
                "points_shift": config['points_shift'],
                "divide_channels": config['divide_channels'],
                "use_prompts": config['use_prompts'],
                "box_shift": config.get('bbox_shift', 10)
            },
            reinit=True
        )
        wandb.run.name = f"{exp_name}{fold_suffix}"

    best_val_dice = 0

    # Main training loop
    for epoch in range(config['epochs']):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        train_loss, train_dice, train_precision, train_recall, train_accuracy, train_specificity, train_f1 = (
            train_or_validate(model, train_dataloader, optimizer, loss, scaler, device,
                              dice_metric, cm_metric, post_trans, config, train=True))

        val_loss, val_dice, val_precision, val_recall, val_accuracy, val_specificity, val_f1 = (
            train_or_validate(model, val_dataloader, optimizer, loss, scaler, device,
                              dice_metric, cm_metric, post_trans, config, train=False))

        lr_scheduler.step()

        if config['use_wandb']:
            wandb.log({
                'epoch': epoch + 1,
                'train_dice': train_dice,
                'train_loss': train_loss,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_accuracy': train_accuracy,
                'train_f1': train_f1,
                'train_specificity': train_specificity,
                'val_dice': val_dice,
                'val_loss': val_loss,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_accuracy': val_accuracy,
                'val_specificity': val_specificity,
                'val_f1': val_f1
            })

        if config['save_models']:
            save_model(epoch + 1, model, optimizer, ckpt_path, exp_name,
                       f'{fold_suffix}_last' if fold is not None else 'last')
            print(f"Last model saved: {exp_name}{fold_suffix}_last.pt")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_precision = val_precision
            best_val_recall = val_recall
            best_val_acc = val_accuracy
            best_val_f1 = val_f1
            best_val_specificity = val_specificity
            if config['save_models']:
                save_model(epoch + 1, model, optimizer, ckpt_path, exp_name,
                           f'{fold_suffix}_best' if fold is not None else 'best')
                print(f"Best model saved: {exp_name}{fold_suffix}_best.pt")

        print(f"Best validation dice{fold_suffix}: {best_val_dice}")
        print(f"Current validation dice{fold_suffix}: {val_dice}")

    # Save results to CSV
    fold_num = fold + 1 if fold is not None else None
    write_to_csv(exp_name, fold_num, best_val_dice, best_val_precision, best_val_recall,
                 best_val_acc, best_val_f1, best_val_specificity, ckpt_path)

    if config['use_wandb']:
        wandb.finish()

    return best_val_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model with YAML configuration')
    parser.add_argument('--config', type=str, default='train_config.yaml',
                        help='Path to configuration YAML file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set seeds for reproducibility
    set_seeds(config['seed'])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Check if k-fold cross-validation is enabled
    if config.get('n_folds', 1) > 1:
        print(f"Starting {config['n_folds']}-fold cross-validation")
        for fold in range(config['n_folds']):
            print(f"\nStarting fold {fold + 1}/{config['n_folds']}")
            run_training(config, fold=fold, device=device)
    else:
        print("Starting single training run")
        run_training(config, device=device)