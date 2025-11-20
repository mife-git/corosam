import torch
from datetime import datetime
import yaml


def create_model(model_name, device, checkpoint=None, pretrained=True, use_adapter=False,
                 use_conv_adapter=False, use_sam_med_adapter=False, update_adapter_only=False,
                 channel_reduction = 0.5,
                 input_channels=3, num_classes=1, dropout=0., train_neck=True):
    if "MedSAM" in model_name:
        from models.medsam.medsam import MedSAM
        net = MedSAM(in_chans=input_channels,
                     use_adapter=use_adapter,
                     use_conv_adapter=use_conv_adapter,
                     use_sam_med_adapter=use_sam_med_adapter,
                     dropout=dropout,
                     dropout_path=dropout,
                     channel_reduction=channel_reduction)
        if pretrained:
            try:
                net.load_state_dict(torch.load(checkpoint, map_location="cpu")['model'])
            except Exception as e:
                print(e)
                net.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)
        if use_adapter and update_adapter_only:
            for name, param in net.image_encoder.named_parameters():
                if "Adapter" not in name:  # only update parameters in adapter for encoder
                    param.requires_grad = False
        if train_neck:
            for _, param in net.image_encoder.neck.named_parameters():
                param.requires_grad = True

    if "SAMMed2d" in model_name:
        from models.sam_med2d.sam_med2d import SAMMed2D
        net = SAMMed2D(
            in_chans=input_channels,
            use_adapter=use_adapter
        )
        if pretrained:
            net.load_state_dict(torch.load(checkpoint, map_location="cpu",  weights_only=False)['model'])
        if use_adapter and update_adapter_only:
            for name, param in net.image_encoder.named_parameters():
                if "Adapter" not in name:  # only update parameters in adapter for encoder
                    param.requires_grad = False

    return net.to(device)


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_serializable(obj):
    data = vars(obj)
    serializable_data = {}
    for key, value in data.items():
        try:
            yaml.dump({key: value})
            serializable_data[key] = value
        except TypeError:
            serializable_data[key] = str(value)
    return serializable_data
