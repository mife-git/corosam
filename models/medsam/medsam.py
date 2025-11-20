import torch
import torch.nn as nn
from models.medsam.tiny_vit import TinyViT
from models.medsam.transformer import TwoWayTransformer
from models.medsam.mask_decoder import MaskDecoder
from models.medsam.prompt_encoder import PromptEncoder


def postprocess_masks(masks, new_size, original_size):
    """
    Do cropping and resizing
    """
    # Crop
    masks = masks[:, :, :new_size[0], :new_size[1]]
    # Resize
    masks = torch.nn.functional.interpolate(
        masks,
        size=(original_size[0], original_size[1]),
        mode="bilinear",
        align_corners=False,
    )

    return masks


class MedSAM(nn.Module):
    def __init__(self, in_chans=1, dropout=0., dropout_path=0., use_adapter=True, use_conv_adapter=False,
                 use_sam_med_adapter=False, channel_reduction=0.5):
        super().__init__()
        self.in_chans = in_chans
        self.dropout = dropout
        self.dropout_path = dropout_path
        self.use_adapter = use_adapter
        self.use_conv_adapter = use_conv_adapter
        self.use_sam_med_adapter = use_sam_med_adapter
        self.channel_reduction = channel_reduction

        self.image_encoder = TinyViT(
            img_size=256,
            in_chans=self.in_chans,
            embed_dims=[
                64,  # (64, 256, 256)
                128,  # (128, 128, 128)
                160,  # (160, 64, 64)
                320  # (320, 64, 64)
            ],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.,
            drop_rate=self.dropout,
            drop_path_rate=self.dropout_path,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
            use_adapter=self.use_adapter,
            use_conv_adapter=self.use_conv_adapter,
            use_sam_med_adapter=self.use_sam_med_adapter,
            channel_reduction=self.channel_reduction
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
                if_mask_decoder_adapter=False
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(256, 256),
            mask_in_chans=16
        )

    def forward(self, image, boxes=None, points=None, masks=None):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )  # (B, 1, 256, 256)

        return low_res_masks


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MedSAM(in_chans=1, dropout=0.1, dropout_path=0.1, use_adapter=True)
    model = model.to(device)

    batch_size = 2
    image_size = (256, 256)
    dummy_image = torch.randn(batch_size, 1, *image_size).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output_masks = model(image=dummy_image)

    print("Output masks shape:", output_masks.shape)
    print("Output masks :", output_masks)
