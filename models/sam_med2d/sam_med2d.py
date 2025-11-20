import torch
import torch.nn as nn
from functools import partial
from models.sam_med2d.image_encoder import ImageEncoderViT
from models.sam_med2d.transformer import TwoWayTransformer
from models.sam_med2d.mask_decoder import MaskDecoder
from models.sam_med2d.prompt_encoder import PromptEncoder
import torch.nn.functional as F


class SAMMed2D(nn.Module):
    def __init__(self, in_chans=1, use_adapter=True):
        super().__init__()
        self.in_chans = in_chans
        self.use_adapter = use_adapter

        self.image_encoder = ImageEncoderViT(
            img_size=256,
            in_chans=in_chans,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            embed_dim=768,
            depth=12,
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=(2, 5, 8, 11),
            window_size=14,
            out_chans=256,
            adapter_train=self.use_adapter,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(16, 16),
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
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        masks = F.interpolate(low_res_masks, (256, 256), mode="bilinear", align_corners=False)

        return masks


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SAMMed2D(in_chans=1, use_adapter=True)
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
