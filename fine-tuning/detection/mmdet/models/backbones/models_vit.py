# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from ..builder import BACKBONES
from .pos_embed import interpolate_pos_embed

@BACKBONES.register_module()
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, embed_dim=768,out_indices = [3, 5, 7, 11], norm_layer=partial(nn.LayerNorm, eps=1e-6),num_classes = 1, mim_model='',**kwargs):
        super(VisionTransformer, self).__init__(embed_dim=embed_dim, norm_layer = norm_layer, num_classes=num_classes,**kwargs)

        self.out_indices = out_indices

        # -----------------------------
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            nn.SyncBatchNorm(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        if mim_model is not None:
            self.load_pretrained_weight(ckpt_path=mim_model)



    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        Hp ,Wp = 32,32

        assert  Hp * Wp == x.shape[1], '32x32'

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                xp = x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                features.append(xp.contiguous())

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        return tuple(features)


    def init_weights(self,pretrained=None):
        return

    def load_pretrained_weight(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % ckpt_path)
        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint
        state_dict = self.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(self, checkpoint_model)

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

