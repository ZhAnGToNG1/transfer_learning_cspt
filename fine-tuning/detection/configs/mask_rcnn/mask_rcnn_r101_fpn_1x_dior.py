_base_ = ['../_base_/models/mask_rcnn_r50_fpn.py',
          '../_base_/schedules/schedule_1x.py',
          '../_base_/default_runtime.py',
          '../_base_/datasets/dior.py']
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),

    roi_head=dict(
        bbox_head=dict(
            num_classes=20,
        )
    )
)
