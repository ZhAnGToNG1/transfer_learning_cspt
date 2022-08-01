_base_ = [
    '../_base_/models/mask_rcnn_vit_base_fpn.py',
    '../_base_/datasets/nwpuvhr.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        num_classes=10,
        mim_model=None
    ),
    roi_head=dict(
        bbox_head=dict(
        num_classes=10
        )
    )
)