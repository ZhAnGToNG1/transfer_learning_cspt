# Copyright (c) OpenMMLab. All rights reserved.


from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class VaihingenDataset(CustomDataset):
    CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree',
               'car', 'clutter')

    PALETTE = [[255, 255, 255], [0, 0, 255], [0, 0, 255], [0, 255, 0],
               [255, 255, 0], [255, 0, 0]]

    def __init__(self, **kwargs):
        super(VaihingenDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)






