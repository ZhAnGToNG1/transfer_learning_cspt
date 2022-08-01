# Copyright (c) OpenMMLab. All rights reserved.


from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class GIDDataset(CustomDataset):
    """ISPRS Potsdam dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    CLASSES = ('industrial land', 'urban residential', 'rural residential', 'traffic land',
               'paddy field', 'irrigated land', 'dry cropland', 'garden plot', 'arbor woodland',
               'shrub land', 'natural grassland', 'artificial grassland', 'river',
               'lake', 'pond')

    PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
               [255, 255, 0], [255, 0, 0],[0,  200, 250],[0,     150, 200],
               [0, 0, 200],[200,  200, 0],[250,  200,    0],[150,  150, 250],
               [150, 0, 250],[200,  0, 200],[150, 200, 150]]

    def __init__(self, **kwargs):
        super(GIDDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)
