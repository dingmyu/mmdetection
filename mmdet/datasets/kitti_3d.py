import numpy as np

from .custom import CustomDataset
from .registry import DATASETS
import mmcv


@DATASETS.register_module
class KittiDataset(CustomDataset):

    CLASSES = ('Car', 'Cyclist', 'Pedestrian')

    def load_annotations(self, ann_file):
        ann = mmcv.load(ann_file)
        # self.img_ids = list(range(len(ann)))
        # self.cat_ids = list(range(3))
        # self.cat2label = {
        #     cat_id: i + 1
        #     for i, cat_id in enumerate(self.cat_ids)
        # }
        return ann


    # def get_ann_info(self, idx):
    #     return self.img_infos[idx]['ann']

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if self.filter_empty_gt and len(img_info['ann']['labels']) == 0:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def get_file_name(self, idx):
        return self.img_infos[idx]['filename']