import os
from typing import Sequence, Optional, Dict, Callable
from PIL import Image
import tqdm
import numpy as np
from torch.utils import data
import torch

from .segmentation_list import SegmentationList


class Pipe(SegmentationList):
    CLASSES = ['pipe']
    ID_TO_TRAIN_ID = {i + 1: 0 for i in range(20)}
    TRAIN_ID_TO_COLOR = [(255, 0, 0) for _ in range(20)] + [[0, 0, 0]]
    EVALUATE_CLASSES = CLASSES

    def __init__(self, root, split='train', data_folder='ColorImages', label_folder='SegmentationMaps', **kwargs):
        assert split in ['train', 'val']

        data_list_file = os.path.join(root, "image_list", "{}.txt".format(split))
        self.split = split
        super(Pipe, self).__init__(root, Pipe.CLASSES, data_list_file, data_list_file,
                                   data_folder, label_folder,
                                   id_to_train_id=Pipe.ID_TO_TRAIN_ID,
                                   train_id_to_color=Pipe.TRAIN_ID_TO_COLOR, **kwargs)
