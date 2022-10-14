
import os
from typing import Optional, overload
import argparse

import cv2
import numpy as np

from .. import register_dataset
from ..dataset_base import BaseImageDataset
from ...transforms import image as tf

VOC_CLASS_LIST = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted_plant', 'sheep', 'sofa', 'train',
                  'tv_monitor']

NYUV2_CLASS_LIST = []


class CenterCrop(object):
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, inputs, target_label):
        h, w, _ = inputs.shape
        th, tw = self.size
        x = int(round((w - tw) / 2.))
        y = int(round((h - th) / 2.))

        inputs = inputs[y: y + th, x: x + tw]
        target_label = target_label[y: y + th, x: x + tw]

        return inputs, target_label


class OfficialCrop(object):
    """
    [h_range=[45, 471], w_range=[41, 601]] -> (427, 561)
    official cropping to get best depth
    """
    def __call__(self, inputs, target_label):
        h, w, _ = inputs.shape
        assert h > 471 and w > 601, "inputs height must > 417, width > 601"
        inputs = inputs[45:471 + 1, 41:601 + 1]
        target_label = target_label[45:471 + 1, 41:601 + 1]
        return inputs, target_label

class DepthPredCrop(object):
    """
    640 * 480 -> dowmsample(320, 240) -> crop(304, 228) -> upsample(640, 480)
    """
    def __init__(self):
        self.center_crop = CenterCrop((228, 304))

    def __call__(self, inputs, target_label):
        inputs = cv2.resize(inputs, (320, 240), interpolation=cv2.INTER_LINEAR)
        target_label = cv2.resize(target_label, (320, 240), interpolation=cv2.INTER_NEAREST)

        inputs, target_label = self.center_crop(inputs, target_label)

        inputs = cv2.resize(inputs, (640, 480), interpolation=cv2.INTER_LINEAR)
        target_label = cv2.resize(target_label, (640, 480), interpolation=cv2.INTER_NEAREST)
        return inputs, target_label


@register_dataset("nyuv2", "segmentation")
class NYUV2Dataset(BaseImageDataset):
    """
        Dataset class for the NYU-V2 dataset

        The structure NYUV2 dataset should be something like this
        + pascal_voc/VOCdevkit/VOC2012/
        + --- Annotations
        + --- JPEGImages
        + --- SegmentationClass
        + --- SegmentationClassAug_Visualization/
        + --- ImageSets
        + --- list
        + --- SegmentationClassAug
        + --- SegmentationObject

    """
    def __init__(self, opts, is_training: Optional[bool] = True, is_evaluation: Optional[bool] = False):
        """

        :param opts: arguments
        :param is_training: Training or validation mode
        :param is_evaluation: Evaluation mode
        """
        super(NYUV2Dataset, self).__init__(opts=opts, is_training=is_training, is_evaluation=is_evaluation)
        self.channels = getattr(opts, "dataset.nyuv2.channels", ['rgb', 'hha'])  # nyuv2数据通道定义
        # transform 未定义
        root = self.root
        self.classes = 40

        self.nyuv2_root_dir = os.path.join(root, 'nyuv2')
        imlist_name = 'train.txt'
        imglist_fp = os.path.join(self.nyuv2_root_dir, imlist_name)
        imglist = self.read_imglist(imglist_fp)
        # voc_root_dir = os.path.join(root, 'VOC2012')
        # voc_list_dir = os.path.join(voc_root_dir, 'list')

        crop_process = None
        crop_paras = dict(type='official_origin', padding_size=(480, 640))
        if crop_paras['type'] == "blank_crop":
            crop_process = CenterCrop(crop_paras['center_crop_size'])
        elif crop_paras['type'] == "official_crop":
            crop_process = OfficialCrop()
        elif crop_paras['type'] == "depth_pred_crop":
            crop_process = DepthPredCrop()

        if self.is_training:
            # use the NYUV2 Dataset train data with augmented data
            data_file = os.path.join(self.nyuv2_root_dir, 'train.txt')
        else:
            data_file = os.path.join(self.nyuv2_root_dir, 'val.txt')

        self.images = []
        self.masks = []
        with open(data_file, 'r') as lines:
            for line in lines:
                rgb_img_loc = line.strip()
                mask_img_loc = rgb_img_loc
                # assert os.path.isfile(rgb_img_loc), 'RGB file does not exist at: {}'.format(rgb_img_loc)
                # assert os.path.isfile(mask_img_loc), 'Mask image does not exist at: {}'.format(rgb_img_loc)
                self.images.append(rgb_img_loc)
                self.masks.append(mask_img_loc)

        self.use_coco_data = False  # 不使用COCO_DATA
        self.ignore_label = 255
        self.bgrnd_idx = 0
        setattr(opts, "model.segmentation.n_classes", self.classes)

    def read_inpus(self, imgname):
        inputs = []
        if 'rgb' in self.channels:
            img_fp = os.path.join(self.nyuv2_root_dir, 'image', imgname + '.png')
            img = cv2.imread(img_fp).astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            inputs.append(img)
        if 'hha' in self.channels:
            hha_fp = os.path.join(self.nyuv2_root_dir, 'hha', imgname + '.png')
            ahh = cv2.imread(hha_fp).astype(np.float32)     # cv2.read will change hha into ahh
            hha = cv2.cvtColor(ahh, cv2.COLOR_BGR2RGB)
            inputs.append(hha)
        if "depth" in self.channels:
            dep_fp = os.path.join(self.nyuv2_root_dir, 'depth', imgname + '.png')
            dep = cv2.imread(dep_fp, cv2.IMREAD_UNCHANGED).astype(np.float32)
            dep = np.expand_dims(dep, axis=-1)
            inputs.append(dep)
        assert 0 < len(self.channels) == len(inputs), \
            "NYU Dataset input channels must be in ['rgb', 'hha', 'depth']"
        img = np.concatenate(inputs, axis=-1)

        mask_fp = os.path.join(self.nyuv2_root_dir, 'label' + str(self.classes), imgname + '.png')
        mask = cv2.imread(mask_fp, cv2.IMREAD_GRAYSCALE)
        mask -= 1       # 0->255

        return img, mask

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument('--dataset.pascal.use-coco-data', action='store_true', help='Use MS-COCO data for training')
        group.add_argument('--dataset.pascal.coco-root-dir', type=str, default=None, help='Location of MS-COCO data')
        return parser

    def training_transforms(self, size: tuple, ignore_idx: Optional[int] = 255):
        aug_list = [
                tf.RandomResize(opts=self.opts),
                tf.RandomCrop(opts=self.opts, size=size),
                tf.RandomHorizontalFlip(opts=self.opts),
                tf.NumpyToTensor(opts=self.opts)
            ]

        return tf.Compose(opts=self.opts, img_transforms=aug_list)

    def validation_transforms(self, size: tuple, *args, **kwargs):
        aug_list = [
            tf.Resize(opts=self.opts, size=size),
            tf.NumpyToTensor(opts=self.opts)
        ]
        return tf.Compose(opts=self.opts, img_transforms=aug_list)

    def evaluation_transforms(self, size: tuple, *args, **kwargs):
        aug_list = []
        if getattr(self.opts, "evaluation.segmentation.resize_input_images", False):
            aug_list.append(tf.Resize(opts=self.opts, size=size))

        aug_list.append(tf.NumpyToTensor(opts=self.opts))
        return tf.Compose(opts=self.opts, img_transforms=aug_list)

    def __getitem__(self, batch_indexes_tup):
        crop_size_h, crop_size_w, img_index = batch_indexes_tup
        crop_size = (crop_size_h, crop_size_w)

        if self.is_training:
            _transform = self.training_transforms(size=crop_size, ignore_idx=self.ignore_label)
        elif self.is_evaluation:
            _transform = self.evaluation_transforms(size=crop_size)
        else:
            _transform = self.validation_transforms(size=crop_size)

        mask = self.read_mask(self.masks[img_index])
        img = self.read_image(self.images[img_index])

        img, mask = self.read_inpus()

        im_height, im_width = img.shape[:2]

        data = {
            "image": img,
            "mask": None if self.is_evaluation else mask
        }

        data = _transform(data)

        if self.is_evaluation:
            # for evaluation purposes, resize only the input and not mask
            data["mask"] = mask

        data["label"] = data["mask"]
        del data["mask"]

        if self.is_evaluation:
            img_name = self.images[img_index].split(os.sep)[-1].replace('jpg', 'png')
            data["file_name"] = img_name
            data["im_width"] = im_width
            data["im_height"] = im_height

        return data

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        from utils.tensor_utils import tensor_size_from_opts
        im_h, im_w = tensor_size_from_opts(opts=self.opts)

        if self.is_training:
            transforms_str = self.training_transforms(size=(im_h, im_w))
        elif self.is_evaluation:
            transforms_str = self.evaluation_transforms(size=(im_h, im_w))
        else:
            transforms_str = self.validation_transforms(size=(im_h, im_w))

        return "{}(\n\troot={}\n\tis_training={}\n\tsamples={}\n\tuse_coco={}\n\ttransforms={}\n)".format(
            self.__class__.__name__,
            self.root,
            self.is_training,
            len(self.images),
            self.use_coco_data,
            transforms_str
        )
