import math

import albumentations as A
import cv2
from albumentations import DualTransform
from albumentations.pytorch import ToTensorV2


def isotropically_resize_image_v2(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if min(w, h) == size:
        return img
    if w < h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (math.ceil(w), math.ceil(h)), interpolation=interpolation)
    return resized


class IsotropicResizeV2(DualTransform):

    def __init__(self, min_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResizeV2, self).__init__(always_apply, p)
        self.min_side = min_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, **params):
        return isotropically_resize_image_v2(img, size=self.min_side, interpolation_down=self.interpolation_down,
                                             interpolation_up=self.interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return "max_side", "interpolation_down", "interpolation_up"

    def apply_to_bbox(self, bbox, **params):
        pass

    def apply_to_keypoint(self, keypoint, **params):
        pass

    def get_params_dependent_on_targets(self, params):
        pass


def create_train_transform(model_cfg):
    size = model_cfg['input_size'][1]
    mean = model_cfg['mean']
    std = model_cfg['std']
    return A.Compose([
        # A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        # A.GaussNoise(p=0.1),
        # A.Blur(blur_limit=3, p=0.05),
        A.HorizontalFlip(),
        A.OneOf([
            IsotropicResizeV2(min_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            IsotropicResizeV2(min_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            IsotropicResizeV2(min_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
        ], p=1),
        A.CenterCrop(height=size, width=size),
        # A.OneOf([A.RandomBrightnessContrast(), A.FancyPCA(), A.HueSaturationValue()], p=0.7),
        # A.ToGray(p=0.2),
        # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def create_val_test_transform(model_cfg):
    size = model_cfg['input_size'][1]
    mean = model_cfg['mean']
    std = model_cfg['std']
    return A.Compose([
        IsotropicResizeV2(min_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        A.CenterCrop(height=size, width=size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
