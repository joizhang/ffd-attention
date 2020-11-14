import math
import random

import albumentations as A
import cv2
import numpy as np
from albumentations import DualTransform
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import binary_dilation
from torchvision import transforms


def create_train_transform(input_size, model):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size[1], input_size[2])),
        transforms.ToTensor(),
        transforms.Normalize(mean=model.default_cfg['mean'], std=model.default_cfg['std'])
    ])
    mask_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((19, 19)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    return mask_transform, transform

def create_train_transform(model_cfg):
    size = model_cfg['input_size'][1]
    mean = model_cfg['mean']
    std = model_cfg['std']
    return A.Compose([
        A.ImageCompression(quality_lower=20, quality_upper=100, p=0.5),
        A.GaussNoise(p=0.1),
        A.GaussianBlur(blur_limit=3, p=0.05),
        A.HorizontalFlip(),
        A.OneOf([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
        ], p=1),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0),
        # A.ToGray(p=0.1),
        # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def create_val_test_transform(model_cfg):
    size = model_cfg['input_size'][1]
    mean = model_cfg['mean']
    std = model_cfg['std']
    return A.Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def prepare_bit_masks(mask):
    h, w = mask.shape
    mid_w = w // 2
    mid_h = w // 2
    masks = []
    ones = np.ones_like(mask)
    ones[:mid_h] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[mid_h:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:, :mid_w] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:, mid_w:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:mid_h, :mid_w] = 0
    ones[mid_h:, mid_w:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:mid_h, mid_w:] = 0
    ones[mid_h:, :mid_w] = 0
    masks.append(ones)
    return masks


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def remove_eyes(image, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[:2]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image


def remove_nose(image, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[:2]
    x3, y3 = landmarks[2]
    mask = np.zeros_like(image[..., 0])
    x4 = int((x1 + x2) / 2)
    y4 = int((y1 + y2) / 2)
    line = cv2.line(mask, (x3, y3), (x4, y4), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image


def remove_mouth(image, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[-2:]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 3)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image


def remove_landmark(image, landmarks):
    if random.random() > 0.5:
        image = remove_eyes(image, landmarks)
    elif random.random() > 0.5:
        image = remove_mouth(image, landmarks)
    elif random.random() > 0.5:
        image = remove_nose(image, landmarks)
    return image


def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


class IsotropicResize(DualTransform):

    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down,
                                          interpolation_up=interpolation_up)

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
