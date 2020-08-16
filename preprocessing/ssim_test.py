import os
import cv2
from skimage.metrics import structural_similarity
import numpy as np

if __name__ == '__main__':
    # TODO: This file will be deleted
    root_dir = 'F:\\Celeb-DF-v2'
    sample_path = os.path.join(root_dir, 'samples')
    ori_path = os.path.join(sample_path, 'id0_0005_0_1.png')
    fake_path = os.path.join(sample_path, 'id0_id23_0005_0_1.png')
    diff_path = 'diff.png'
    img1 = cv2.imread(ori_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(fake_path, cv2.IMREAD_COLOR)
    try:
        d, a = structural_similarity(img1, img2, multichannel=True, full=True)
        a = 1 - a
        diff = (a * 255).astype(np.uint8)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(diff_path, diff)
    except:
        pass
