import argparse
import json
import os
from functools import partial
from glob import glob
from multiprocessing.pool import Pool
from pathlib import Path

import cv2
from tqdm import tqdm

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def extract_video(param, root_dir, crops_dir):
    video, bboxes_path = param
    video_name = os.path.splitext(os.path.basename(video))[0]
    img_dir = os.path.join(root_dir, crops_dir, video_name)
    if os.path.exists(img_dir):
        return
    os.makedirs(img_dir, exist_ok=True)

    with open(bboxes_path, "r") as bbox_f:
        bboxes_dict = json.load(bbox_f)

    capture = cv2.VideoCapture(video)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(frames_num):
        capture.grab()
        if i % 10 != 0:
            continue
        success, frame = capture.retrieve()
        if not success or str(i) not in bboxes_dict:
            continue
        crops = []
        bboxes = bboxes_dict[str(i)]
        if bboxes is None:
            continue
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
            w = xmax - xmin
            h = ymax - ymin
            p_h = h // 3
            p_w = w // 3
            crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
            h, w = crop.shape[:2]
            crops.append(crop)
        for j, crop in enumerate(crops):
            cv2.imwrite(os.path.join(img_dir, "{}_{}.png".format(i, j)), crop)


def get_video_paths(root_dir):
    paths = []
    for video_fold in os.listdir(root_dir):
        if 'real' not in os.path.basename(video_fold) and 'synthesis' not in os.path.basename(video_fold):
            continue
        for video_path in os.listdir(video_fold):
            dir = Path(json_path).parent
            with open(json_path, "r") as f:
                metadata = json.load(f)
            for k, v in metadata.items():
                original = v.get("original", None)
                if not original:
                    original = k
                bboxes_path = os.path.join(root_dir, "boxes", original[:-4] + ".json")
                if not os.path.exists(bboxes_path):
                    continue
                paths.append((os.path.join(dir, k), bboxes_path))

    return paths


def main():
    parser = argparse.ArgumentParser(description="Extracts crops from video")
    parser.add_argument("--root-dir", help="root directory")
    parser.add_argument("--crops-dir", help="crops directory")
    args = parser.parse_args()
    os.makedirs(os.path.join(args.root_dir, args.crops_dir), exist_ok=True)
    params = get_video_paths(args.root_dir)
    with Pool(processes=1) as p:
        with tqdm(total=len(params)) as pbar:
            for v in p.imap_unordered(partial(extract_video, root_dir=args.root_dir, crops_dir=args.crops_dir), params):
                pbar.update()


if __name__ == '__main__':
    main()
