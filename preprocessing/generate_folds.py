import argparse
import json
import os
import random
import numpy as np
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path
from glob import glob

import cv2
import pandas as pd
from tqdm import tqdm

from preprocessing.generate_diffs import get_original_with_fakes

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract image diffs")
    parser.add_argument("--root-dir", help="root directory", default="")
    parser.add_argument("--out", type=str, default="folds02.csv", help="CSV file to save")
    parser.add_argument("--seed", type=int, default=777, help="Seed to split, default 777")
    parser.add_argument("--n_splits", type=int, default=16, help="Num folds, default 10")
    args = parser.parse_args()
    return args


def get_paths(vid, label, root_dir):
    ori_vid, fake_vid = vid
    ori_dir = os.path.join(root_dir, "crops", ori_vid)
    fake_dir = os.path.join(root_dir, "crops", fake_vid)
    data = []
    for frame in range(320):
        if frame % 10 != 0:
            continue
        for actor in range(2):
            image_id = "{}_{}.png".format(frame, actor)
            ori_img_path = os.path.join(ori_dir, image_id)
            fake_img_path = os.path.join(fake_dir, image_id)
            img_path = ori_img_path if label == 0 else fake_img_path
            try:
                # img = cv2.imread(img_path)[..., ::-1]
                if os.path.exists(img_path):
                    data.append([img_path, label, ori_vid])
            except:
                pass
    return data


def split_into_folds(args):
    sz = 50 // args.n_splits
    folds = []
    for fold in range(args.n_splits):
        folds.append(list(range(sz * fold, sz * fold + sz if fold < args.n_splits - 1 else 50)))
    return folds


def get_video_with_fold(args, folds):
    video_fold = {}
    for d in os.listdir(args.root_dir):
        if "dfdc" in d and 'zip' not in d:
            part = int(d.split("_")[-1])
            for f in os.listdir(os.path.join(args.root_dir, d)):
                if "metadata.json" in f:
                    with open(os.path.join(args.root_dir, d, "metadata.json")) as metadata_json:
                        metadata = json.load(metadata_json)

                    for k, v in metadata.items():
                        fold = None
                        for i, fold_dirs in enumerate(folds):
                            if part in fold_dirs:
                                fold = i
                                break
                        assert fold is not None
                        video_id = k[:-4]
                        video_fold[video_id] = fold
    return video_fold


def get_video_with_fold2(root_dir):
    np.random.seed(47)
    video_folds = {}
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            video_id = k[:-4]
            original = v['original']
            if original is not None:
                # fake
                original_id = original[:-4]
                if original_id in video_folds:
                    video_folds[video_id] = video_folds[original_id]
                else:
                    random_fold = np.random.randint(0, 5)
                    video_folds[video_id] = random_fold
                    video_folds[original[:-4]] = random_fold
            else:
                # real
                if video_id not in video_folds:
                    random_fold = np.random.randint(0, 5)
                    video_folds[video_id] = random_fold

    return video_folds


def save_folds(args, ori_fakes, video_fold):
    data = []
    ori_ori = set([(ori, ori) for ori, fake in ori_fakes])
    with Pool(processes=1) as p:
        # original label=0
        with tqdm(total=len(ori_ori)) as pbar:
            func = partial(get_paths, label=0, root_dir=args.root_dir)
            for v in p.imap_unordered(func, ori_ori):
                pbar.update()
                data.extend(v)
        # fake label=1
        with tqdm(total=len(ori_fakes)) as pbar:
            func = partial(get_paths, label=1, root_dir=args.root_dir)
            for v in p.imap_unordered(func, ori_fakes):
                pbar.update()
                data.extend(v)
    fold_data = []
    for img_path, label, ori_vid in data:
        path = Path(img_path)
        video = path.parent.name
        file = path.name
        assert video_fold[video] == video_fold[ori_vid], \
            "original video and fake have leak  {} {}".format(ori_vid, video)
        fold_data.append([video, file, label, ori_vid, int(file.split("_")[0]), video_fold[video]])
    # random.shuffle(fold_data)
    columns = ["video", "file", "label", "original", "frame", "fold"]
    pd.DataFrame(fold_data, columns=columns).to_csv(args.out, index=False)


def main():
    args = parse_args()
    ori_fakes = get_original_with_fakes(args.root_dir)
    # folds = split_into_folds(args)
    # print(folds)
    # video_fold = get_video_with_fold(args, folds)
    # video_fold = get_video_with_fold2(args.root_dir)

    # for fold in range(len(folds)):
    #     holdoutset = {k for k, v in video_fold.items() if v == fold}
    #     trainset = {k for k, v in video_fold.items() if v != fold}
    #     assert holdoutset.isdisjoint(trainset), "Folds have leaks"

    save_folds(args, ori_fakes, video_fold)


if __name__ == '__main__':
    main()
