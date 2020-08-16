import argparse
import os
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

LIST_OF_TESTING_VIDEOS = 'List_of_testing_videos.txt'


def parse_args():
    parser = argparse.ArgumentParser(description="Extract image diffs")
    parser.add_argument("--root-dir", help="root directory", default="")
    parser.add_argument("--out", type=str, default="folds.csv", help="CSV file to save")
    parser.add_argument('--test', dest='test', action='store_true', help='Generate test fold.csv')
    args = parser.parse_args()
    return args


def get_test_videos(root_dir) -> list:
    test_videos = []
    for line in open(os.path.join(root_dir, LIST_OF_TESTING_VIDEOS), 'r'):
        line_split = line.split()
        assert len(line_split) == 2
        test_videos.append(line_split[1])
    return test_videos


def get_real_fake_pairs_for_test(root_dir):
    pairs = []
    test_videos = get_test_videos(root_dir)
    for test_video in test_videos:
        video_fold = Path(test_video).parent.name
        video_path = Path(test_video).name
        if 'real' in video_fold:
            pairs.append((video_path[:-4], video_path[:-4]))
        else:
            video_path_split = video_path.split('_')
            assert len(video_path_split) == 3
            pairs.append(('{}_{}'.format(video_path_split[0], video_path_split[2][:-4]), video_path[:-4]))

    return pairs


def get_real_fake_pairs(root_dir):
    pairs = []
    pairs_for_test = set(get_real_fake_pairs_for_test(root_dir))
    for video_fold in os.listdir(root_dir):
        if 'real' not in video_fold and 'synthesis' not in video_fold:
            continue
        for video_path in os.listdir(os.path.join(root_dir, video_fold)):
            if 'real' in video_fold:
                pair = (video_path[:-4], video_path[:-4])
                if pair not in pairs_for_test:
                    pairs.append(pair)
            else:
                video_path_split = video_path.split('_')
                assert len(video_path_split) == 3
                pair = ('{}_{}'.format(video_path_split[0], video_path_split[2][:-4]), video_path[:-4])
                if pair not in pairs_for_test:
                    pairs.append(pair)

    return pairs


def get_paths(vid, label, root_dir):
    ori_vid, fake_vid = vid
    ori_dir = os.path.join(root_dir, "crops", ori_vid)
    fake_dir = os.path.join(root_dir, "crops", fake_vid)
    data = []
    for frame in os.listdir(fake_dir):
        ori_img_path = os.path.join(ori_dir, frame)
        fake_img_path = os.path.join(fake_dir, frame)
        img_path = ori_img_path if label == 0 else fake_img_path
        if os.path.exists(img_path):
            data.append([img_path, label, ori_vid])
    return data


def save_folds(args, ori_fake_pairs):
    data = []
    ori_ori_pairs = set([(ori, ori) for ori, fake in ori_fake_pairs if ori == fake])
    ori_fake_pairs = set(ori_fake_pairs) - ori_ori_pairs
    with Pool(processes=1) as p:
        # original label=0
        with tqdm(total=len(ori_ori_pairs)) as pbar:
            func = partial(get_paths, label=0, root_dir=args.root_dir)
            for v in p.imap_unordered(func, ori_ori_pairs):
                pbar.update()
                data.extend(v)
        # fake label=1
        with tqdm(total=len(ori_fake_pairs)) as pbar:
            func = partial(get_paths, label=1, root_dir=args.root_dir)
            for v in p.imap_unordered(func, ori_fake_pairs):
                pbar.update()
                data.extend(v)
    fold_data = []
    for img_path, label, ori_vid in data:
        path = Path(img_path)
        video = path.parent.name
        file = path.name
        fold_data.append([video, file, label, ori_vid, int(file.split("_")[0])])
    # random.shuffle(fold_data)
    columns = ["video", "file", "label", "original", "frame"]
    pd.DataFrame(fold_data, columns=columns).to_csv(args.out, index=False)


def main():
    args = parse_args()
    if args.test:
        ori_fake_pairs = get_real_fake_pairs_for_test(args.root_dir)
    else:
        ori_fake_pairs_for_test = get_real_fake_pairs_for_test(args.root_dir)
        ori_fake_pairs = get_real_fake_pairs(args.root_dir)
        assert set(ori_fake_pairs).isdisjoint(set(ori_fake_pairs_for_test))

    print(len(ori_fake_pairs))
    # ori_ori_pairs = set([(ori, ori) for ori, fake in ori_fake_pairs])
    # ori_fake_pairs = set(ori_fake_pairs) - ori_ori_pairs
    # print(len(ori_fake_pairs), len(ori_ori_pairs))

    save_folds(args, ori_fake_pairs)


if __name__ == '__main__':
    # Total: 6529
    # Test: 518
    # Train and validation: 6011
    main()
