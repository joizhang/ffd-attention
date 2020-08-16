import os
from pathlib import Path

from preprocessing.generate_folds import parse_args

LIST_OF_TESTING_VIDEOS = 'List_of_testing_videos.txt'


def get_test_videos(root_dir) -> list:
    test_videos = []
    with open(os.path.join(root_dir, LIST_OF_TESTING_VIDEOS), 'rb') as f:
        lines = f.readlines()
    for line in lines:
        line_split = line.split()
        assert len(line_split) == 2
        test_videos.append(str(line_split[1]))
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


def main():
    args = parse_args()
    print(len(get_test_videos(args.root_dir)))
    ori_fake_pairs = get_real_fake_pairs_for_test(args.root_dir)
    print(len(ori_fake_pairs))


if __name__ == '__main__':
    main()
