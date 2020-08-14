import os
import shutil

if __name__ == '__main__':
    # TODO: This file will be deleted
    root_dir = 'F:\\Celeb-DF-v2'
    sample_path = os.path.join(root_dir, 'samples')
    os.makedirs(sample_path, exist_ok=True)
    crops_path = os.path.join(root_dir, 'crops')
    for crop_video in os.listdir(crops_path):
        crop_video_path = os.path.join(crops_path, crop_video)
        video_crop_list = os.listdir(crop_video_path)
        if len(video_crop_list) < 2:
            continue
        for i in range(2):
            src = os.path.join(crop_video_path, video_crop_list[i])
            dst = os.path.join(sample_path, '{}_{}'.format(crop_video, video_crop_list[i]))
            shutil.copyfile(src, dst)
