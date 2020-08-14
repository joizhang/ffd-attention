## ffd-attention

More experiments about fake face detection by attention mechanism

## Preprocessing

```shell script
echo "Extracting bounding boxes from original videos"
PYTHONPATH=. python preprocessing/detect_original_faces.py --root-dir $DATA_ROOT

echo "Extracting crops as pngs"
PYTHONPATH=. python preprocessing/extract_crops.py --root-dir $DATA_ROOT --crops-dir crops

echo "Extracting landmarks"
PYTHONPATH=. python preprocessing/generate_landmarks.py --root-dir $DATA_ROOT

echo "Extracting SSIM masks"
PYTHONPATH=. python preprocessing/generate_diffs.py --root-dir $DATA_ROOT

echo "Generate folds"
PYTHONPATH=. python preprocessing/generate_folds.py --root-dir $DATA_ROOT --out folds.csv
```

## Reference

![The proposed network framework with attention mechanism](https://github.com/joizhang/ffd-attention/blob/master/images/readme_fig.png)

Hao Dang, Feng Liu, Joel Stehouwer, Xiaoming Liu, Anil K. Jain; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 5781-5790 [\[PDF\]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Dang_On_the_Detection_of_Digital_Face_Manipulation_CVPR_2020_paper.pdf)

[https://github.com/JStehouwer/FFD_CVPR2020](https://github.com/JStehouwer/FFD_CVPR2020)

[https://github.com/selimsef/dfdc_deepfake_challenge](https://github.com/selimsef/dfdc_deepfake_challenge)
