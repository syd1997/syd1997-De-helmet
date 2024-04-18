import os
import glob
import json
from tqdm import tqdm
import cv2
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.core import DatasetEnum
import mmcv

SAVE_VIS = False

config_file = './Co-DETR/projects/configs/codino_swinl_V1_trainall.py'
checkpoint_file = './Co-DETR/work_dir/competition_helmet/codino_swinl_V1_0318_1/epoch_8.pth'
out_dir = './data/ai_city_challenge_2024/track_5/output/codino_swinl_V1_0318_1_epoch_8_res'
img_dir = './data/ai_city_challenge_2024/track_5/aicity2024_track5_test/images_V2/'
image_paths = glob.glob(os.path.join(img_dir, "*.png"))

os.makedirs(out_dir, exist_ok=True)
out_pred_dir = os.path.join(out_dir, 'preds')
os.makedirs(out_pred_dir, exist_ok=True)
if SAVE_VIS:
    out_vis_dir = os.path.join(out_dir, 'vis')
    os.makedirs(out_vis_dir, exist_ok=True)

score_thr = 0.1
model = init_detector(config_file, checkpoint_file, DatasetEnum.COCO, device='cuda:2')

hflip = False
resize = False
scale_factor = 0.8
resize_scale = (int(1920*scale_factor), int(1080*scale_factor))
print('hflip:', hflip)
print('resize:', resize)
if resize:
    print('resize_scale:', resize_scale)

for image_path in tqdm(image_paths):
    image_orig = cv2.imread(image_path)
    orig_h, orig_w = image_orig.shape[0], image_orig.shape[1]

    if hflip:
        image = cv2.flip(image_orig, 1)
    else:
        image = image_orig
    if resize:
        image = cv2.resize(image, resize_scale)

    result = inference_detector(model, image)  # a list, its length equals to num_classes
    data = {'labels':[], 'scores':[], 'bboxes':[]}
    for label in range(len(result)):
        for k in range(len(result[label])):
            if hflip:  # convert result to original image
                result[label][k][0] = orig_w - result[label][k][0]
                result[label][k][2] = orig_w - result[label][k][2]
            if resize:
                result[label][k][0] /= scale_factor
                result[label][k][1] /= scale_factor
                result[label][k][2] /= scale_factor
                result[label][k][3] /= scale_factor
            data['labels'].append(label)
            data['scores'].append(float(result[label][k][4]))
            x1,y1,x2,y2 = float(result[label][k][0]), float(result[label][k][1]), float(result[label][k][2]), float(result[label][k][3])
            data['bboxes'].append([x1,y1,x2,y2])

    out_pred_path = os.path.join(out_pred_dir, os.path.basename(image_path)[:-4]+'.json')
    with open(out_pred_path, 'w') as f:
        json.dump(data, f)

    if SAVE_VIS:
        out_vis_path = os.path.join(out_vis_dir, os.path.basename(image_path))
        show_result_pyplot(model, image_orig, result, score_thr=score_thr, out_file=out_vis_path)