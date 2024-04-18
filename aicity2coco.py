import os
import cv2
import json
from tqdm import tqdm
import argparse

ground_truth='train_set_new.txt'

def yolo2coco():
    root_path = ground_truth
    with open('labels.txt') as f:
        classes = f.read().strip().split()
    # images dir name
    with open(ground_truth) as f:
        indexes = f.readlines()
    
    dataset = {'categories': [], 'annotations': [], 'images': []}
    for i, cls in enumerate(classes, 0):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
    # 标注的id
    ann_id_cnt = 0
    for k, index in enumerate(tqdm(indexes)):
        # 支持 png jpg 格式的图片。
        # index=(path box box ...) 
        # box=(x,y,h,w,category)
        idd=index.split()
        path=idd[0]
        boxes=idd[1:]
        # 读取图像的宽和高
        im = cv2.imread(path)
        if im is None:
            continue
            
        height, width, _ = im.shape
        # 添加图像的信息
        dataset['images'].append({'file_name': path,
                                    'id': k,
                                    'width': width,
                                    'height': height})
        # if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
            # 如没标签，跳过，只保留图片信息。
            # continue
        txtfile=path.replace('frames','labels').replace('jpg','json')
        for box in boxes:
            box = box.strip().split(',')
            x = float(box[0])
            y = float(box[1])
            w = float(box[2])
            h = float(box[3])
            category = int(box[4])
            # convert x,y,w,h to x1,y1,x2,y2
            H, W, _ = im.shape
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            dataset['annotations'].append({
                'area': width * height,
                'bbox': [x1, y1, width, height],
                'category_id': category,
                'id': ann_id_cnt,
                'image_id': k,
                'iscrowd': 0,
                # mask, 矩形是从左上角点按顺时针的四个顶点
                'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
            })
            ann_id_cnt += 1
        if not os.path.exists(txtfile[:46]):
            os.mkdir(txtfile[:46])

    # 保存结果
    json_name = 'train_set_all.json'
    with open(json_name, 'w') as f:
        json.dump(dataset, f)
        print('Save annotation to {}'.format(json_name))

if __name__ == "__main__":
    yolo2coco()