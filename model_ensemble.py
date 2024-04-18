# import os
# import glob
# import json
# import numpy as np
# import cv2
import os

import pandas as pd
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 20)
from tqdm import tqdm
from ensemble import ensemble_boxes_nms


if __name__ == "__main__":

######################################################################################################################
    pred_path_1 = "./submit/codino_swinl_V1_0318_1_epoch_8.txt"
    pred_path_2 = "./submit/deta_final_noCRM.txt"

    save_name = '2m_plus_16_nms_V2'
    save_txt_path = f'./submit/{save_name}.txt'
    os.makedirs(os.path.basename(save_txt_path), exist_ok=True)
    print(save_txt_path)
######################################################################################################################

    df1 = pd.read_csv(pred_path_1, header=None)
    df1.columns=['video_id','frame_id','x1','y1','width','height','category_id','score']
    df2 = pd.read_csv(pred_path_2, header=None)
    df2.columns = ['video_id', 'frame_id', 'x1', 'y1', 'width', 'height', 'category_id', 'score']

    """xywh to xyxy"""
    df1['x2'] = df1['x1'] + df1['width']
    df1['y2'] = df1['y1'] + df1['height']
    df2['x2'] = df2['x1'] + df2['width']
    df2['y2'] = df2['y1'] + df2['height']

    df_final = []

    for video_id in tqdm(range(1,101)):
        df1_video = df1[df1['video_id']==video_id]
        df2_video = df2[df2['video_id']==video_id]
        min_frame_id = min(df1_video['frame_id'].min(),
                           df2_video['frame_id'].min(),
                           )
        max_frame_id = max(df1_video['frame_id'].max(),
                           df2_video['frame_id'].max(),
                           )
        # print(video_id,min_frame_id,max_frame_id)
        for frame_id in range(min_frame_id, max_frame_id+1):
            df1_frame = df1_video[df1_video['frame_id'] == frame_id]
            df2_frame = df2_video[df2_video['frame_id'] == frame_id]
            preds_1 = df1_frame.loc[:,['x1','y1','x2','y2', 'category_id', 'score']].values
            preds_2 = df2_frame.loc[:,['x1','y1','x2','y2', 'category_id', 'score']].values

            bboxes_before = [preds_1[:,0:4].tolist() if len(preds_1) else [],
                             preds_2[:,0:4].tolist() if len(preds_2) else [],
                             ]
            labels_before = [preds_1[:,4].astype(int).tolist() if len(preds_1) else [],
                             preds_2[:,4].astype(int).tolist() if len(preds_2) else [],
                             ]
            scores_before = [preds_1[:,5].tolist() if len(preds_1) else [],
                             preds_2[:,5].tolist() if len(preds_2) else [],
                             ]
            bboxes_fuse, scores_fuse, labels_fuse = ensemble_boxes_nms.nms(bboxes_before, scores_before, labels_before, thresh=0.01, iou_thr=0.7)

            for bbox_fuse, score_fuse, label_fuse in zip(bboxes_fuse, scores_fuse, labels_fuse):
                x1 = bbox_fuse[0]
                y1 = bbox_fuse[1]
                width = bbox_fuse[2] - bbox_fuse[0]
                height = bbox_fuse[3] - bbox_fuse[1]
                category_id = int(label_fuse)
                df_final.append({'video_id': int(video_id),
                           'frame_id': int(frame_id),
                           'x1': int(round(x1)),
                           'y1': int(round(y1)),
                           'width': int(round(width)),
                           'height': int(round(height)),
                           'category_id': int(category_id),
                           'score': float(score_fuse),
                           })
    df_final= pd.DataFrame(df_final)
    print(df_final.shape)
    print(df_final.head())
    df_final.to_csv(save_txt_path, header=None, index=None, sep=',')
    print(save_txt_path)

