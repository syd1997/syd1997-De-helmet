import os
import json
import pandas as pd
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 20)
import glob
import pickle
from tqdm import tqdm
import seaborn as sns

#######################################################################################################################
json_dir = "./data/ai_city_challenge_2024/track_5/output/codino_swinl_V1_0318_1_epoch_8_res/preds"
save_txt_path = "./data/ai_city_challenge_2024/track_5/submit/codino_swinl_V1_0318_1_epoch_8.txt"

#######################################################################################################################

def encode(st):
    a = int(st.split('/')[-1].split('_')[0])
    b = int(st.split('/')[-1].split('_')[1][:-5])
    return a*1e5+b

json_paths = glob.glob(os.path.join(json_dir, "*.json"))
print(len(json_paths))
json_paths=sorted(json_paths, key=lambda x: encode(x))
# res = []
df = []

for i in tqdm(range(len(json_paths))):
    with open(json_paths[i], 'r') as f:
        data = json.load(f)
        json_name = os.path.basename(json_paths[i])  # '100_1.json'
        img_name = json_name.split('.json')[0]
        video_id = img_name.split('_')[0]
        frame_id = img_name.split('_')[1]
        labels = data['labels']
        scores = data['scores']
        bboxes = data['bboxes']  # xyxy
    for k in range(len(labels)):
        x1,y1 = bboxes[k][0], bboxes[k][1]
        width = bboxes[k][2] - bboxes[k][0]
        height = bboxes[k][3] - bboxes[k][1]
        category_id = int(labels[k]) + 1  # from 0~8 to 1~9
        score = scores[k]
        # res.append(f"{video_id},{frame_id},{x1:.6f},{y1:.6f},{width:.6f},{height:.6f},{category_id},{score:.6f}\n")
        df.append({'video_id': int(video_id),
                       'frame_id': int(frame_id),
                       'x1': int(round(x1)),
                       'y1': int(round(y1)),
                       'width': int(round(width)),
                       'height': int(round(height)),
                       'category_id': int(category_id),
                       'score': float(score),
                       })
df = pd.DataFrame(df)
print(df.shape)
df.to_csv(save_txt_path, header=None, index=None, sep=',')
print(save_txt_path)


# df = pd.DataFrame(df)
# print(df.shape)
# # ax = sns.displot(df, x="score", kind="kde")
#
# df = df[df['score']>=0.01]
# print(df.shape)
#
# df['ratio'] = df.loc[:,['width','height']].apply(lambda x:x['width']/x['height'] if x['width']/x['height']>1 else x['height']/x['width'], axis=1)
# df['area'] = df['width'] * df['height']
#
# final_res = []
# for i in tqdm(range(len(df))):
#     pred = df.iloc[i]
#     video_id = int(pred['video_id'])
#     frame_id = int(pred['frame_id'])
#     x1 = int(pred['x1'])
#     y1 = int(pred['y1'])
#     width = int(pred['width'])
#     height = int(pred['height'])
#     category_id = int(pred['category_id'])
#     score = float(pred['score'])
#     final_res.append(f"{video_id},{frame_id},{x1:.6f},{y1:.6f},{width:.6f},{height:.6f},{category_id},{score:.6f}\n")



