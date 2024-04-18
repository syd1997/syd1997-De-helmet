import os
import json
import pandas as pd
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 20)
import glob
from tqdm import tqdm

# Replace with your path here.
deta_dir = "$DEAT_RESULTS.TXT"
save_txt_path = "$SAVE_DIR.TXT"

if not os.path.exists(os.path.dirname(save_txt_path)):
    os.makedirs(os.path.dirname(save_txt_path))

def encode(st):
    a = int(st.split('/')[-1].split('_')[0])
    b = int(st.split('/')[-1].split('_')[1][:-4])
    return a*1e5+b

deta_paths = glob.glob(os.path.join(deta_dir, "*.txt"))
print(len(deta_paths))
deta_paths=sorted(deta_paths, key=lambda x: encode(x))
# res = []
df = []

for i in tqdm(range(len(deta_paths))):
    data = pd.read_csv(deta_paths[i], sep=" ", header=None)
    txt_name = os.path.basename(deta_paths[i])  # '100_1.txt'
    img_name = txt_name.split('.txt')[0]
    video_id = img_name.split('_')[0]
    frame_id = img_name.split('_')[1]
    labels = data.iloc[:,0].values
    scores = data.iloc[:,5].values
    x1s = data.iloc[:,1].values
    y1s = data.iloc[:,2].values
    x2s = data.iloc[:,3].values
    y2s = data.iloc[:,4].values
    widths = x2s - x1s
    heights = y2s - y1s
    for k in range(len(labels)):
        category_id = int(labels[k]) + 1  # from 0~8 to 1~9
        if category_id in [6,8]:
            continue
        if not category_id in [1,2,3,4,5,6,7,8,9]:
            continue
        score = scores[k]
        x1 = x1s[k]
        y1 = y1s[k]
        width = widths[k]
        height = heights[k]
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