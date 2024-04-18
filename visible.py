import cv2
import os
import numpy as np
from tqdm import tqdm
# path='frames/'
path='./data/crop_test_frame/images/'
visible_path='./data/crop_test_frame/final_results_025/'
if not os.path.exists(visible_path):
    os.mkdir(visible_path)
# txt='gt.txt'
txt = './2m_nms_V1.txt'
with open(txt) as file:
    lis=file.readlines()
box_frame=[]
f=1
colors=[
    [255, 128, 0],
    [255, 153, 51],
    [255, 178, 102],
    [230, 230, 0],
    [255, 153, 255],
    [153, 204, 255],
    [255, 102, 255],
    [255, 51, 255],
    [102, 178, 255],
    [51, 153, 255],
    [255, 153, 153],
    [255, 102, 102],
    [255, 51, 51],
    [153, 255, 153],
    [102, 255, 102],
    [51, 255, 51]
]
font = cv2.FONT_HERSHEY_SIMPLEX
video_list=[]
video_info=[]
frame_info=[]
# video_list=[video_info,...]
# video_info=[frame_info,...]
# frame_info=[[box],...]
# box=[x,y,h,w,category]
for i in tqdm(lis):
    t=i.split(',')
    video,frame,x,y,h,w,category,score=t
    video,frame,xl,yl,xr,yr,category, score=int(video),int(frame),int(x),int(y),int(x)+int(h),int(y)+int(w),int(category),float(score[:-1])
    box=[int(xl),int(yl),int(xr),int(yr),category, score]
    if frame==len(video_info)+1:
        frame_info.append(box)
        continue
    elif video==len(video_list)+1:
        video_info.append(frame_info)
        # print('new_frame',frame_info)
        frame_info=[]
        frame_info.append(box)
    else:
        video_info.append(frame_info)
        video_list.append(video_info)
        # print('new_video')
        video_info=[]
        frame_info=[]
        frame_info.append(box)
video_info.append(frame_info)
video_list.append(video_info)
for v_idx in tqdm(range(len(video_list))):
    print('video idx',v_idx)
    video=video_list[v_idx]
    #print('video',video)
    for f_idx in range(len(video)):
        frame=video[f_idx]
        #print('frame',f_idx,frame)
        img_path=path+str(str(int(v_idx+1)))+'_'+str(str(int(f_idx+1)))+'.png'
        #print('origin_image',img_path)
        if not os.path.exists(img_path):
            continue
        visible_img=visible_path+str('%03d'%int(v_idx+1))+'/'+str('%06d'%int(f_idx+1))+'.jpg'
        #print('visible_img',visible_img)
        if not os.path.exists(visible_path+str('%03d'%int(v_idx+1))+'/'):
            os.mkdir(visible_path+str('%03d'%int(v_idx+1))+'/')
        image=cv2.imread(img_path)
        for bbox in frame:
            if bbox[5]<0.25:
                continue
            print('visible_img',visible_img)
            label_size1 = cv2.getTextSize(str(bbox[4]), font, 1, 2)
            # print(bbox[1],label_size1[0][1])
            text_origin1 = np.array([bbox[0], bbox[1] - label_size1[0][1]])
            cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color=colors[bbox[4]])
            cv2.rectangle(image,tuple(text_origin1),tuple(text_origin1+label_size1[0]),color=colors[bbox[4]])
            cv2.putText(image,str(bbox[4]),(bbox[0],bbox[1]-5),font,1,(255, 255, 255), 2)
        # print(visible_img)
        #print(visible_img,img_path)
        cv2.imwrite(visible_img,image)

exit()