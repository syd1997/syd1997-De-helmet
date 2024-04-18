import os
import json
txt='gt.txt'
src='./frames/'
label=[]
with open(txt) as file:
    lis=file.readlines()
total_list=[]
frame=[]
i=0
for video in sorted(os.listdir(src)):
    v_path=os.path.join(src,video)
    for frame in sorted(os.listdir(v_path)):
        f_path=os.path.join(v_path,frame)
        frames=[f_path]
        while video==str('%03d'%int(lis[i].split(',')[0])) and frame.split('.')[0] == str('%06d'%int(lis[i].split(',')[1])):
            old=lis[i].split(',')
            
            old[-1]=old[-1][0]
            box=','.join(old[2:7])
            frames.append(box)
            i+=1
        total_list.append(frames)
label='train_set_all.txt'
with open(label,'a+') as file:
    for line in total_list:
        file.write(' '.join(line)+'\n')