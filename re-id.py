
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os
import time
import torch
from copy import deepcopy
from model.SiameseReId import SiameseReId
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import time


model = YOLO("yolov8n-seg.pt") # COCO128 classes https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml
# model = YOLO("yolov8n-seg.pt")
siamese_net = SiameseReId(os.path.join('model','weights','model_final.pt'))


def crop_segmentation(segment,box,image):
    img = image.copy()
    box = box.astype(int)
    w,h,c = img.shape
    mask = (cv2.resize(segment,(h,w)) > 0).astype("uint8")
    img_segm = cv2.bitwise_and(img,img,mask=mask)
    img_segm = img_segm[box[1]:box[3],box[0]:box[2]]
    return cv2.cvtColor(img_segm, cv2.COLOR_BGR2RGB)

file_path = 'friends.mp4'
cap = cv2.VideoCapture(file_path)
video_out = cv2.VideoWriter("./out_firends.mp4", cv2.VideoWriter_fourcc(*'DIVX'), int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))


i = 0
# set initial circular buffer index to 0


# initialized track_id array
track_id_df = pd.DataFrame(columns=['fv','color'])

# get video resolution
vid_w, vid_h = int(cap.get(3)), int(cap.get(4))
vid_w, vid_h


frames_window = 10
# generate a circular buffer for video frame(4)
buffer_frames = np.zeros((frames_window,vid_h, vid_w,3),dtype="uint8")


frame_n = 0
while frame_n < 500:
        
    # populate circular buffer with frame
    while i < frames_window:
        ret, frame = cap.read()
        buffer_frames[i] = frame.copy()
        i+=1
        if not ret:
            cap.release()
            video_out.release()
            quit() #kill the program


    # four_detectiona[0] = four_detectiona[2]
    # four_detectiona[1] = four_detectiona[3]
    # i = 2

    
    # detections array associated to frames
    buffer_detections = [None]*frames_window


    
    # make prediction for each frame in the buffer
    for frame_id in range(frames_window):
        # deep copy of frame image (yolo draw on the predicted image)
        frame_copy = deepcopy(buffer_frames[frame_id])

        # get results and convert to numpy
        results = model.predict(frame_copy)[0].cpu().numpy()

        # filter prediction of class person with conf > 0.5
        idx = np.where((results.boxes.cls == 0) & (
            results.boxes.conf > 0.6))  # filter person

        # extract masks,boxes and bb centers
        masks = results.masks.masks[idx].copy()
        boxes = results.boxes.xyxy[idx].copy()
        centers = results.boxes.xywh[idx][:, :2].copy().astype(int)

        # populate detection array with associeted detections
        buffer_detections[frame_id] = pd.DataFrame({
                                                'box': boxes.tolist(),
                                                'mask': masks.tolist(),
                                                'center': centers.tolist(),
                                                'box_id': np.full(len(idx[0]), -1).tolist(),
                                                'track_id': np.full(len(idx[0]), -1).tolist()
                                                })

        # delete yolo drawed frame
        del frame_copy, results


    
    # count the max detections number in per frame
    cluster_numbers = max([len(j) for j in buffer_detections])
    cluster_numbers

    
    # create a list containing all the detections per frame
    centers = []

    for j in buffer_detections:
        for val in j['center'].values:
            centers.append(val)

    
    # fit kmeans cluster with n_cluster equal to max detections per frame
    kmeans = KMeans(n_clusters=cluster_numbers, random_state=0, n_init="auto").fit(centers)

    
    # for each detections find the box number
    for det in buffer_detections:
        det['box_id'] = det['center'].apply(lambda x: kmeans.predict(np.array([x]).tolist())[0])

    del cluster_numbers, centers, kmeans

    
    # function to get tid given the detection image [return -1 if no similarity find]
    def get_tid(track_id_df,det_img):
        similarity = track_id_df['fv'].apply(lambda x: siamese_net.similarity(det_img,torch.tensor(x)).detach().cpu().numpy()[0][0])
        if similarity.empty:
            return -1
        else:
            return similarity.idxmax() if similarity.loc[similarity.idxmax()] > 0.8 else -1

    
    # for each detection calculate feature vector and the correspondent track_id [-1 if no similarity found]
    for i,det in enumerate(buffer_detections):
        det['fv'] = det.apply(lambda x: siamese_net.fv_encoding(Image.fromarray(crop_segmentation(np.array(x['mask']),np.array(x['box']),buffer_frames[i]))).cpu().numpy(),axis=1)
        det['track_id'] = det.apply(lambda x: get_tid(track_id_df,Image.fromarray(crop_segmentation(np.array(x['mask']),np.array(x['box']),buffer_frames[i]))),axis=1)

    
    # for each detection id get the four tid prediction
    counter = {}

    for i,det in enumerate(buffer_detections):
        for index, row in det.iterrows():
            tmp_box_id = row['box_id']
            if str(tmp_box_id) not in counter:
                counter[str(tmp_box_id)] = [row['track_id']]
            else:
                counter[str(tmp_box_id)].append(row['track_id'])


    
    # for each detection count the occurence of tid associeted, if < 3 return None (not valid)
    for key in counter:
        # counter[key] = max(counter[key],key=counter[key].count)
        counting_inst = {str(u):counter[key].count(u) for u in np.unique(np.array(counter[key]))}
        candidate_tid = max(counting_inst,key=counting_inst.get)
        tid = candidate_tid if counting_inst[candidate_tid] >= frames_window//2 else None
        counter[key] = tid


    
    # for each detection, set the processed tid
    for det in buffer_detections:
        det['track_id'] = det['box_id'].apply(lambda x: counter[str(x)])

    
    del counter

    
    added = {}
    for det in buffer_detections:
        for index, row in det.iterrows():
            if row['track_id'] is not None:
                if int(row['track_id']) == -1:
                    if row['box_id'] not in added:
                        # add fv to track_id and return last index
                        color = np.random.randint(0,255,3).tolist()
                        track_id_df = pd.concat([track_id_df, pd.DataFrame({"fv": [row['fv']], 'color':[color]})], ignore_index=True)
                        new_tid = track_id_df.index[-1]
                        added[row['box_id']] = new_tid
                        det._set_value(index,'track_id',new_tid)
                        row['track_id'] = new_tid
                        print(f"aggiunto {row['box_id']}")
                    else:
                        det._set_value(index,'track_id',added[row['box_id']])
                        row['track_id'] = added[row['box_id']]

                # track_id_df.loc[int(row['track_id'])]['fv'] = row['fv']

    
    del added

    
    for f in range(frames_window//2):
        drew_frame = buffer_frames[f].copy()
        for indx, row in buffer_detections[f].iterrows():
            if row['track_id'] is not None:
                color = track_id_df.loc[int(row['track_id'])]['color']
                drew_frame = cv2.rectangle(drew_frame, (int(row['box'][0]), int(row['box'][1])), (int(row['box'][2]), int(row['box'][3])), color, 3)
                drew_frame = cv2.putText(drew_frame, str(row['track_id']), (int(row['box'][0]),int(row['box'][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2, cv2.LINE_AA)

        # cv2.imwrite(f"./prova/{time.time()}.jpg", drew_frame)
        video_out.write(drew_frame.copy())
        del drew_frame
            

    
    del buffer_detections

    # TODO: fix this
    buffer_frames[0] = buffer_frames[5]
    buffer_frames[1] = buffer_frames[6]
    buffer_frames[2] = buffer_frames[7]
    buffer_frames[3] = buffer_frames[8]
    buffer_frames[4] = buffer_frames[9]

    i = frames_window//2

    frame_n += frames_window//2

    

cap.release()
video_out.release()

