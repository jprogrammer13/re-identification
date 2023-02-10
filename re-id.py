
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
import argparse
import progressbar


# COCO128 classes https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml
model = YOLO("yolov8m-seg.pt")
# model = YOLO("yolov8n-seg.pt")
siamese_net = SiameseReId(os.path.join('model', 'weights', 'model_final.pt'))


def crop_segmentation(segment, box, image):
    img = image.copy()
    box = box.astype(int)
    w, h, c = img.shape
    mask = (cv2.resize(segment, (h, w)) > 0).astype("uint8")
    img_segm = cv2.bitwise_and(img, img, mask=mask)
    img_segm = img_segm[box[1]:box[3], box[0]:box[2]]
    return cv2.cvtColor(img_segm, cv2.COLOR_BGR2RGB)

# function to get tid given the detection image [return -1 if no similarity find]


def get_tid(track_id_df, det_img):
    similarity = track_id_df['fv'].apply(lambda x: siamese_net.similarity(
        det_img, torch.tensor(x)).detach().cpu().numpy()[0][0])
    if similarity.empty:
        return -1
    else:
        return similarity.idxmax() if similarity.loc[similarity.idxmax()] > 0.8 else -1


def exec_re_id(_video_in, _video_out, _frames_window=10, total_frame=-1):

    file_path = _video_in
    cap = cv2.VideoCapture(file_path)
    video_out = cv2.VideoWriter(_video_out, cv2.VideoWriter_fourcc(*'DIVX'), int(cap.get(
        cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    buffer_idx = 0
    # set initial circular buffer index to 0

    # initialized track_id array
    track_id_df = pd.DataFrame(columns=['fv', 'color'])

    # get video resolution
    vid_w, vid_h, vid_length = int(cap.get(3)), int(
        cap.get(4)), int(cap.get(7))

    if total_frame == -1:
        total_frame = vid_length

    # Generate progressbar
    widgets = [' [',
               progressbar.Timer(format='elapsed time: %(elapsed)s'),
               '] ',
               progressbar.Bar('*'), ' (',
               progressbar.ETA(), ') ',
               ]

    bar = progressbar.ProgressBar(
        max_value=total_frame, widgets=widgets).start()

    frames_window = _frames_window
    # frames_window = np.array([np.gcd(i,total_frame) for i in range(5,10+1)]).max()

    # generate a circular buffer for video frame(frame_window)
    buffer_frames = np.zeros((frames_window, vid_h, vid_w, 3), dtype="uint8")

    # detections array associated to frames
    buffer_detections = [None]*frames_window

    frame_n = 0
    while frame_n < total_frame:

        # populate circular buffer with frame
        load_frame_idx = deepcopy(buffer_idx)

        while load_frame_idx < frames_window:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                video_out.release()
                quit()  # kill the program

            buffer_frames[load_frame_idx] = frame.copy()
            load_frame_idx += 1

        # make prediction for circular frame buffer
        for frame_id in range(buffer_idx, frames_window):
            # deep copy of frame image (yolo draw on the predicted image)
            frame_copy = deepcopy(buffer_frames[frame_id])

            # get results and convert to numpy
            results = model.predict(frame_copy, verbose=False)[0].cpu().numpy()

            if len(results) > 0: #if someone detected 

                # filter prediction of class person with conf > 0.5
                idx = np.where((results.boxes.cls == 0) & (
                    results.boxes.conf > 0.6))  # filter person

                #TODO: implement filter of box ratio (no person)
                # print(results.boxes.xywh)

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
                }).copy()
            else:
                # populate detection array with empty detections
                buffer_detections[frame_id] = pd.DataFrame({
                    'box': [],
                    'mask': [],
                    'center': [],
                    'box_id': [],
                    'track_id': []
                }).copy()

            # delete yolo drawed frame
            del frame_copy, results

        # count the max detections number in per frame
        cluster_numbers = max([len(j) for j in buffer_detections])
        # cluster_numbers

        # create a list containing all the detections per frame
        centers = []

        for j in buffer_detections:
            for val in j['center'].values:
                centers.append(val)
        
        if len(centers) > 0: # if someone detected 

            # fit kmeans cluster with n_cluster equal to max detections per frame
            kmeans = KMeans(n_clusters=cluster_numbers,
                            random_state=0, n_init="auto").fit(centers)

            # for each detections find the box number
            for det in buffer_detections:
                det['box_id'] = det['center'].apply(
                    lambda x: kmeans.predict(np.array([x]).tolist())[0])

            del cluster_numbers, centers, kmeans

            # for each detection calculate feature vector and the correspondent track_id [-1 if no similarity found]
            for i, det in enumerate(buffer_detections):
                # in case of propagated -1 recalculate the track_id else keep the propagated
                if len(det) > 0:  # if someone detected 
                    det['track_id'] = det.apply(lambda x: get_tid(track_id_df, Image.fromarray(crop_segmentation(np.array(
                        x['mask']), np.array(x['box']), buffer_frames[i]))) if int(x['track_id']) == -1 else x['track_id'], axis=1)
                    if (i >= buffer_idx):  # calculate fv only for the missing one
                        det['fv'] = det.apply(lambda x: siamese_net.fv_encoding(Image.fromarray(crop_segmentation(
                            np.array(x['mask']), np.array(x['box']), buffer_frames[i]))).cpu().numpy(), axis=1)

            # for each detection id get the four tid prediction
            counter = {}

            for i, det in enumerate(buffer_detections):
                if len(det) > 0:  # if someone detected 
                    for index, row in det.iterrows():
                        tmp_box_id = row['box_id']
                        if str(tmp_box_id) not in counter:
                            counter[str(tmp_box_id)] = [row['track_id']]
                        else:
                            counter[str(tmp_box_id)].append(row['track_id'])

            # for each detection count the occurence of tid associeted, if < 3 return None (not valid)
            for key in counter:
                # counter[key] = max(counter[key],key=counter[key].count)
                counting_inst = {str(u): counter[key].count(
                    u) for u in np.unique(np.array(counter[key]))}
                candidate_tid = max(counting_inst, key=counting_inst.get)
                tid = candidate_tid if counting_inst[candidate_tid] >= (
                    frames_window//2) else None
                counter[key] = tid

            used = []
            # implement filter on duplicate tid
            for id in counter:
                if counter[id] is not None:  # if None don't evaluate
                    if int(counter[id]) != -1:  # ignore -1 (to add)
                        if counter[id] not in used:
                            used.append(counter[id])
                        else:
                            counter[id] = None

            # for each detection, set the processed tid
            for det in buffer_detections:
                det['track_id'] = det['box_id'].apply(lambda x: counter[str(x)])

            del counter
            del used

            added = {}
            for det in buffer_detections:
                if len(det) > 0:  # if someone detected 
                    for index, row in det.iterrows():
                        if row['track_id'] is not None:
                            if int(row['track_id']) == -1:
                                if row['box_id'] not in added:
                                    # add fv to track_id and return last index
                                    color = np.random.randint(0, 255, 3).tolist()
                                    track_id_df = pd.concat([track_id_df, pd.DataFrame(
                                        {"fv": [row['fv']], 'color':[color]})], ignore_index=True)
                                    new_tid = track_id_df.index[-1]
                                    added[row['box_id']] = new_tid
                                    det._set_value(index, 'track_id', new_tid)
                                    row['track_id'] = new_tid
                                    # print(f"aggiunto {row['box_id']}")
                                else:
                                    det._set_value(index, 'track_id',
                                                added[row['box_id']])
                                    row['track_id'] = added[row['box_id']]

                            # track_id_df.loc[int(row['track_id'])]['fv'] = row['fv']

            del added

            for f in range(frames_window//2):
                drew_frame = buffer_frames[f].copy()
                for indx, row in buffer_detections[f].iterrows():
                    if row['track_id'] is not None:
                        color = track_id_df.loc[int(row['track_id'])]['color']
                        drew_frame = cv2.rectangle(drew_frame, (int(row['box'][0]), int(
                            row['box'][1])), (int(row['box'][2]), int(row['box'][3])), color, 3)
                        drew_frame = cv2.putText(drew_frame, str(row['track_id']), (int(row['box'][0]), int(
                            row['box'][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2, cv2.LINE_AA)
                    else:
                        color = (255, 255, 255)
                        drew_frame = cv2.rectangle(drew_frame, (int(row['box'][0]), int(
                            row['box'][1])), (int(row['box'][2]), int(row['box'][3])), color, 1)

                # cv2.imwrite(f"./prova/{time.time()}.jpg", drew_frame)
                video_out.write(drew_frame.copy())
                del drew_frame

        else:
            for f in range(frames_window//2):
                drew_frame = buffer_frames[f].copy()
                video_out.write(drew_frame.copy())
                del drew_frame


        # clean detection processing
        for det in buffer_detections:
            det["box_id"] = det["box_id"].apply(lambda x: -1)
            det["track_id"] = det["track_id"].apply(
                lambda x: x if x is not None else -1)

        # swap and time propagate with ciruclar buffer
        for f_swap in range(frames_window//2):
            buffer_frames[f_swap] = buffer_frames[f_swap +
                                                  (frames_window//2)].copy()
            buffer_detections[f_swap] = buffer_detections[f_swap +
                                                          (frames_window//2)].copy()

        buffer_idx = (frames_window//2)

        frame_n += (frames_window//2)

        # update progreessbar
        bar.update(frame_n)

    cap.release()
    video_out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Person RE-ID, produce a video with labaeled persons')
    parser.add_argument('--video_in',  help='path of the input video')
    parser.add_argument('--video_out', help='path of the final video')
    parser.add_argument('--frames_window', default=10,
                        help='path of the final video')
    parser.add_argument('--n_frames',  default=-1,
                        help='number of frames to process')

    args = parser.parse_args()

    exec_re_id(args.video_in, args.video_out, int(
        args.frames_window), int(args.n_frames))
