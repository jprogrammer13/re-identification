import pandas as pd
import numpy as np
from PIL import Image
import cv2
import argparse
import sys
import os
from sklearn.cluster import KMeans
import sklearn
import torch

import progressbar

from video_handler.VideoHandler import VideoHandler
from detection.Detection import Detection
from model.SiameseReId import SiameseReId


class ReID():
    def __init__(self, video_path: str, n_frames: int = -1,  batch_size: int = -1) -> None:
        self.__video_handler: VideoHandler = VideoHandler(video_path)
        self.__detection: Detection = Detection()
        self.__siamese_net: SiameseReId = SiameseReId("./model/weights/model_final_mars.pt")
        self.__features: pd.DataFrame = pd.DataFrame(columns=["fv", "color"])
        self.__vid_w, self.__vid_h, video_frame_count = self.__video_handler.get_video_w_h_fc()

        self.__frame_count: int = video_frame_count

        if n_frames != -1:
            if n_frames <= video_frame_count:
                self.__frame_count = n_frames
            else:
                sys.exit(f"Error: "
                         f"The number of frames ({n_frames}) has to be "
                         f"less or equal than the number of frames "
                         f"of the video (for {video_path}: {video_frame_count})"
                         f"")

        self.__batch_size = 10

        if batch_size != -1:
            self.__batch_size = batch_size
        else:
            bs: int = np.array([np.gcd(i, self.__frame_count) for i in range(10, 20+1)]).max()

            if bs > 5:
                self.__batch_size = bs
                # sys.exit(f"Error generating batch size with frame number = {self.__frame_count}")

        print("Batch size: ", self.__batch_size)
        print("Frames count ", self.__frame_count)

        self.__frames_buffer: list[np.ndarray] = [np.zeros((self.__vid_h, self.__vid_w, 3), dtype="uint8")] * self.__batch_size
        self.__detections_buffer: list[pd.DataFrame] = [None] * self.__batch_size

        self.__counter = {}

        widgets = [" [",
                   progressbar.Timer(format="elapsed time: %(elapsed)s"),
                   "] ",
                   progressbar.Bar('*'), " (",
                   progressbar.ETA(), ") \n",
                   ]
        self.__progress_bar: progressbar.ProgressBar = progressbar.ProgressBar(max_value=self.__frame_count, widgets=widgets).start()

        self.exec()

    def __get_buffer_frames(self, buffer_head: int) -> bool:
        buffer_id = buffer_head
        while buffer_id < self.__batch_size:
            ret, frame = self.__video_handler.video_read_frame()
            if not ret:
                return ret
            self.__frames_buffer[buffer_id] = frame.copy()  # TODO copy si o no?
            buffer_id += 1
        return True

    def __buffers_swap(self, head: int) -> None:
        self.__frames_buffer[:head] = self.__frames_buffer[head:]
        self.__detections_buffer[:head] = self.__detections_buffer[head:]

    def __get_kmeans_model(self, max_bb: int) -> sklearn.cluster._kmeans.KMeans:
        centers = np.array([])
        for det in self.__detections_buffer:
            if not np.any(centers):
                centers = det["center"]
            else:
                centers = pd.concat((centers, det["center"]), axis=0)

        # fit kmeans cluster with n_cluster equal to max detections per frame
        return KMeans(n_clusters=max_bb, random_state=0, n_init="auto").fit(centers.tolist())

    def __crop_segmentation(self, mask: list, box: list, img: np.ndarray) -> Image:
        # img = img.copy()
        w, h, _ = img.shape
        mask = (cv2.resize(np.array(mask), (h, w)) > 0).astype("uint8")
        img_segm = cv2.bitwise_and(img, img, mask=mask)
        img_segm = img_segm[box[1]:box[3], box[0]:box[2]]
        return Image.fromarray(cv2.cvtColor(img_segm, cv2.COLOR_BGR2RGB))

    def __find_box_id(self) -> None:
        # len of bounding boxes dataframe per frame
        max_bb_in_batch = max([len(frame_bb) for frame_bb in self.__detections_buffer])
        if max_bb_in_batch > 0:  # at least one frame with persons
            kmeans = self.__get_kmeans_model(max_bb_in_batch)

            for df in self.__detections_buffer:
                df["box_id"] = df["center"].apply(lambda row: kmeans.predict([np.array(row)]).tolist()[0])

    def __get_track_id_of_detection_img(self, img: Image) -> int:
        similarity = self.__features["fv"].apply(lambda x: self.__siamese_net.similarity(
            img, torch.tensor(x)).detach().cpu().numpy()[0][0])
        if similarity.empty:
            return -1
        else:
            return similarity.idxmax() if similarity.loc[similarity.idxmax()] > 0.8 else -1

    def __calculate_features_vectors(self, buffer_head: int) -> None:
        for i, det in enumerate(self.__detections_buffer):
            if not det.empty:
                det["track_id"] = det.apply(
                    lambda row: self.__get_track_id_of_detection_img(
                        self.__crop_segmentation(row["mask"], row["xyxy"], self.__frames_buffer[i])
                    )
                    if row["track_id"] == -1
                    else row["track_id"],
                    axis=1
                )

                # if i >= buffer_head:
                #     det["fv"] = det.apply(
                #         lambda row: self.__siamese_net.fv_encoding(
                #             self.__crop_segmentation(row["mask"], row["xyxy"], self.__frames_buffer[i])
                #         ).cpu().numpy(),
                #         axis=1
                #     )

    def __update_counter(self, box_id: int, track_id: int):
        self.__counter.setdefault(str(box_id), []).append(track_id)

    def __count_box_ids(self) :
        for i, det in enumerate(self.__detections_buffer):
            if not det.empty:
                det.apply(lambda row: self.__update_counter(row["box_id"], row["track_id"]), axis=1)

        used = set()
        for key in self.__counter:
            counting_inst = {str(u): self.__counter[key].count(u) for u in np.unique(np.array(self.__counter[key]).astype(int))}

            candidate_tid = max(counting_inst, key=counting_inst.get)
            tid = candidate_tid if counting_inst[candidate_tid] >= (self.__batch_size // 2) else np.nan
            self.__counter[key] = tid
            # if np.nan don't evaluate & ignore -1 (to add)
            # if tid is not np.nan and int(tid) != -1:
            if not pd.isna(tid) and int(tid) != -1:
                # print(int(tid) not in used)
                if int(tid) not in used:
                    self.__counter[key] = int(tid)
                    used.add(self.__counter[key])
                else:
                    self.__counter[key] = np.nan

        for det in self.__detections_buffer:
            det["track_id"] = det["box_id"].apply(
                lambda row: int(self.__counter[str(row)]) if not pd.isna(self.__counter[str(row)]) else np.nan
            )

        self.__counter = {}

    def __add_features_vectors(self):
        added = {}

        for i, det in enumerate(self.__detections_buffer):
            for index, row in det.iterrows():  # doesn't loop on empty datasets
                # if row["track_id"] is not np.nan and int(row["track_id"]) == -1:  # not np.nan
                if not pd.isna(row["track_id"]) and int(row["track_id"]) == -1:  # not np.nan
                    if row["box_id"] not in added:
                        color = np.random.randint(0, 255, 3).tolist()

                        fv = self.__siamese_net.fv_encoding(
                                self.__crop_segmentation(row["mask"], row["xyxy"], self.__frames_buffer[i])
                            ).cpu().numpy()

                        self.__features = pd.concat(
                            [self.__features, pd.DataFrame({"fv": [fv], "color": [color]})],
                            # [self.__features, pd.DataFrame({"fv": [row["fv"]], "color": [color]})],
                            ignore_index=True
                        )
                        new_tid = self.__features.index[-1]
                        added[row["box_id"]] = new_tid
                        det._set_value(index, "track_id", new_tid)
                        row["track_id"] = new_tid
                    else:
                        det._set_value(index, "track_id", added[row["box_id"]])
                        row["track_id"] = added[row["box_id"]]

    def __draw_video_frames(self, max: int) -> None:
        for i in range(max):
            det = self.__detections_buffer[i]
            drew_frame = self.__frames_buffer[i]
            if not det.empty:
                for _, row in det.iterrows():
                    color = (255, 255, 255)
                    text = "Unkown"

                    # if row["track_id"] is not np.nan and int(row["track_id"]) != -1:
                    if not pd.isna(row["track_id"]) and int(row["track_id"]) != -1:
                        color = self.__features.loc[int(row["track_id"])]["color"]
                        text = str(int(row["track_id"]))

                    drew_frame = self.__video_handler.frame_draw_info(drew_frame, row["xyxy"], color, text)

            self.__video_handler.video_write(drew_frame)

    def __clean_detections_box_ids_track_ids(self):
        for det in self.__detections_buffer:
            det["box_id"] = [-1] * len(det["box_id"])
            # det["track_id"] = det["track_id"].apply(lambda x: x if x is not np.nan else -1)
            # det.loc[np.where(det["track_id"] == np.nan)[0], "track_id"] = -1
            det.loc[np.where(pd.isna(det["track_id"]))[0], "track_id"] = -1

    def exec(self) -> None:
        buffer_head = 0
        frame_nr = 0

        while frame_nr < self.__frame_count:
            # initial frames buffer
            ret = self.__get_buffer_frames(buffer_head)

            # detections for frames in frames_buffer
            self.__detections_buffer[buffer_head:self.__batch_size] = \
                self.__detection.get_segmentation_list_of_dataframes(
                    self.__frames_buffer[buffer_head:self.__batch_size]
                )

            self.__find_box_id()

            self.__calculate_features_vectors(buffer_head)

            self.__count_box_ids()

            self.__add_features_vectors()

            # draw detections
            self.__draw_video_frames(self.__batch_size // 2 if ret else self.__batch_size)

            # clean detection processing
            self.__clean_detections_box_ids_track_ids()

            # circulate circular buffers
            self.__buffers_swap(self.__batch_size // 2)

            # increment indexes
            buffer_head = (self.__batch_size // 2)
            frame_nr += (self.__batch_size // 2)
            self.__progress_bar.update(frame_nr)

            if not ret:
                self.__video_handler.set_last_frame()
                break

        self.__video_handler.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person RE-ID, produces a video with labelled persons")
    parser.add_argument("--video_path",  help="Path of input video")
    parser.add_argument("--n_frames",  default=-1, help="Number of frames to process")
    parser.add_argument("--batch_size", default=-1, help="Number of frames to process as a batch")

    args = parser.parse_args()

    ReID(args.video_path, int(args.n_frames), int(args.batch_size))
