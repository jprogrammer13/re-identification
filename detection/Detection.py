from ultralytics import YOLO
import cv2
import ultralytics
import numpy as np
import pandas as pd
from copy import deepcopy
from PIL import Image


class Detection:
    def __init__(self, epsilon: float = 30, confidence: float = 0.6, m_size: str = "s") -> None:
        self.__epsilon = epsilon
        # self.__confidence = confidence
        self.__det_model = YOLO(f"yolov8{m_size}.pt")
        self.__seg_model = YOLO(f"yolov8{m_size}-seg.pt")
        # COCO128 classes https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml

    def __distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        return np.linalg.norm(point1 - point2)

    def __is_closer_enough(self, dist: float) -> bool:
        return dist <= self.__epsilon

    def __closer_enough(self, point1: np.ndarray, point2: np.ndarray) -> float:
        dist = self.__distance(point1, point2)
        return dist if self.__is_closer_enough(dist) else np.nan

    def __boxes_to_data(self, x: ultralytics.yolo.engine.results.Results, frame_nr: int) -> dict:
        x = x.boxes.numpy()
        return {
            "frame_nr": frame_nr,
            "xyxy": x.xyxy[0],
            "xywh": x.xywh[0],
            "conf": x.conf[0],
            "center": x.xywh[0][0:2],
            "class": x.cls[0]
        }

    def __from_tensor_boxes_to_bounding_boxes_dict(self, results: list, confidence: float = 0.6) -> dict:
        det = {}
        for frame in range(len(results)):
            det[frame] = []
            boxes = results[frame].boxes.cpu().numpy()
            for box in boxes:
                if int(box.cls[0] == 0) and (box.conf[0] > confidence):
                    det[frame].append({
                        "xyxy": box.xyxy[0],
                        "xywh": box.xywh[0],
                        "conf": box.conf[0],
                        "center": box.xywh[0][0:2]
                    })
        return det

    def __from_tensor_boxes_to_bounding_boxes_dataframe(self, results: list, confidence: float = 0.6) -> pd.DataFrame:
        df = pd.DataFrame(list(results))

        applied_df = df.T.apply(
            lambda frames: [
                self.__boxes_to_data(x, df.index[df[0] == row].tolist()[0]) for row in frames for x in row
            ], axis=1, result_type="expand").T.apply(
            lambda x: x[0], axis=1, result_type="expand"
        )

        return applied_df.drop(applied_df[~(applied_df["class"] == 0) | (applied_df["conf"] < confidence)].index).drop("class", axis=1).reset_index(drop=True)

    def __from_tensor_masks_to_masks_dict(self, frames: np.ndarray, results: list, confidence: float = 0.6) -> dict:
        det = {}
        for frame in range(len(results)):
            det[frame] = []
            boxes = results[frame].boxes.cpu().numpy()
            masks = results[frame].masks.masks.cpu().numpy()
            h, w, _ = frames[frame].shape

            for i in range(len(boxes)):
                box = boxes[i]
                mask = masks[i]
                if int(box.cls[0] == 0) and (box.conf[0] > confidence):
                    det[frame].append({
                        "xyxy": box.xyxy[0],
                        "xywh": box.xywh[0],
                        "conf": box.conf[0],
                        "center": box.xywh[0][0:2],
                        "mask": (cv2.resize(mask, (h, w)) > 0).astype("uint8")
                    })
        return det

    def __from_tensor_masks_to_masks_list_of_dataframes(
            self, frames: np.ndarray, results: list, confidence: float = 0.6
    ) -> list[pd.DataFrame]:

        to_return = []

        for frame_nr in range(len(results)):
            result = results[frame_nr]

            idx = np.where(((result.boxes.cls == 0) & (result.boxes.conf > confidence)).cpu().numpy())

            df = pd.DataFrame(
                columns=["frame_nr", "xyxy", "xywh", "conf", "center", "class", "masks", "box_id", "track_id"]
            )

            if np.any(idx):
                masks = result.masks.masks[idx].cpu().numpy()
                boxes = result.boxes[idx].cpu().numpy()
                # centers = result.boxes.xywh[idx][:, :2].cpu().numpy().astype(int)

                # h, w, _ = frames[frame_nr].shape

                df = pd.DataFrame({
                    "frame_nr": frame_nr,
                    "xyxy": boxes.xyxy.astype(int).tolist(),
                    "xywh": boxes.xywh.tolist(),
                    "conf": boxes.conf.tolist(),
                    "center": boxes.xywh[:, 0:2].astype(int).tolist(),
                    "class": boxes.cls.astype(int).tolist(),
                    "mask": masks.tolist(),
                    "box_id": np.full(len(idx[0]), -1).tolist(),
                    "track_id": np.full(len(idx[0]), -1).tolist()
                })

                # df["mask"] = df["mask"].apply(lambda row: (cv2.resize(np.array(row), (h, w)) > 0).astype("uint8").tolist())
            else:
                print("No person found")
            to_return.append(df)

        return to_return

    def get_detection_dict(self, frames: np.ndarray, should_save: bool = False) -> dict:
        results = self.__det_model(deepcopy(frames), save=should_save)

        return self.__from_tensor_boxes_to_bounding_boxes_dict(results)

    def get_detection_dataframe(self, frames: np.ndarray, should_save: bool = False) -> pd.DataFrame:
        results = self.__det_model(deepcopy(frames), save=should_save)

        return self.__from_tensor_boxes_to_bounding_boxes_dataframe(results)

    def get_segmentation_dict(self, frames: np.ndarray, should_save: bool = False) -> dict:
        results = self.__seg_model(deepcopy(frames), save=should_save)

        return self.__from_tensor_masks_to_masks_dict(frames, results)

    def get_segmentation_list_of_dataframes(
            self,
            frames: list[np.ndarray],
            should_save: bool = False,
            verbose: bool = False
    ) -> list[pd.DataFrame]:
        results = self.__seg_model.predict(deepcopy(frames), save=should_save, verbose=verbose)

        return self.__from_tensor_masks_to_masks_list_of_dataframes(frames, results)

    def get_mean_bbox_dict(self, bboxes: dict) -> dict:
        mean_bb = []

        if len(bboxes) > 1:
            same_bb = []
            for bbox0 in bboxes[0]:
                same_bb.append([])
                same_bb[-1].append(bbox0)

                for bbox1 in bboxes[1]:  # bbox nel frame 1
                    if self.__is_closer_enough(self.__distance(bbox0["center"], bbox1["center"])):  # bbox0 <-> bbox1
                        same_bb[-1].append(bbox1)

                if len(bboxes) > 2:
                    for bbox2 in bboxes[2]:  # bbox nel frame 2
                        if self.__is_closer_enough(self.__distance(same_bb[-1][-1]["center"], bbox2["center"])):  # (bbox1 or bbox0) <-> bbox2
                            same_bb[-1].append(bbox2)

            for boxes in same_bb:
                bbs = {
                    "xyxy": np.mean(np.array([frame["xyxy"] for frame in boxes]), axis=0),
                    "xywh": np.mean(np.array([frame["xywh"] for frame in boxes]), axis=0),
                    "conf": np.mean(np.array([frame["conf"] for frame in boxes])),
                    "center": np.mean(np.array([frame["center"] for frame in boxes]), axis=0),
                }
                mean_bb.append(bbs)

        else:
            for bbox0 in bboxes[0]:
                mean_bb.append({
                    "xyxy": bbox0["xyxy"],
                    "xywh": bbox0["xywh"],
                    "conf": bbox0["conf"],
                    "center": bbox0["center"]
                })

        return mean_bb

    # NON FUNZIONA ed è lento
    def get_mean_bbox_dataframe(self, bboxes: pd.DataFrame) -> pd.DataFrame:
        # df = frames_detections.copy()

        # f0_f1 = pd.merge(df[df["frame_nr"] == 0], df[df["frame_nr"] == 1], how="cross")
        # f0_f1 = pd.concat([f0_f1, f0_f1.apply(
        #     lambda row: {"distance": self.__closer_enough(row["center_x"], row["center_y"])}, axis=1,
        #     result_type="expand"
        # )], axis=1).dropna()
        #
        # f1_f2 = pd.merge(df[df["frame_nr"] == 1], df[df["frame_nr"] == 2], how="cross")
        # f1_f2 = pd.concat([f1_f2, f1_f2.apply(
        #     lambda row: {"distance": self.__closer_enough(row["center_x"], row["center_y"])}, axis=1,
        #     result_type="expand"
        # )], axis=1).dropna()
        #
        # f0_f2 = pd.merge(df[df["frame_nr"] == 0], df[df["frame_nr"] == 2], how="cross")
        # f0_f2 = pd.concat([f0_f2, f0_f2.apply(
        #     lambda row: {"distance": self.__closer_enough(row["center_x"], row["center_y"])}, axis=1,
        #     result_type="expand"
        # )], axis=1).dropna()

        merged = pd.merge(
            pd.merge(
                bboxes[bboxes["frame_nr"] == 0],
                bboxes[bboxes["frame_nr"] == 1],
                how="cross"
            ),
            bboxes[bboxes["frame_nr"] == 2].add_suffix("_z"),
            how="cross"
        )

        merged = pd.concat([merged, merged.apply(
                lambda row: {
                    "dist_x_y": self.__closer_enough(row["center_x"], row["center_y"]),
                    "dist_y_z": self.__closer_enough(row["center_y"], row["center_z"]),
                    "dist_x_z": self.__closer_enough(row["center_x"], row["center_z"])
                }, axis=1, result_type="expand"
            )], axis=1).reset_index(drop=True)

        droppedna = merged.dropna()
        df = merged[merged["dist_x_z"] == merged.drop(merged[merged["dist_x_z"].isin(droppedna["dist_x_z"])].index)["dist_x_z"].min()]
        df = pd.concat([droppedna, df.loc[:, ~df.columns.str.endswith("_y")].head(1)], axis=0)

        return df