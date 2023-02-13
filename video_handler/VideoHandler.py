import numpy as np
import cv2
import sys


class VideoHandler:
    def __init__(self, video_path: str, window_name: str = "frame") -> None:
        self.__video_path = video_path
        self.__window_name = window_name

        self.__capture = cv2.VideoCapture(self.__video_path)

        if not self.__capture.isOpened():
            sys.exit()

        self.__video_out = cv2.VideoWriter(
            str(self.__video_path.split('.')[0] + "_out.mp4"),
            cv2.VideoWriter_fourcc(*'avc1'),
            int(self.__capture.get(cv2.CAP_PROP_FPS)),
            (int(self.__capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.__capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        )

    def __is_light_or_dark(self, color: tuple):
        r, g, b = color
        hsp = np.sqrt(0.299 * (r * r) + 0.587 * (g * g) + 0.114 * (b * b))
        return hsp > 127.5  # light

    def __frame_draw_bb(self, frame: np.ndarray, bb: list, color: tuple, thickness: int = 3) -> np.ndarray:
        return cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), color, thickness)

    def __frame_draw_track_id(
            self, frame: np.ndarray, text: str, pos: tuple, color: tuple,
            font: int = cv2.FONT_HERSHEY_SIMPLEX, font_scale: int = 1, thick: int = 2, line_type: int = cv2.LINE_AA
    ) -> np.ndarray:

        text_color = (0, 0, 0) if self.__is_light_or_dark(color) else (255, 255, 255)

        text_size, _ = cv2.getTextSize(text, font, font_scale, thick)
        text_w, text_h = text_size
        frame_text = cv2.rectangle(frame, (pos[0], pos[1]), (pos[0] + text_w, pos[1] - text_h), color, -1)
        frame_text = cv2.putText(
            frame_text, text,
            (pos[0], pos[1] - font_scale - 1),
            font, font_scale, text_color, thick, line_type
        )

        return frame_text

        # return cv2.putText(frame, text, (pos[0], pos[1]), font, font_scale, color, thick, line_type)

    def get_video_w_h_fc(self) -> tuple:
        return (
            int(self.__capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.__capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self.__capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def frame_draw_info(self, frame: np.ndarray, bb: list, color: tuple, text: str) -> np.ndarray:
        bb_frame = self.__frame_draw_bb(frame, bb, color)
        text_frame = self.__frame_draw_track_id(bb_frame, text, (bb[0], bb[1]), color)
        return text_frame

    def video_read_frame(self) -> tuple:
        return self.__capture.read()

    def video_write(self, frame: np.ndarray) -> None:
        self.__video_out.write(frame)  # .copy() ?

    def set_last_frame(self):
        self.__capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        self.__capture.release()
        self.__video_out.release()

