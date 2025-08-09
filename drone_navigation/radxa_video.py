import time  
from pathlib import Path
from threading import (
    Thread,
)

import cv2
import os
from datetime import datetime

import numpy as np

from ultralytics import YOLO
from ultralytics.engine.results import Results
import logging

logger = logging.getLogger(__name__)


class YoloRKNN:
    def __init__(self, model_path: Path):
        self.__model = self.__setup_model(model_path)

    def __setup_model(self, path: Path):
        model = YOLO(path.as_posix(), task="detect")

        return model

    def predict(self, image, **kwargs) -> list[Results]:
        return self.__model.predict(image, **kwargs)


class WebcamStream:
    def __init__(
        self, stream_id: str | int = 0, model: YoloRKNN | None = None
    ):
        self.stream_id = stream_id
        self.__model = model
        self.__writer = None
        self.__res = None

        self.vcap = cv2.VideoCapture(self.stream_id)

        width, height = (640, 640)

        ret = self.vcap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if not ret:
            logger.error(f"Can't set camera frame width -> {width}")

        ret = self.vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not ret:
            logger.error(f"Can't set camera frame height -> {height}")

        if ret:
            print(
                f"Camera resolution: {self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)}"
            )

        if not self.vcap.isOpened():
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)

       
        self.succsess, self.frame = self.vcap.read()
        if not self.succsess:
            print("[Exiting] No more frames to read on init")
            exit(0)

        print("Init camera")

        self.stopped = True  # reference to the thread for reading next available frame from input stream
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads keep running in the background while the program is executing

    # method for starting the thread for grabbing next available frame in input stream
    def start(self):
        self.stopped = False
        self.t.start()  # method for reading next frame

    def update(self):
        while True:
            if self.stopped:
                break

            self.succsess, self.frame = self.vcap.read()
            if not self.succsess:
                print("[Exiting] No more frames to read in update")
                self.stopped = True
                break

            if self.__model is not None:
                self.__res = self.__model.predict(self.frame, verbose=False)

        self.vcap.release()  # method for returning latest read frame

    def read(self):
        res = self.__res or None
        return res, self.frame  # method called to stop reading frames

    def stop(self):
        self.stopped = True

    @property
    def resolution(self):
        return np.array(
            [
                int(self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ]
        )

    

    def create_videowriter(self) -> cv2.VideoWriter:
    # Получаем текущую директорию
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Создаем поддиректорию 'outputs', если её нет
        output_dir = os.path.join(current_dir, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        # Генерируем уникальное имя файла на основе текущего времени
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"output_{timestamp}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        print(output_path)
        
        # Получаем текущее разрешение кадра
        frame_width = int(self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Создаем VideoWriter с текущим разрешением
        return cv2.VideoWriter(
            output_path,
            cv2.VideoWriter.fourcc(*"mp4v"),
            30,  # FPS (можно настроить)
            (frame_width, frame_height)  # Используем реальное разрешение кадра
        )




if __name__ == "__main__":
    model = YoloRKNN(Path("Training_a_neural_network_to_find_an_object_of_interest/weights/best_first-rk3588.rknn"))


    webcam_stream = WebcamStream(
        stream_id=0,
        # stream_id= "Training_a_neural_network_to_find_an_object_of_interest/videos/output_pascal_line2.mp4",

        model=model
    ) 
    

    webcam_stream.start()  # processing frames in input stream

    video_writer = webcam_stream.create_videowriter()

    num_frames_processed = 0


    start = time.time()
    try:
        while True:
            if webcam_stream.stopped:
                break

            results, frame = webcam_stream.read()
             # Визуализация результатов (пример)
            if results:
                annotated_frame = results[0].plot()
                cv2.imshow("YOLO Detection", annotated_frame)
                video_writer.write(annotated_frame)
                cv2.waitKey(10)
            else:
                cv2.imshow("Camera", frame)
                video_writer.write(frame)
                cv2.waitKey(10)


           

            num_frames_processed += 1
            cv2.imshow("frame", frame)
            key = cv2.waitKey(10)
            if key == ord("q"):
                break



        # end = time.time()
        # webcam_stream.stop()  # stop the webcam stream# printing time elapsed and fps
        # video_writer.release()
        # elapsed = end - start
        # fps = num_frames_processed / elapsed
        # print(
        #     f"FPS: {fps} , Elapsed Time: {elapsed} , Frames Processed: {num_frames_processed}"
        # )

    # closing all windows
    except KeyboardInterrupt:
        video_writer.release()
        cv2.destroyAllWindows()
