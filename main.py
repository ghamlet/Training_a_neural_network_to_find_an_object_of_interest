from pioneer_sdk import Pioneer, Camera
import cv2
import time
from collections import defaultdict, deque
from flight_utils import load_flight_coordinates, get_config
from drone_navigation import FlightMissionRunner, CustomPioneer, WebcamStream, YoloRKNN
from ultralytics import YOLO
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




COORDS_TEST = [

    [0,0,1],
]


if __name__ == "__main__":
    try:
        flight_height = 1.5

        MAP_POINTS = load_flight_coordinates()


        model = YoloRKNN(Path("rknn_model/best_sana-rk3566.rknn"))

        webcam_stream = WebcamStream(stream_id=0,
                                     model=model)
        

        webcam_stream.start()  # processing frames in input stream
        video_writer = webcam_stream.create_videowriter()
        num_frames_processed = 0



        pioneer_conf = get_config('local')  # или 'global'   local

        # Инициализация миссии
        mission = FlightMissionRunner(MAP_POINTS)
        
        # Инициализация дрона
        pioneer = CustomPioneer(
            name="pioneer",
            ip=pioneer_conf["ip"],
            mavlink_port=pioneer_conf["port"],
            connection_method="udpout",
            device="dev/serial0",
            baud=115200,
            verbose=True
        )

        

        pioneer.arm()
        pioneer.takeoff()
        
        # Начало миссии
        first_point = mission.get_next_point()
        x, y, z = first_point
        pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)
        

        # Основной цикл миссии
        while not mission.is_complete():
            if webcam_stream.stopped:
                break


            results, frame = webcam_stream.read()
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



            if pioneer.point_reached(threshold=0.3):
                # Переход к следующей точке
                next_point = mission.get_next_point()
                if next_point:
                    x, y, z = next_point
                    pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)
            

           

    except Exception as e:
        print(f"Ошибка обработки: {str(e)}")
    
              

    except KeyboardInterrupt:
        print("\nПолучен сигнал прерывания (Ctrl+C)")
        video_writer.release()
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
    
    finally:
        print("Завершение работы")
        pioneer.land()
        pioneer.disarm()
        pioneer.close_connection()
        cv2.destroyAllWindows()