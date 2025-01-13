import cv2
import mediapipe as mp
from glob import glob
from collections import defaultdict
import pickle
import numpy as np
from tqdm import tqdm
import math
from typing import List, Mapping, Optional, Tuple, Union


import os

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

image_save = True

def main():

    data_save = defaultdict(list)

    for video_path in tqdm(glob("#mp4 format video list")):
        v_name = os.path.basename(video_path)
        v_folder = video_path.split("/")[-2]
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f'Faild to load video file {video_path}'

        # optional
        return_heatmap = False
        num = 0
        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=True,
            model_complexity=1) as pose:
            while (cap.isOpened()):
                num +=1
                flag, img = cap.read()
                if not flag:
                    break

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                img.flags.writeable = False
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = pose.process(img)
                image_hight, image_width, _ = img.shape
                try:
                    pointslist = []
                    for i in range(0,33):
                        # 2d
                        landmark = results.pose_landmarks.landmark[i]
                        x,y = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                    image_width, image_hight)
                        
                        pointslist.append([x,y,landmark.z,landmark.visibility])

                        # 3d
                        # landmark = results.pose_world_landmarks.landmark[i]
                        # pointslist.append([landmark.x,landmark.y,landmark.z])


                    # save pose_results
                    data_save[f"{v_folder}_{v_name}"].append([pointslist])
                except:
                    print(v_name)
                    print(num)

                if image_save:
                    img.flags.writeable = True
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    mp_drawing.draw_landmarks(
                        img,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                    mp_drawing.plot_landmarks(
                        results.pose_world_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                        save_path=f"tmp/{num}.png")


            cap.release()

    with open('video2points.pkl', 'wb') as jfile:
        
        # A new file will be created
        pickle.dump(data_save, jfile)

    print(len(data_save))


if __name__ == '__main__':
    main()
