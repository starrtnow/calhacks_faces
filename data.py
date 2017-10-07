import os
import cv2

def split_video(video_path, out_dir):
    video = cv2.VideoCapture(video_path)
    success, image = video.read()
    count = 0
    while success:
        success, image = video.read()
        image_path = os.path.join(out_dir, "frame{}".format(count))
        cv2.imwrite(image_path, image)
        count += 1




