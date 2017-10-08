import os
import cv2
#import openface
from scipy import misc

def split_video(video_path, out_dir):
    video = cv2.VideoCapture(video_path)
    success, image = video.read()
    count = 0
    while success:
        success, image = video.read()
        image_path = os.path.join(out_dir, "frame{}.png".format(count))
        cv2.imwrite(image_path, image)
        print("Split frame {} to {}".format(count, image_path))
        count += 1

#align = openface.AlignDlib("dl.dat")
faceCascade = cv2.CascadeClassifier("cs.xml")
def cut_head(image_path, out_dir, resolution=(64, 64)):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bb = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors=5,
            minSize=(30, 30)
    )

    if len(bb) < 1:
        print("No faces detected")
        return

    x, y, w, h = bb[0]
    image = image[y:(y+h), x:(x+w)]
    image = cv2.resize(image, resolution, interpolation=cv2.INTER_AREA)

    _, image_name = os.path.split(image_path)

    cv2.imwrite(os.path.join(out_dir, image_name), image)

def paste_head(in_image, back_image):
    s_img = cv2.imread(in_image)
    l_img = cv2.imread(back_image)

    gray = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
    bb = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors=5,
            minSize=(30, 30)
    )


    x, y, w, h = bb[0]
    s_img = cv2.resize(s_img, (w, h), interpolation=cv2.INTER_AREA)
    l_img[y:y+s_img.shape[0], x:x+s_img.shape[1]] = s_img
    return l_img 













