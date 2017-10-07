import os
import cv2
import openface
from scipy import misc

def split_video(video_path, out_dir):
    video = cv2.VideoCapture(video_path)
    success, image = video.read()
    count = 0
    while success:
        success, image = video.read()
        image_path = os.path.join(out_dir, "frame{}.png".format(count))
        print("Split frame {}".format(count))
        cv2.imwrite(image_path, image)
        count += 1

align = openface.AlignDlib("dl.dat")
def cut_head(image_path, out_dir):
    image = misc.imread(image_path)
    bb = align.getLargestFaceBoundingBox(image, skipMulti=True)

    if bb == None:
        print("No faces detected")
        return

    top, left, bottom, right = bb.top(), bb.left(), bb.bottom(), bb.right()
    image = image[top:bottom, left:right]

    _, image_name = os.path.split(image_path)

    misc.imsave(os.path.join(out_dir, image_name), image)






