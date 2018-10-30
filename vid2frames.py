import cv2
import numpy as np
import sys

def vid2frames(video_path, dest):
    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(cv2.CAP_PROP_BRIGHTNESS, 100.0) 
    vidcap.get(cv2.CAP_PROP_FPS)
    success, image = vidcap.read()
    count = 0
    success = True

    while success:
        cv2.imwrite(dest + '/frame%d.jpg' % count, image)
        success, image = vidcap.read()
        print("Hi")
        count += 1
        
if __name__ == '__main__':
    if not len(sys.argv) == 3:
        print 'Usage: python vid2frames.py video_path dest_path'
        exit()
    else:
        video_path = sys.argv[1]
        dest_path = sys.argv[2]
        vid2frames(video_path, dest_path)