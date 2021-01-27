import numpy as np
import cv2
import os
import json
import argparse

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

parser = argparse.ArgumentParser()
parser.add_argument("data_directory",  type= dir_path)
args = parser.parse_args()

def face_detect(img_path):
    # load face detectors
    front_face_cascade = cv2.CascadeClassifier('save/haarcascade_frontalface_default.xml')
    Profile_face_cascade = cv2.CascadeClassifier('save/haarcascade_profileface.xml')

    #set image path to loop through all the images
    # img_path = 'Project3_FaceDetection/Validation folder/images/'

    # empty list for results
    json_list = []
    print(f'----------------------------------------- Face Detection Start -----------------------------------------\n')
    print(f'-----Image Directory: {img_path} ')
    count = 1
    for img_name in os.listdir(img_path):
        img = cv2.imread(f'{img_path}/{img_name}')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        print('** Now looking at Image: {} -- count {} **'.format(img_name,count), end="\r", flush=True)
        count += 1
        front_faces = front_face_cascade.detectMultiScale(gray, 1.05, 5)
        # Profile_faces = Profile_face_cascade.detectMultiScale(gray, 1.04, 6)
        for (x, y, w, h) in front_faces:
            result = {"iname": f'{img_name}', "bbox": [int(x), int(y), int(w), int(h)]}
            json_list.append(result)

        # for (x, y, w, h) in Profile_faces:
        #     result = {"iname": f'{img_name}', "bbox": [int(x), int(y), int(w), int(h)]}
        #     json_list.append(result)

    with open(f"{img_path}/results.json", 'w') as f:
        json.dump(json_list, f)
        print(f'----- result.json saved to {img_path} ')

if __name__ == '__main__':
    face_detect(args.data_directory)


# python FaceDetector.py "Project3_FaceDetection/Validation folder/images"
# python ComputeFBeta.py "Project3_FaceDetection/Validation folder/images/results.json" "Project3_FaceDetection/Validation folder/ground-truth.json"

