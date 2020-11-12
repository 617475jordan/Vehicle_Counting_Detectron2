# LIBRARIES (EXPECTING YOU HAVE ALREADY INSTALLED THE DETECTRON2 PACKAGES)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import random

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo


# VIDEO IMPORT

cap = cv2.VideoCapture('traffic_clip.mp4')

'''
We are going to deveolpe a little program which detects the number of vehicles which drive through a certain highway. To make it simple,
we are only going to track the vehicles going away (i.e appearing from the bottom of the screen). To do so, we are going to use the 
library Detectron2, making use of the object box detection.

First of all we will define the region in which the selected vehicles drive (the leftmost side of the screen), exploring the number of pixels
of the input. As it is a centered image, we will get only the vehicles whose center x cordinate lies in between 0 and width/2.
'''

width = cap.get(3)
height = cap.get(4)


'''
So, now we create a function which recieve a vector with (x1,y1,x2,y2) and returns the position of the center (xc,yc). We need to keep in
mind that Detectron2 does not give us a integer pixel value.
'''

def center(x):
  x1,y1,x2,y2 = x
  return (x1+x2)/2,(y1+y2)/2


'''
So what we obtained from previous results is that: 
  The detection rectangle is [(0,500),(w/2,540)]

  The lines sepearating the lanes are start, from left to right at [50,235,410,550]

  Hence, we need to create a function that return the lane the car is at :
    [0] --> left
    [1] --> middle
    [2] --> right

'''

detection_box = [0,500,int(width/2),540]
lines = [50,235,410,550]

def which_lane(x):
  xx,_ = center(x)
  
  if lines[0]<xx<lines[1]:
    return 0
  elif lines[1]<xx<lines[2]:
    return 1
  else: 
    return 2

'''
So what we want our code to do is:
  -Check if a car (or cars) is inside the box and mark the lane as occupied, with a function.
  -Advance a few frames, and check for cars inside the boxes. Four different things can happen:
    · A previously empty lane is still empty, we do nothing.
    · A previously empty lane is now occupied, we increase the counter and mark it as occupied.
    · A previously occupied lane is now not occupied, we mark it as empty.
    · A previously occupied lane is still empty, we do nothing. 
'''

def inside_box(x):
  xx,yy=center(x)
  x1,y1,x2,y2 = detection_box
  if (x1<xx<x2) and (y1<yy<y2):
    return True
  else:
    return False

'''
We define some colors to enhence the visualization, in BGR, and some parameters for the text:
'''

red = (0,0,255)

blue = (255,0,0)

orange = (10,180,253)

black = (0,0,0)

white = (255,255,255)


font = cv2.FONT_HERSHEY_SIMPLEX

fontSize = 2

thickness = 2

'''
First we start with some definitions and imports:
  · cap = import again the video so it starts at 0
  · skip_frames : how often we are going to inspect the video, which is captured at 25fps/second.
  · total_frames : total number of frames of the video
  · count = number of cars the have went through the area
  · lane_count = represent the count for each lane
  · num_frame = frame to inspect
  · lines_old = array used to see if a lane is empty or not
  · objects_old = keep track of what kind of object goes through
  · objects_dict = dictionary for the numerical reference of the object
  · objects_count = dicctionary to store values
  · We import the pretrained model configuration and wieghts.
  · We define the treshold
  · We store the model in predictor as a DefaultPredictor()
'''

cap = cv2.VideoCapture('traffic_clip.mp4') 

skip_frames = 6

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 4*25

count = 0

lane_count = np.zeros(3)

num_frame = 1

lines_old = np.zeros(3)

objects_old = np.array(['empty']*3)

objects_dict = {2 : 'car', 7 : 'truck', 3 : 'motor', 5: 'bus',  0 : 'motor'}

objects_count = {'car' : 0, 'truck' : 0, 'bus' : 0, 'motor' : 0}

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
predictor = DefaultPredictor(cfg)


link = '/content/result.mp4'
video = cv2.VideoWriter(link,cv2.VideoWriter_fourcc(*'mp4v'),25,(int(width),int(height)),True)

'''
We start with the creation of the loop:
  - Inspect every desired frame.
  - Make a copy of the line array.
  - Look at the predicted boxes to see if there are any cars inside the detecting box, detecting the line in which they are,
  and storing the values in a new array.
  - Following the structure criteria, we follow: if the values are the same or 0, we do nothing. If the value is now 1, we increase the counter.
'''


while(cap.isOpened()):
  if num_frame > total_frames:
    break
    
  cap.set(1,num_frame-1)
  ret, frame = cap.read()
  if (num_frame + skip_frames) % skip_frames == 0:
    output = predictor(frame)
    lines_new = np.zeros(3)
    objects_new = np.array(['empty']*3)

    boxes = []
    i = 0

    for box in output['instances'].pred_boxes:
      x1 = int(box[0].item())
      y1 = int(box[1].item())
      x2 = int(box[2].item())
      y2 = int(box[3].item())
      if x1 < width/2:
        boxes.append([[x1,y1,x2,y2],output['instances'].pred_classes[i].item()])
      i +=1

  
    for box in boxes:
      x,y = center(box[0])
      if inside_box(box[0]):
        lines_new[which_lane(box[0])] = 1
        objects_new[which_lane(box[0])] = objects_dict[box[1]]
        
        
    for i in range(3):
      if (lines_old[i] != lines_new[i]) and lines_new[i] == 1:
        count +=1
        lane_count[i] +=1
        objects_count[objects_new[i]] += 1 
    if 1 in lines_new:
      color = red
    else:
      color = blue

    lines_old = np.copy(lines_new)
    objects_old = np.copy(objects_new)
  x1,y1,x2,y2 = detection_box
  blk = np.zeros(frame.shape, np.uint8)
  cv2.rectangle(blk, (x1, y1), (x2, y2), color, cv2.FILLED)
  frame = cv2.addWeighted(frame, 1.0, blk, 0.25, 1)
  frame = cv2.putText(frame,'Counts = ' + str(count),(195,100),font,fontSize,black,thickness)
  frame = cv2.putText(frame,'L1=' + str(int(lane_count[0])),(210,150),font,1,black,thickness)
  frame = cv2.putText(frame,'L2=' + str(int(lane_count[1])),(350,150),font,1,black,thickness)
  frame = cv2.putText(frame,'L3=' + str(int(lane_count[2])),(490,150),font,1,black,thickness)

  frame = cv2.putText(frame,'car',(900,50),font,1,black,thickness)
  frame = cv2.putText(frame,str(int(objects_count['car'])),(1150,50),font,1,black,thickness)
  frame = cv2.putText(frame,'truck',(900,80),font,1,black,thickness)
  frame = cv2.putText(frame,str(int(objects_count['truck'])),(1150,80),font,1,black,thickness)
  frame = cv2.putText(frame,'motorbike',(900,110),font,1,black,thickness)
  frame = cv2.putText(frame,str(int(objects_count['motor'])),(1150,110),font,1,black,thickness)

  num_frame += 1
  video.write(frame)

cap.release()
video.release()
cv2.destroyAllWindows()
