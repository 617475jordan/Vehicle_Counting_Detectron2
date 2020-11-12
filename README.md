# Vehicle Counting with Detectron2
Project about vehicle counting in a highway using Detectron2, a [software](https://github.com/facebookresearch/detectron2/) provided by Facebook AI Research, which implements state of the art object detection algorithms. This is a case sensitive approach, as the detection area, the different car lines and the FPS update is done
to work in accordance with the footage provided. 

**How it works:**
 - First of all, the detection area is defined (i.e. the area in the footage that every time a different vehicle goes through it the count is updated) and also the range of every car lane. It must be kept in mind, that due to the positioning of the videocamera in the recorded footage, the lines separating the lanes have to be shifted towards the left of the frame.
 - We are going to count as a detection when the center of the detected vehicle is inside the detaction area, and also we are going to record the lane it is at. To do so, the functions **center(x)** and **which_lane(x)** have been created. 
 - The model detects the objects of the feed images every N frames *(stored at the variable skip_frames)*, so that we increase speed. It must be noted that the number of skipped frames must be in accordance with velocity of the cars, the FPS of the footage or the width of the detection area. For instance,  if we skip a lot of frames, there is a good chance that a car is not inside the detection area in any of the the evaluated frames. Moreover, we shold be concerned about what happens if it is the other way around and the vehicle stays for more than one frame inside the detection area. In this case, what we do is that we compare the lanes occupancy with the one of the previously inspected frame, and only update the counter if it was previously empty.
 *It should also be notted that these problems could be avoided if on top of object detection we performed object tracking.*
 - Hence, the project keeps track on how many vehicles go through each lane and computes also the **total count**. Moreover, the data about the **types of vehicles** is also collected and displayed on screen.
 
 **How to use it**
 
The libraries used are **numpy**,**openCV (cv2)** and **detectron2**. The latter is the only one which could give you problems with the installation, which you will find how to do it [here](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). You can also download in the assets folder the short clip I used to perform the detection task, and make that you check the full original [video](https://www.youtube.com/watch?v=wqctLW0Hb_0&t=1272s&ab_channel=AndreyNikishaev) (32 minuts worth of footage) if you want extra data.
