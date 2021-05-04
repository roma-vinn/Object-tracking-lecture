# Motion detection and object tracking lecture

## Agenda
- Problem statement
- Basic approach
    - Optical flow
    - Lukas-Kanade algorithm
- Tracking-by-detection
    - Kalman filter
    - Hungarian algorithm
    - SORT
    - DeepSORT
- Tracking with network flows

### Links used
1. Article about Optical Flow: https://nanonets.com/blog/optical-flow/
2. OpenCV tutorial on Optical Flow: https://docs.opencv.org/master/d4/dee/tutorial_optical_flow.html
3. The least squares fitting explanation: https://mathworld.wolfram.com/LeastSquaresFitting.html
4. Paper "Pyramidal Implementation of the Lucas Kanade Feature Tracker": http://robots.stanford.edu/cs223b04/algo_tracking.pdf
5. Kalman filer site: https://www.kalmanfilter.net/alphabeta.html
6. Explanation of Hungarian algorithm: https://brilliant.org/wiki/hungarian-matching/#the-hungarian-algorithm-using-an-adjacency-matrix
7. Original SORT paper: https://arxiv.org/abs/1602.00763
8. Original DeepSORT paper: https://arxiv.org/abs/1703.07402
9. Fascinating TUM lecture about MOT: https://www.youtube.com/watch?v=BR3Y5bAz5Dw&t
10. Super useful practical article from experienced engineer,
 but hard to read without a prior knowledge: https://habr.com/ru/company/recognitor/blog/505694/


# Task
|Subtask|Points|
|-------|--------|
|Add people tracker to the [video](https://github.com/opencv/opencv/blob/master/samples/data/vtest.avi), submit your code + video result (in my Telegram). Play with init/lifespan constants to obtain better results.|10|
|Compare trackers without appearance metrics ([SORT](https://github.com/abewley/sort) / [Norfair](https://github.com/tryolabs/norfair) etc.) to ones that use it ([DeepSORT](https://github.com/nwojke/deep_sort) / [FairMOT](https://github.com/ifzhang/FairMOT) etc.)|10|
|Online presentation of your code (Google Meet/Zoom): show me, how you obtained your video results|10|
|Answer my additional questions|5|

## Deadline
Must be submitted not later than **05.05.2021**.

## Description of given code
In this repo you can also find instructions along with the `blank.py` script that:
- reads video frame-by-frame in a format used by YOLOv5,
- performs object detection (particularly, detects people),
- writes video with detected bounding boxes on the video.

## Installation
**Highly recommend to use virtual environment instead of global interpretator.**

In your terminal:
1. `git clone https://github.com/ultralytics/yolov5.git`
2. `cd yolov5`
3. `pip install -r requirements.txt`
4. `cd ..`

## Weights
You can download weights for YOLOv5 from its [official GitHub repo](https://github.com/ultralytics/yolov5/releases).

## Usage
In your terminal:

`python blank.py --input-path $INPUT_VIDEO --save-path $OUTPUT_VIDEO --weights-path $PATH_TO_MODEL_WEIGHTS`
