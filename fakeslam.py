#!/usr/bin/env python3
import os
import sys
import time
import cv2


import numpy as np
from slam import SLAM

if __name__ == "__main__":
  cap = cv2.VideoCapture('../tinySLAM/Fast Driving Car On Straight Road.mp4')

  W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  CNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  F = float(os.getenv("F", "525"))

  # if W > 1024:
  #     downscale = 1024.0/W
  #     F *= downscale
  #     H = int(H * downscale)
  #     W = 1024
  print("using camera %dx%d with F %f" % (W,H,F))

  K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
  Kinv = np.linalg.inv(K)
  slam = SLAM(W, H, K)

  fn = 0
  pos_x = 0
  dir_x = True
  while 1:
    fn += 1
    ret, frame = cap.read()

    slam.process_frame(frame, None, None)

    img = slam.mapp.frames[-1].annotate(frame)

    # flip flop
    if pos_x > 10:
      dir_x = False
    elif pos_x < -10:
      dir_x = True
    pos_x += 0.5 * (1 if dir_x else -1)
