import numpy as np
import cv2
import os
import sys
import time

def read_file(sn,tn):
    s = cv2.imread(sn)
    s = cv2.cvtColor(s,cv2.COLOR_BGR2LAB)
    t = cv2.imread(tn)
    t = cv2.cvtColor(t,cv2.COLOR_BGR2LAB)
    return s, t

def get_mean_and_std(x):
    x_mean, x_std = cv2.meanStdDev(x)
    x_mean = np.hstack(np.around(x_mean,2))
    x_std = np.hstack(np.around(x_std,2))
    return x_mean, x_std

def print_progress(start_time, percent):
    """
    start_time (float): value returned by time.clock() at beginning of algorithm
    percent (float): decimal percentage
    """
    time_passed = time.clock() - start_time
    if percent==0:
        est_total_time = 0
        time_passed = int(time_passed)
    else:
        est_total_time = int(time_passed / percent)
        time_passed = int(time_passed)

    total_bar_size = 50
    bar_size = int(percent * total_bar_size)
    bar_string = "="*bar_size + (total_bar_size-bar_size)*" "
    sys.stdout.write("\r") # flush progress bar
    sys.stdout.write("[{}] {}%, [{}s / {}s]".format(bar_string, int(percent*100), time_passed, est_total_time))
    sys.stdout.flush()

def color_transfer(source, target, result_name):
    """
    Inputs: 
        source (string): image name
        target (string): the style image name
        result_name (string): what you want to name the resulting image
    
    Note: All including type, .jpg, .png, etc

    Generates source image with target image coloring and saves as
    result_name.
    """
    start_time = time.clock()

    s, t = read_file(source,target)
    s_mean, s_std = get_mean_and_std(s)
    t_mean, t_std = get_mean_and_std(t)

    height, width, channel = s.shape
    total_steps = height * width * channel

    for i in range(0,height):
        for j in range(0,width):
            for k in range(0,channel):
                x = s[i,j,k]
                x = ((x-s_mean[k])*(t_std[k]/s_std[k]))+t_mean[k]
                # round or +0.5
                x = round(x)
                # boundary check
                x = 0 if x<0 else x
                x = 255 if x>255 else x
                s[i,j,k] = x
                step = i*width*channel + j*channel + k
                print_progress(start_time, step/total_steps)

    sys.stdout.write("\n") # ends progress bar

    s = cv2.cvtColor(s,cv2.COLOR_LAB2BGR)
    cv2.imwrite('result/r'+result_name ,s)

#RUN "python color_transfer" to try this out
#feel free to add your own images and change inputs here.
color_transfer("smile.jpg", "starryBig.jpg", "starrySmile.jpg")
#os.system("pause")
