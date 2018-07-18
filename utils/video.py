'''
    description: Utility scripts for video preprocessing
    author: Pedro Atencio Ortiz
    mail: pedroatencio@itm.edu.co
    date: 16-07-2018
'''
import numpy as np
import pickle

def vsampling_random(video_generator, prob):
    ''' 
    Returns video frames sampled randomly uniform with a probability.
    
    Arguments:
        video_generator -- a generator of numpy ndarray containing video. 
        prob -- probability of keeping frames from video.
        
    Returns:
        sampled_video -- numpy ndarray containing sampled video.
    '''
    sampled_video = [frame for frame in video_generator if np.random.rand() > (1.0 - prob)]
    
    return sampled_video

def vsampling_uniform(video_generator, slide_width=1, shift=1, irate=30, frate_s=1):
    ''' 
    Returns video frames sampled uniformly with sliding window of width 
    slide_width in seconds and shift of shift seconds, using a irate of rate fps.
    Finally every segment is sampled to frate frames per segment.

    Arguments:
        video_generator -- video_generator -- a generator of numpy ndarray containing video.
        slide_width -- time window in seconds to consider as a segment.
        shift -- time window in seconds to shift sliding window with slide_with
        irate -- frame rate of input video
        frate_s -- number of frames to be extracted from each segment

    Returns:
        sampled_video -- numpy ndarray containing sampled video.
    '''
    v = 0
    segment = []
    sampled_video = []
    segment_width = irate * slide_width
    k = int(segment_width / 2)-1

    i = 0
    
    print("Sampling video to 1 frame per second.")

    for frame in video_generator:
        segment.append(frame) #accumulates frames that constitute a segment
        
        if((i+1) % segment_width == 0):
            #print(i+1)
            #print("append")
            f = segment[k]
            segment = [] #reset segment to create a new one
            sampled_video.append(f)
        i += 1 

    return sampled_video
    
    