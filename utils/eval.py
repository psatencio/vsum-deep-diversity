'''
    program: Utility scripts for experimental evaluations
    author: Pedro Atencio Ortiz
    copyright: 2017
'''

def SumMe_evaluation(videoName, reference_summary, plot_graph=True):
    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Demo for the evaluation of video summaries
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % This script takes a random video, selects a random summary
    % Then, it evaluates the summary and plots the performance compared to the human summaries
    %
    %%%%%%%%
    % publication: Gygli et al. - Creating Summaries from User Videos, ECCV 2014
    % author:      Michael Gygli, PhD student, ETH Zurich,
    % mail:        gygli@vision.ee.ethz.ch
    % date:        05-16-2014
    %%%%%%%%
    % Modified by: Pedro Atencio, 2018.
    % Program has been recoded as a function to be able to use by an external method.
    '''
    import os 
    import sys
    from summe import *
    import numpy as np
    import random
    import pickle as pk

    ''' PATHS ''' 
    HOMEDATA='video_dataset/SumMe/GT';
    HOMEVIDEOS='../../video_dataset/SumMe/videos/';

    #In this example we need to do this to now how long the summary selection needs to be
    gt_file=HOMEDATA+'/'+videoName+'.mat'
    gt_data = scipy.io.loadmat(gt_file)
    nFrames=gt_data.get('nFrames')

    '''Example summary vector''' 
    #selected frames set to n (where n is the rank of selection) and the rest to 0
    n = 20
    summary_selections={}
    summary_selections[0] = reference_summary * n

    '''Evaluate'''
    #get f-measure at 15% summary length
    [f_measure,summary_length]=evaluateSummary(summary_selections[0],videoName,HOMEDATA)
    
    print('F-measure : %.3f at length %.2f' % (f_measure, summary_length))

    '''plotting'''
    methodNames={'S-VSUM'};
    if(plot_graph):
        plotAllResults(summary_selections,methodNames,videoName,HOMEDATA);
    
    return f_measure