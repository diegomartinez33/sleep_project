# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 22:12:26 2018

@author: diega
"""
import numpy as np
import math
import pdb
import time

## Create Data for Experiments module

# Divide signal in 30s windows

# Extract windows depending on size
def sepWins(signalDict, winTime, dataType='SC', events='Yes'):
    """ Function to separete epochs from signals of PSG 
    Depending on:
        WinTime : must be a 5 multiple and less than 30
        dataType : 'SC' or 'ST'
        events : 'Yes' or 'Not' to include the events marker channel
        
    """
    
    signals = signalDict['sigData']
    targets = signalDict['Targets']
    
    #Diferent sampling frequency dor data
    Fs_1 = 100
    if dataType == 'ST':
        Fs_2 = 10
    else:
        Fs_2 = 1
    
    #Define window size depending on sampling frequency
    winSize_1 = winTime * Fs_1
    winSize_2 = winTime * Fs_2
    
    #Max windows as total annotated windows (problems with )
    annotedSigVal = int(targets[-1][1]) * Fs_1  
    signals_test = signals[0][:annotedSigVal]
    max_windows = math.floor(len(signals_test)/winSize_1)
    print(max_windows)
    
    
    #Eliminate event markers
    if events == 'No':
        signals = np.delete(signals,-1)
    
    
    # Get windows
    cont = 0
    cont2 = 0
    for sig in signals:
        #pdb.set_trace()
        signalWins = max_windows
        #windows for 1Hz or 10 Hz data
        if math.floor(len(sig)/1000000) < 1:
            #(len(sig) % 1000) > 0:
            #signalWins = math.floor(len(sig)/winSize_2)
            sig_1 = sig[:signalWins*winSize_2]
            sig_2 = sig[signalWins*winSize_2:]
            crop_signal = np.reshape(sig_1,(1,signalWins,winSize_2))
            if cont2 >0:
                Data2 = np.append(Data2,crop_signal,axis=0)
            else:
                Data2 = crop_signal
            cont2+=1
        #windows 100 Hz data
        else:
            #signalWins = math.floor(len(sig)/winSize_1)
            sig_1 = sig[:signalWins*winSize_1]
            sig_2 = sig[signalWins*winSize_1:]
            crop_signal = np.reshape(sig_1,(1,signalWins,winSize_1))
            if cont >0:
                Data = np.append(Data,crop_signal,axis=0)
            else:
                Data = crop_signal
            cont+=1
    
    print(Data.shape) # 100Hz signals
    if 'Data2' in locals():
        print(Data2.shape) # 1Hz signals
    
    #### Get targets (only to a window size <= 30)
    
    #% Classes:
    #% awake = 'W' -----------------> 0
    #% NREM Sleep Stage 1 = '1' ----> 1
    #% NREM Sleep Stage 2 = '2' ----> 2
    #% NREM Sleep Stage 3 = '3' ----> 3
    #% NREM Sleep Stage 4 = '4' ----> 4
    #% REM Sleep Stage = 'R' -------> 5
    #% Movement_time = 'e','?'------> 6
    
    # Classes dictionary
    Classes = {
            'W' : 0,
            '1' : 1,
            '2' : 2,
            '3' : 3,
            '4' : 4,
            'R' : 5,
            'e' : 6,
            '?' : 6,
    }
    
    #print(targets.shape)
    #print(targets[-1][1])
    
    # Change letters for numbers
    labs = [[] for i in range(max_windows)]
    anot_num = 0
    for j in range(max_windows):
        aux2 = (j + 1) * winTime       
        if aux2 > int(targets[anot_num][1]):
            anot_num += 1
        classStr = targets[anot_num][0]        
        labs[j] = Classes[classStr]   
    labs = np.array(labs)
    
    #Dictionary to save all data
    # Keys = ['data', 'data2', 'targets']
    infoPerWin = {}
    infoPerWin['data'] = Data
    try:
        infoPerWin['data2'] = Data2
    except:
        print('data2 key is unused')
        infoPerWin['data2'] = np.array([])
    infoPerWin['targets'] = labs
    return infoPerWin

def winSamp(dataDict,winTime,sampSize, dataType='SC'):
    """ Function to make subsampling of signals epochs
    
    signalDict must be a dictionary with data ('data' and 'data2') sampled 
    by 30 s windows and its targets ('targets')
    
    data shape (channels,wins,sizeWins) = (3,2650, 3000) or (4, 2650, 30)
    winTime is the length of window selected (5s,10s,15s) and must be a divisor 
    of 30 s
    Size of sampling (sampSize) in seconds (1s, 0,5s, etc...)
    
    """
    
    #Info from 'data' key of dictionary dataDict
    Fs_1 = 100
    data1 = dataDict['data']
    
    #Info from 'data2' key of dictionary dataDict
    if dataType == 'ST':
        Fs_2 = 10
    else:
        Fs_2 = 1
    data2 = dataDict['data2']
    
    targets = dataDict['targets']
    
    if winTime == 30:
        return dataDict
    else:
        def cutWindows(Data,Fs):
            """ Function that make the cuts of sampling for all windows """
            movTime = sampSize * Fs
            winSize = winTime * Fs
            channels = Data.shape[0]
            finalData = np.empty((channels,1,winSize))
            finalTargets = np.empty((1))
            for i in range(Data.shape[1]):
                origWin = Data[:,i,:]
                aux1 = 0
                aux2 = winSize
                while aux2 <= origWin.shape[1]:
                    sampWin = origWin[:,aux1:aux2]
                    sampWin = np.expand_dims(sampWin,axis=1)
                    finalData = np.append(finalData,sampWin,axis=1)
                    finalTargets = np.append(finalTargets,targets[i])
                    aux1 += movTime
                    aux2 += movTime
                print(i)
            
            dataSampled = {}
            dataSampled['data'] = finalData
            dataSampled['targets'] = finalTargets
            return dataSampled
        
        # For data 1
        sampData1 = cutWindows(data1,Fs_1)
            
        #For data 2
        sampData2 = cutWindows(data2,Fs_2)
        
        #Checks if targets array are the same length for data 1 and data 2
        if sampData1['targets'].all() == sampData2['targets'].all():
            print('Targets are the same for both type of data sampling (100 and 1 Hz)')
        else:
            raise NameError('Targets are not the same for both type of data sampling (100 and 1 Hz)')
    
    print(sampData1['data'].shape)
    print(sampData2['data'].shape)
    print(sampData1['targets'].shape)
    
    #Dictionary to save all cuted and oversampled data
    # Keys = ['data', 'data2', 'targets']
    FinalSampData = {}
    FinalSampData['data'] = sampData1['data']
    FinalSampData['data2'] = sampData2['data']
    FinalSampData['targets'] = sampData1['targets']
    return FinalSampData
        
def getSigImages(dataDict,ostype=1):
    """ Function that creates arrays from figures of signals for the first 3 
    channels of the PSG signals.
    Returns a 3D array with all images
    
    Select ostype = 0 if you are in a python available interface
    Select ostype = 1 if your uisng the cluster to create backends for figures
    
    """

    if ostype == 1:
        import matplotlib
        matplotlib.use('Agg')
    
    import matplotlib.pyplot as plt
    from PIL import Image
    
    dataEx = dataDict['data']
    
    #fig = plt.figure()
    
    
    def getImg(arrDat):
        # Function to create an image fromn an 2D array data
        plt.clf()
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
        x = range(len(arrDat[0,:]))
        
        for i in range(arrDat.shape[0]):
            y = arrDat[i,:] - ((2 ** i) * 200)
            ax.plot(x, y, color='black', linewidth=1)
        
        ax.set_xlim(x[0], x[-1])
        ax.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #plt.savefig("test.png", bbox_inches='tight', pad_inches = 0)
        
        fig.canvas.draw ( )
          
        # Get the RGBA buffer from the figure
        w,h = fig.canvas.get_width_height()
        buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
        
        buf.shape = ( w, h, 4 )
         
        # canvas.tostring_argb give pixmap in ARGB mode. 
        # Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll ( buf, 3, axis = 2 )
        
        w, h, d = buf.shape
        
        im = Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )
        im = im.convert('L')
        arr_im = np.array(im)
        plt.close(fig)
        return arr_im
    
    print('number of images are: ' + str(dataEx.shape[1]))
    im1 = getImg(dataEx[:,0,:])
    AllImages = np.expand_dims(im1,axis=0)
    #print('Image: ',1)
    
    # Create images for all data in the dictionary
    for i in range(1,dataEx.shape[1]):
        someData = dataEx[:,i,:]
        im2 = getImg(someData)
        im2 = np.expand_dims(im2,axis=0)
        AllImages = np.append(AllImages,im2,axis=0)
        #print('Image: ',i)
    
    return AllImages
        
        

    