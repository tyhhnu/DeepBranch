#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 21:06:36 2018

@author: ttt
"""

import os
import re as rere
import numpy as np
from scipy import io as sio

def loadimg(file):
    if file.endswith('.mat'):
        filecont = sio.loadmat(file)
        img = filecont['img']
        for z in range(img.shape[-1]): # Flip the image upside down
            img[:,:,z] = np.flipud(img[:,:,z])
        img = np.swapaxes(img, 0, 1)
    elif file.endswith('.tif'):
        img = loadtiff3d(file)
    elif file.endswith('.nii') or file.endswith('.nii.gz'):
        import nibabel as nib
        img = nib.load(file)
        img = img.get_data()
    else:
        raise IOError("The extension of " + file + 'is not supported. File extension supported are: *.tif, *.mat, *.nii')

    return img
    

def loadtiff3d(filepath):
    """Load a tiff file into 3D numpy array"""
    from libtiff import TIFF
    tiff = TIFF.open(filepath, mode='r')
    stack = []
    for sample in tiff.iter_images():
        stack.append(np.rot90(np.fliplr(np.flipud(sample))))
    out = np.dstack(stack)
    tiff.close()

    return out


def writetiff3d(filepath, block):
    from libtiff import TIFF
    try:
        os.remove(filepath)
    except OSError:
        pass

    tiff = TIFF.open(filepath, mode='w')
    block = np.swapaxes(block, 0, 1)

    for z in range(block.shape[2]):
        tiff.write_image(np.flipud(block[:, :, z]), compression=None)
    tiff.close()

def writetiff2d(filepath, block):
    from libtiff import TIFF
    try:
        os.remove(filepath)
    except OSError:
        pass

    tiff = TIFF.open(filepath, mode='w')
    block = np.swapaxes(block, 0, 1)

#    for z in range(block.shape[2]):
    tiff.write_image(np.flipud(block[:, :]), compression=None)
    tiff.close()


def loadswc(filepath):
    '''
    Load swc file as a N X 7 numpy array
    '''
    swc = []
    with open(filepath) as f:
        lines = f.read().split("\n")
        for l in lines:
            if not l.startswith('#'):
                cells = l.split(' ')
                if len(cells) ==7:
                    cells = [float(c) for c in cells]
                    # cells[2:5] = [c-1 for c in cells[2:5]]
                    swc.append(cells)
    return np.array(swc)
def loadmarker(filepath):
    '''
    Load swc file as a N X 7 numpy array
    '''
    marker = []
    with open(filepath) as f:
        lines = f.read().split("\n")
        for l in lines:
            if not l.startswith('#'):
               cells = rere.split(', ',l)
               if len(cells) == 7:
                   cells = [float(c) for c in cells[0:3]]
                   marker.append(cells)
    return np.array(marker)
def loadmarker1(filepath):
    '''
    Load swc file as a N X 7 numpy array
    '''
    marker = []
    with open(filepath) as f:
        lines = f.read().split("\n")
        for l in lines:
            if not l.startswith('#'):
               cells = rere.split(', ',l)
               if len(cells) == 8:
                   cells = [float(c) for c in cells[0:4]]
                   marker.append(cells)
    return np.array(marker)
def savemarker(filepath,marker):
    
    with open(filepath, 'w') as f:
        for i in range(marker.shape[0]):
            markerp=[marker[i,0],marker[i,1],marker[i,2],0,1,' ',' ']
        
            print('%.3f, %.3f, %.3f, %d, %d, %s, %s'  %  (markerp[0], markerp[1], markerp[2], markerp[3],
                                                          markerp[4], markerp[5], markerp[6]),file=f)
def savemarker1(filepath,marker):
    
    with open(filepath, 'w') as f:
        for i in range(marker.shape[0]):
            markerp=[marker[i,0],marker[i,1],marker[i,2],marker[i,3],1,' ',' ']
        
            print('%.3f, %.3f, %.3f, %.3f, %d, %s, %s'  %  (markerp[0], markerp[1], markerp[2], markerp[3],
                                                          markerp[4], markerp[5], markerp[6]),file=f)
def saveswc(filepath, swc):
    if swc.shape[1] > 7:
        swc = swc[:, :7]

    with open(filepath, 'w') as f:
        for i in range(swc.shape[0]):
            print('%d %d %.3f %.3f %.3f %.3f %d' %
                  tuple(swc[i, :].tolist()), file=f)

            
def crop(img, thr):
    """Crop a 3D block with value > thr"""
    ind = np.argwhere(img > thr)
    x = ind[:, 0]
    y = ind[:, 1]
    z = ind[:, 2]
    xmin = max(x.min() - 10, 0)
    xmax = min(x.max() + 10, img.shape[0])
    ymin = max(y.min() - 10, 1)
    ymax = min(y.max() + 10, img.shape[1])
    zmin = max(z.min() - 10, 2)
    zmax = min(z.max() + 10, img.shape[2])

    return img[xmin:xmax, ymin:ymax, zmin:zmax], np.array(
        [[xmin, xmax], [ymin, ymax], [zmin, zmax]])