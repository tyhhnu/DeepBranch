#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:00:51 2018

@author: ttt
"""

from io1 import loadmarker, loadimg, loadtiff3d, writetiff3d, savemarker
import numpy as np



def padding(image,margin):
    pimg = np.zeros((image.shape[0]+2*margin,
                     image.shape[1]+2*margin,
                     image.shape[2]+2*margin))
    pimg[margin:margin + image.shape[0], 
         margin:margin + image.shape[1], 
         margin:margin + image.shape[2]] = image
    return pimg 
def de_padding(pimg,margin):
    imgvox = np.zeros((pimg.shape[0]-2*margin,
                     pimg.shape[1]-2*margin,
                     pimg.shape[2]-2*margin))
    imgvox = pimg[margin:margin + imgvox.shape[0], 
         margin:margin + imgvox.shape[1], 
         margin:margin + imgvox.shape[2]]
    return imgvox
if __name__=='__main__':
    marker = loadmarker('/media/tyhhnu/tyh1/shuju/ms1_b.marker')
    marker = marker.astype(int)
    imgvox = loadimg('/media/tyhhnu/tyh1/shuju/ms1.tif')
    margin = 28
    R = 12
    marker = marker - 1
    marker[:,1] = imgvox.shape[1] - marker[:,1]
    marker = marker + margin
    pimg = padding(imgvox , margin)
    bimg = np.zeros(pimg.shape)
    for i in range(marker.shape[0]):
        blocks = pimg[marker[i,0]-R:marker[i,0]+R+1,
                      marker[i,1]-R:marker[i,1]+R+1,
                      marker[i,2]-4:marker[i,2]+4+1]
#        blocks = blocks>(pimg[marker[i,0],marker[i,1],marker[i,2]]-10)
        blocks = blocks>0
        bimg[marker[i,0]-R:marker[i,0]+R+1,
                      marker[i,1]-R:marker[i,1]+R+1,
                      marker[i,2]-4:marker[i,2]+4+1] = blocks
    image_t =   de_padding(bimg,margin)
    idx = np.argwhere(image_t)
    idx[:,1] = imgvox.shape[1] - idx[:,1]
#    savemarker('/media/ttt/tyh1/shuju/ms1_rb2.marker' , idx)       
    writetiff3d('/media/tyhhnu/tyh1/shuju/ms1_b2.tif', (image_t*1).astype('uint8'))