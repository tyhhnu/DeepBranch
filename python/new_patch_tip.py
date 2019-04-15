#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:23:56 2018

@author: ttt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 11:01:25 2018

@author: ttt
"""
import numpy as np
import math
import h5py
from io1 import loadimg, writetiff3d, savemarker, loadswc,loadmarker
import scipy.ndimage
import time



def tiqu(pimg,
         labelmap,
         idx,
         img_name,
         h5file,
         nrotate=1,
         patch_type = 'mip',
         extract_batch_size = 2000,
         radii = [10,15,20]):
    
    
        K = min(radii)
        num_s = idx.shape[0]
        print('Creating dataset')
        print('num_s: %d' % num_s)
        h5 = h5py.File(h5file, 'a')
        img_grp = h5.create_group(img_name)
        data_grp = img_grp.create_group('data')
        # Determine the depth of the block
        if patch_type == 'mip':
            block_depth = 3
        elif patch_type == 'nov':
            block_depth = 9
        elif patch_type == '3d':
            block_depth = 2 * K + 1

        data_grp.create_dataset('x', (num_s, max(nrotate, 1), max(len(radii),1),  2 * K + 1, 2 * K + 1, block_depth))
        data_grp.create_dataset('label', (num_s, 1))
        data_grp.create_dataset('c', (num_s, 3))
        h5.close()  # Close for safe write
        
        
        batch_start = 0
        while True:
            batch_end = batch_start + extract_batch_size
            batch_end = batch_end if batch_end <= num_s else num_s
            start = time.time()
            batch_candidates = idx[batch_start:batch_end, :]
            blocks = np.zeros((batch_candidates.shape[0], max(nrotate, 1), max(len(radii),1), 2 * K + 1, 2 * K + 1, block_depth))
            label = np.zeros((batch_candidates.shape[0],1))

            if patch_type == 'mip':
                for i in range(batch_candidates.shape[0]):
                    label[i,0] = labelmap[batch_candidates[i,0], batch_candidates[i,1], batch_candidates[i,2]]
                    for j in range(len(radii)):
            #                print(batch_x[i,:])
                            R = radii[j]
#                            print(K)                            
                            blocks0 = pimg[math.floor(batch_candidates[i,0])-R : math.floor(batch_candidates[i,0])+R+1,
                                  math.floor(batch_candidates[i,1])-R : math.floor(batch_candidates[i,1])+R+1, 
                                  math.floor(batch_candidates[i,2])-R : math.floor(batch_candidates[i,2])+R+1]
#                            print(batch_candidates[i,0], batch_candidates[i,1], batch_candidates[i,2])
#                            print(blocks0.shape)
                            blocks0 = scipy.ndimage.zoom(blocks0, (2*radii[0]+1)/(2*radii[j]+1), order=3)
#                            print(blocks0.shape)
                            blocks0 = mip(blocks0)
                            blocks[i,0,j,:,:,:] = blocks0
                            
#                            K = radii[1]
#                            blocks1 = pimg[math.floor(batch_x[i,0])-K-5 : math.floor(batch_x[i,0])+K+5,
#                                  math.floor(batch_x[i,1])-K-5 : math.floor(batch_x[i,1])+K+5, 
#                                  math.floor(batch_x[i,2])-K-5 : math.floor(batch_x[i,2])+K+5 ]
#                            blocks1 = scipy.ndimage.zoom(blocks1, 0.66, order=3)
#                            blocks1 = mip(blocks1)
#                            blocks[i,0,1,:,:,:] = blocks1
#                            
#                            
#                            K = radii[2]
#                            blocks2 = pimg[math.floor(batch_x[i,0])-K-10 : math.floor(batch_x[i,0])+K+10,
#                                  math.floor(batch_x[i,1])-K-10 : math.floor(batch_x[i,1])+K+10, 
#                                  math.floor(batch_x[i,2])-K-10 : math.floor(batch_x[i,2])+K+10 ]
#                            blocks2 = scipy.ndimage.zoom(blocks2, 0.5, order=3)
#                            blocks2 = mip(blocks2)
#                            blocks[i,0,2,:,:,:] = blocks2
            elif patch_type == 'nov':
                print('还没开发呢')
            elif patch_type == '3d':
                 for i in range(batch_candidates.shape[0]):
                    label[i,0] = labelmap[batch_candidates[i,0], batch_candidates[i,1], batch_candidates[i,2]]
                    for j in range(len(radii)):
            #                print(batch_x[i,:])
                            R = radii[j]
#                            print(K)                            
                            blocks0 = pimg[math.floor(batch_candidates[i,0])-R : math.floor(batch_candidates[i,0])+R+1,
                                  math.floor(batch_candidates[i,1])-R : math.floor(batch_candidates[i,1])+R+1, 
                                  math.floor(batch_candidates[i,2])-R : math.floor(batch_candidates[i,2])+R+1]
#                            print(batch_candidates[i,0], batch_candidates[i,1], batch_candidates[i,2])
#                            print(blocks0.shape)
                            blocks0 = scipy.ndimage.zoom(blocks0, (2*radii[0]+1)/(2*radii[j]+1), order=3)
#                            print(blocks0.shape)
#                            blocks0 = mip(blocks0)
                            blocks[i,0,j,:,:,:] = blocks0
            h5 = h5py.File(h5file, 'a')
            h5[img_name]['data']['x'][batch_start:
                                            batch_end, :, :, :, :, :] = blocks
            h5[img_name]['data']['label'][batch_start:
                                                batch_end, :] = label
            h5[img_name]['data']['c'][batch_start:batch_end,:] = batch_candidates
                    #        task_queue.task_done()
            h5.close()
            end = time.time()
            print(end - start)
            print('save from %d to %d' % (batch_start, batch_end))
            batch_start += extract_batch_size

            if batch_start >= num_s:
                break
def get_thre(blocks):
    hist = np.histogram(blocks,20)
    summ = 0
    hhh = hist[0]
    sizea = hhh[1:]
    sizea = 0.50*(np.sum(sizea))
    jjj = hist[1]
    for j in range(hhh.shape[0]-1):
        summ = summ+hhh[hhh.shape[0]-1-j]
        if(summ>sizea):
            threod = jjj[hhh.shape[0]-j]
            break
    return threod




def padding(image,margin):
    pimg = np.zeros((image.shape[0]+2*margin,
                     image.shape[1]+2*margin,
                     image.shape[2]+2*margin))
    pimg[margin:margin + image.shape[0], 
         margin:margin + image.shape[1], 
         margin:margin + image.shape[2]] = image
    return pimg     

def getcand(imgvox,labelmap,swc,margin):
    ones = np.argwhere(labelmap>0)
    z_s = 3*ones.shape[0] if swc.shape[0]>3*ones.shape[0] else swc.shape[0]
    zeros = np.argwhere(imgvox>0)
    np.random.shuffle(zeros)
    np.random.shuffle(swc)
    idx = np.vstack((ones,zeros[0:ones.shape[0],:],swc[0:z_s,:]))
    np.random.shuffle(idx)
    return idx+margin

def mip(block_3d):
        mip=np.zeros([block_3d.shape[0],block_3d.shape[1],3],dtype=float)
        mip[:,:,0] = np.max(block_3d,0)
        mip[:,:,1] = np.max(block_3d,1)
        mip[:,:,2] = np.max(block_3d,2)
        return mip

if __name__ == '__main__':
    imgvox = loadimg('/media/ttt/E81AA8261AA7EFAC/xuexi/3Dyuantu/OP_5_stack.tif')
    labelmap = np.zeros(imgvox.shape)
    
    smarker = loadmarker('/media/ttt/E81AA8261AA7EFAC/xuexi/3Dyuantu/OP5_b.marker')
    smarker = smarker-1
    smarker = smarker[:,0:3]
#    smarker[:,1]=imgvox.shape[1]-smarker[:,1]-1
    marker = np.empty((0,3))
    for i in range(smarker.shape[0]):
            x,y,z = smarker[i,:]
            blocks = imgvox[math.floor(x)-1:math.floor(x)+2,math.floor(y)-1:math.floor(y)+2,math.floor(z)-1:math.floor(z)+2]
            corrr = np.argwhere(blocks)
            print(corrr.shape)
            print(smarker[i,:])
            corrr[:,0] = corrr[:,0]+math.floor(x)
            corrr[:,1] = corrr[:,1]+math.floor(y)
            corrr[:,2] = corrr[:,2]+math.floor(z)
            marker=np.vstack((corrr,marker))
    smarker = np.array(marker)
    for i in range(smarker.shape[0]):
            labelmap[math.floor(int(smarker[i,0])),math.floor(int(smarker[i,1])),math.floor(int(smarker[i,2]))] = 1
#        print(int(smarker[i,0]),int(smarker[i,1]),int(smarker[i,2]))
    labelmap = labelmap>0
    swc = loadswc('/media/ttt/E81AA8261AA7EFAC/xuexi/3Dyuantu/OP_5_stack.tif.swc')
    swc = swc[:,2:5]
    swc = swc -1
#    swc[:,1]=imgvox.shape[1]-swc[:,1]-1
    swc = swc.astype('int')
#    labelmap = loadimg('/media/ttt/LHQ/fruitfly_1_z_lab.tif')
    h5file='/media/ttt/tyh1/gz/train_data/train2.hdf5'
    img_name = 'OP5.tif'
    margin = 28
    idx = getcand(imgvox,labelmap,swc,margin)
    labelmap = padding(labelmap,margin)
    pimg = padding(imgvox,margin)
    tiqu(pimg,
         labelmap,
         idx,
         img_name,
         h5file,
         nrotate=1,
         patch_type = 'mip',
         extract_batch_size = 2000,
         radii = [10,15,20])












