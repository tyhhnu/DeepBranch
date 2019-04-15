#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 10:10:52 2018

@author: ttt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:56:28 2018

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
from augu import rotateit, scaleit, translateit



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
            block_depth = K 

        data_grp.create_dataset('x', (num_s, max(nrotate, 1), max(len(radii),1),  2 * K, 2 * K, block_depth))
        data_grp.create_dataset('label', (num_s, max(nrotate, 1), max(len(radii),1),  2 * K, 2 * K, block_depth))
        data_grp.create_dataset('c', (num_s, 3))
        h5.close()  # Close for safe write
        
        
        batch_start = 0
        while True:
            batch_end = batch_start + extract_batch_size
            batch_end = batch_end if batch_end <= num_s else num_s
            start = time.time()
            batch_candidates = idx[batch_start:batch_end, :]
            blocks = np.zeros((batch_candidates.shape[0], max(nrotate, 1), max(len(radii),1), 2 * K , 2 * K , block_depth))
            label = np.zeros((batch_candidates.shape[0], max(nrotate, 1), max(len(radii),1), 2 * K , 2 * K , block_depth))

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
#                    print(i)
#                    label[i,0] = labelmap[batch_candidates[i,0], batch_candidates[i,1], batch_candidates[i,2]]
                    for j in range(len(radii)):
            #                print(batch_x[i,:])
                            R = radii[j]
                            RZ = int(radii[j]/2)
#                            print(K)                            
                            blocks0 = pimg[math.floor(batch_candidates[i,0])-R : math.floor(batch_candidates[i,0])+R,
                                  math.floor(batch_candidates[i,1])-R : math.floor(batch_candidates[i,1])+R, 
                                  math.floor(batch_candidates[i,2])-RZ : math.floor(batch_candidates[i,2])+RZ]
                            label0 = labelmap[math.floor(batch_candidates[i,0])-R : math.floor(batch_candidates[i,0])+R,
                                  math.floor(batch_candidates[i,1])-R : math.floor(batch_candidates[i,1])+R, 
                                  math.floor(batch_candidates[i,2])-RZ : math.floor(batch_candidates[i,2])+RZ]
#                            print(batch_candidates[i,0], batch_candidates[i,1], batch_candidates[i,2])
#                            print(blocks0.shape)
                            blocks0 = scipy.ndimage.zoom(blocks0, (2*radii[0]+1)/(2*radii[j]+1), order=3)
                            label0 = scipy.ndimage.zoom(label0, (2*radii[0]+1)/(2*radii[j]+1), order=3)
#                            print(blocks0.shape)
#                            blocks0 = mip(blocks0)
                            blocks0 = augumentor(blocks0)
                            label0 = augumentor(label0)
                                
                                
                            blocks[i,:,j,:,:,:] = blocks0
                            label[i,:,j,:,:,:] = label0
            h5 = h5py.File(h5file, 'a')
            print('开始保存')
            h5[img_name]['data']['x'][batch_start:
                                            batch_end, :, :, :, :, :] = blocks
            h5[img_name]['data']['label'][batch_start:
                                                batch_end,:, :, :, :, :] = label
            h5[img_name]['data']['c'][batch_start:batch_end,:] = batch_candidates
                    #        task_queue.task_done()
            h5.close()
            print('保存完毕')
            end = time.time()
            print(end - start)
            print('save from %d to %d' % (batch_start, batch_end))
            batch_start += extract_batch_size

            if batch_start >= num_s:
                break

def augumentor(vox):
    rri = []
    block_afau = np.zeros((32,vox.shape[0],vox.shape[1],vox.shape[2]))
    rr0 = vox
    rri.append(rr0)
    rr1 = rotateit(vox, 45, isseg=False)
    rri.append(rr1)
    rr2 = rotateit(vox, 90, isseg=False)
    rri.append(rr2)
    rr3 = rotateit(vox, 135, isseg=False)
    rri.append(rr3)
    rr4 = rotateit(vox, 180, isseg=False)
    rri.append(rr4)
    rr5 = rotateit(vox, 225, isseg=False)
    rri.append(rr5)
    rr6 = rotateit(vox, 270, isseg=False)
    rri.append(rr6)
    rr7 = rotateit(vox, 315, isseg=False)
    rri.append(rr7)
    for ri in range(len(rri)):
        pp = rri[ri]
        rt1 = translateit(pp, [0,5,0], isseg=False)
        rt2 = translateit(pp, [5,0,0], isseg=False)
        block_afau[0+ri*4,:,:,:] = scaleit(rt1, 0.75, isseg=False)
        block_afau[1+ri*4,:,:,:]  = scaleit(rt1, 1.25, isseg=False)
        block_afau[2+ri*4,:,:,:]  = scaleit(rt2, 0.75, isseg=False)
        block_afau[3+ri*4,:,:,:]  = scaleit(rt2, 1.25, isseg=False)
    return block_afau
def padding(image,margin):
    pimg = np.zeros((image.shape[0]+2*margin,
                     image.shape[1]+2*margin,
                     image.shape[2]+2*margin))
    pimg[margin:margin + image.shape[0], 
         margin:margin + image.shape[1], 
         margin:margin + image.shape[2]] = image
    return pimg     

def getcand(imgvox,marker,margin,n_each):
#    ones = np.argwhere(labelmap>0)
    zeros = np.argwhere(imgvox>0)
    np.random.shuffle(zeros)
    idx = marker
    np.random.shuffle(idx)
    idx1 = zeros[0:idx.shape[0]*20,:]
#    idx = idx[0:n_each,:]
#    idx = np.vstack((idx,idx1))
    return idx+margin

def mip(block_3d):
        mip=np.zeros([block_3d.shape[0],block_3d.shape[1],3],dtype=float)
        for i in range(block_3d.shape[0]):
            for j in range(block_3d.shape[1]):
                list1=block_3d[i,j,:]
                r=np.float(np.max(list1))
                mip[i,j,0]=r
        for i in range(block_3d.shape[1]):
            for j in range(block_3d.shape[2]):
                list1=block_3d[:,i,j]
                r=np.float(np.max(list1))
                mip[i,j,1]=r
        for i in range(block_3d.shape[0]):
            for j in range(block_3d.shape[2]):
                list1=block_3d[i,:,j]
                r=np.float(np.max(list1))
                mip[i,j,2]=r
        return mip

if __name__ == '__main__':
    margin = 28
    imgvox = loadimg('/media/tyhhnu/tyh1/shuju/ms1.tif')
    labelmap = loadimg('/media/tyhhnu/tyh1/shuju/ms1_b2.tif')
    marker = loadmarker('/media/tyhhnu/tyh1/shuju/ms1_b.marker')
    marker = marker.astype(int)
    marker = marker - 1
    marker[:,1] = imgvox.shape[1] - marker[:,1]
    
    labelmap = labelmap>0
    h5file='/media/tyhhnu/tyh1/shuju/hdf5/train_dense11p2.hdf5'
    img_name = 'ms1tif'
    n_each = 1000
    idx = getcand(imgvox,marker,margin,n_each)
    labelmap = padding(labelmap,margin,)
    pimg = padding(imgvox,margin)
    tiqu(pimg,
         labelmap,
         idx,
         img_name,
         h5file,
         nrotate=32,
         patch_type = '3d',
         extract_batch_size = 50,
         radii = [32])












