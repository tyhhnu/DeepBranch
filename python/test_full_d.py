#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 14:03:42 2018

@author: tyhhnu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:20:16 2018

@author: ttt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:40:33 2018

@author: tyhhnu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:13:25 2018

@author: ttt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:34:53 2018

@author: ttt
"""
import scipy.ndimage
import numpy as np
import os,sys,inspect
import tensorflow as tf
import time
from datetime import datetime
import os.path as osp
import math
import globals3d as g_
import h5py
from io1 import loadimg, writetiff3d, savemarker, loadswc,loadmarker , savemarker1
import model2 as model
from scipy.interpolate import RegularGridInterpolator

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('testdir',  '/media/tyhhnu/LHQ/Untitled Folder 2/m1.tif',
                           """h5file of test data """
                           """and checkpoint.""")
#tf.app.flags.DEFINE_string('swc',  '/media/tyhhnu/LHQ/Untitled Folder 2/m4_f.swc',
#                           """h5file of test data """)
tf.app.flags.DEFINE_string('marker',  '/media/tyhhnu/LHQ/Untitled Folder 2/m1_27_0.5.marker',
                           """h5file of test data """)
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('fweights', '/media/tyhhnu/0005049A000CA49F/tyh/tyh/model3d_1/model4.cpkt-2600',
                            """finetune with a pretrained model""")
#tf.app.flags.DEFINE_string('weights', '/media/tyhhnu/0005049A000CA49F/tyh/tyh/model3d_1/model4.cpkt',
#                            """finetune with a pretrained model""")

np.set_printoptions(precision=3)

#imgvox = loadimg(FLAGS.testdir)
#margin = 28
#pimg = np.zeros((imgvox.shape[0] + 2 * margin,
#                 imgvox.shape[1] + 2 * margin,
#                 imgvox.shape[2] + 2 * margin))
#pimgs = np.zeros((imgvox.shape[0] + 2 * margin,
#                 imgvox.shape[1] + 2 * margin,
#                 imgvox.shape[2] + 2 * margin))
#pimg[margin:margin + imgvox.shape[0], margin:margin + imgvox.
#             shape[1], margin:margin + imgvox.shape[2]] = imgvox
#idx = np.argwhere(pimg)
#print(idx.shape[0])





def tdiam(point, binterp, imgvox):
    '''以下是输入'''
    maxR = 10
    threod = 20
    '''以上是输入'''
    
    ''' 判断是否越界'''
    
    
    
    
    '''越界则报错'''
    pointCell = []
    sl = np.zeros([26,4])
    sl = [ [0, 0, 1, 1], [0, 0, -1, 1], [0, 1, 0, 1], [0, -1, 0, 1],
            [0, 0.7071, 0.7071, 1], [0, -0.7071, -0.7071, 1], [0, 0.7071, -0.7071, 1], [0, -0.7071, 0.7071, 1],    
            [1, 0, 0, 1], [-1, 0, 0, 1], [0.7071, 0, 0.7071, 1], [-0.7071, 0, -0.7071, 1], [0.7071, 0,  -0.7071, 1],
            [-0.7071, 0, 0.7071, 1],     
            [0.7071, 0.7071, 0, 1], [-0.7071, -0.7071, 0, 1], [0.7071, -0.7071, 0, 1], [-0.7071, 0.7071, 0, 1],     
            [0.5774, 0.5774, 0.5774, 1], [-0.5774, -0.5774, -0.5774, 1], [0.5774, 0.5774, -0.5774, 1], [-0.5774, -0.5774, 0.5774, 1],
            [0.5774, -0.5774, 0.5774, 1], [-0.5774, 0.5774, -0.5774, 1], [0.5774, -0.5774, -0.5774, 1], [-0.5774, 0.5774, 0.5774, 1]     
             ]
    sl = np.array(sl)
    sl = sl[:,0:3]
    
    for i in range(maxR):
        ps = point + sl* (i+1)
        pointCell.append(ps)
        
    
    
    
    
    
    
    
    ''' a.size : 26 x maxR'''
    rays = np.stack([binterp(p) for p in pointCell],
                axis=-1)
    rays[rays<threod] = 0
    
    xLenth = []
    for j in range(rays.shape[0]):
        rayR = rays[j,:]
        lll = np.argwhere(rayR)
        xLenth.append(lll.shape[0])
    xLenth = ( np.array(xLenth[::2]) + np.array(xLenth[1::2]) )
    xLenth.sort(axis=0)
    diamter = xLenth[3]
    return diamter

def padding(image,margin):
    pimg = np.zeros((image.shape[0]+2*margin,
                     image.shape[1]+2*margin,
                     image.shape[2]+2*margin))
    pimg[margin:margin + image.shape[0], 
         margin:margin + image.shape[1], 
         margin:margin + image.shape[2]] = image
    return pimg
def get_thre(imgvox):
    thre = 10
    while thre<=255:
        
        idx = np.argwhere(imgvox>thre)
        if idx.shape[0]<100000:
            break
        else:
            thre = thre + 2
    print(thre)
    return thre
def get_threb(blocks):
    hist = np.histogram(blocks,20)
    summ = 0
    hhh = hist[0]
    sizea = hhh[1:]
    sizea = 0.5*(np.sum(sizea))
    jjj = hist[1]
    for j in range(hhh.shape[0]-1):
        summ = summ+hhh[hhh.shape[0]-1-j]
        if(summ>sizea):
            threod = jjj[hhh.shape[0]-j]
            break
    return threod

def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y)"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9

def transitions(neighbours):
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)
#将白色静脉区域细化成骨架结构　　
def Refine(image):
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions 
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6　　　The guarantee is not an isolated point and an endpoint or an internal point
                    transitions(n) == 1 and    # Condition 2: S(P1)=1  　（０，１）The number of rotation of the structure is 1, and the boundary point can be determined by adding other conditions
                    P2 * P4 * P6 == 0  and    # Condition 3  Remove the southeast boundary point
                    P4 * P6 * P8 == 0):         # Condition 4
                    changing1.append((x,y))
        for x, y in changing1: 
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and        # Condition 0
                    2 <= sum(n) <= 6  and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    P2 * P4 * P8 == 0 and       # Condition 3　　　remove the northwest border point
                    P2 * P6 * P8 == 0):            # Condition 4
                    changing2.append((x,y))    
        for x, y in changing2: 
            Image_Thinned[x][y] = 0
    return Image_Thinned
def getcand(imgvox,imgvox1,margin):
    imgvox=imgvox>0
    imgvox1 = imgvox*imgvox1
    idx = np.argwhere(imgvox1>0)
    return idx+margin
def mip(block_3d):
        mip=np.zeros([block_3d.shape[0],block_3d.shape[1],3],dtype=float)
        mip[:,:,0] = np.max(block_3d,0)
        mip[:,:,1] = np.max(block_3d,1)
        mip[:,:,2] = np.max(block_3d,2)
        return mip
def tiqu(pimg,
         batch_candidates,
         nrotate=1,
         patch_type = 'mip',
         radii = [10,15,20]):
    
    
        K = min(radii)
        if patch_type == 'mip':
            block_depth = 3
        elif patch_type == 'nov':
            block_depth = 9
        elif patch_type == '3d':
            block_depth = 2 * K + 1
        blocks = np.zeros((batch_candidates.shape[0], max(nrotate, 1), max(len(radii),1), 2 * K + 1, 2 * K + 1, block_depth))
        if patch_type == 'mip':
                for i in range(batch_candidates.shape[0]):
#                    label[i,0] = labelmap[batch_candidates[i,0], batch_candidates[i,1], batch_candidates[i,2]]
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
                    for j in range(len(radii)):
            #                print(batch_x[i,:])
                            R = radii[j]
#                            print(K)                            
                            blocks0 = pimg[batch_candidates[i,0]-R : batch_candidates[i,0]+R+1,
                                  batch_candidates[i,1]-R : batch_candidates[i,1]+R+1, 
                                  batch_candidates[i,2]-R : batch_candidates[i,2]+R+1]
#                            print(batch_candidates[i,0], batch_candidates[i,1], batch_candidates[i,2])
#                            print(blocks0.shape)
                            blocks0 = scipy.ndimage.zoom(blocks0, (2*radii[0]+1)/(2*radii[j]+1), order=3)
#                            print(blocks0.shape)
#                            blocks0 = mip(blocks0)
                            blocks[i,0,j,:,:,:] = blocks0
        return blocks

def test(pimg,ckptfile,batch_size,idx):
    
    print ('test() called')
    is_testing = bool(ckptfile)
    batch_size = batch_size
    data_size = idx.shape[0]
    predictions = np.zeros((data_size,2))
    print ('training size:', data_size)


    with tf.Graph().as_default():
         
        # placeholders for graph input
        view_ = tf.placeholder('float32', shape=(None, 3, 3, 21, 21), name='im0')
        keep_prob_ = tf.placeholder('float32')

        # graph outputs
        fc8 = model.inference_multiview(view_, g_.NUM_CLASSES, keep_prob_)
#        pr=tf.nn.softmax(fc8)
        prediction = model.classify(fc8)


        # must be after merge_all_summaries
#        validation_loss = tf.placeholder('float32', shape=(), name='validation_loss')
#        validation_summary = tf.summary.scalar('validation_loss', validation_loss)
#        validation_acc = tf.placeholder('float32', shape=(), name='validation_accuracy')
#        validation_acc_summary = tf.summary.scalar('validation_accuracy', validation_acc)

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1000)

        init_op = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        
        if is_testing:
            # load checkpoint file
            saver.restore(sess, ckptfile)
            print ('restore variables done')
#        elif caffemodel:
#            # load caffemodel generated with caffe-tensorflow
#            sess.run(init_op)
#            model.load_alexnet_to_mvcnn(sess, caffemodel)
#            print ('loaded pretrained caffemodel:', caffemodel)
        else:
            # from scratch
            sess.run(init_op)
            print ('init_op done')


        
        
        batch_start = 0
        batch_end = 0
        while batch_start<data_size:
            start = time.time()
            batch_end = batch_start + batch_size
            batch_end = batch_end if batch_end <= data_size else data_size
            batch_candidates = idx[batch_start:batch_end,:]           
#            nsample_each = batch_end - batch_start
            batch_x = tiqu(pimg,
                            batch_candidates,
                            nrotate=1,
                            patch_type = 'mip',
                            radii = [10,15,20]
                    )
            end1 = time.time()
            print(end1-start)
            batch_x = np.squeeze(batch_x)
            batch_x = batch_x.transpose((0,4,1,2,3))
            feed_dict = {view_: batch_x,
                         keep_prob_: 0.5 }
                        
            pred = sess.run(prediction,feed_dict=feed_dict)
            predictions[batch_start:batch_end,:] = pred
            batch_start = batch_end
            print(batch_end)
            end2 = time.time()
            print(end2-end1)
    return predictions
    
#'/media/ttt/Elements/TanYinghui/lung1/model2d/model2.cpkt-9400' 
#    '/media/ttt/Elements/TanYinghui/model2d2/model2.cpkt-11200'
if __name__ == '__main__':
    margin = 28
    ckptfile = '/media/ttt/Elements/TanYinghui/model2d2/model2.cpkt-11200'
    batch_size = 500
    fild = ['/media/ttt/Elements/TanYinghui/TP/5_1.tif',
            '/media/ttt/Elements/TanYinghui/TP/5_4.tif',
            '/media/ttt/Elements/TanYinghui/TP/5_7.tif']
    filr = ['/media/ttt/LANKEXIN/shuju/5_1.tif',
            '/media/ttt/LANKEXIN/shuju/5_4.tif',
            '/media/ttt/LANKEXIN/shuju/5_7.tif']
    film = ['/media/ttt/LANKEXIN/shuju/5_1.tif.marker',
            '/media/ttt/LANKEXIN/shuju/5_4.tif.marker',
            '/media/ttt/LANKEXIN/shuju/5_7.tif.marker']
    for i in range(len(filr)):
        
        imgvox = loadimg(filr[i])
        imgvox1 = loadimg(fild[i])
        imgvox[imgvox<5] = 0
        can_idx = getcand(imgvox,imgvox1,margin)
        can_idx = can_idx.astype(int)
        pimg = padding(imgvox,margin)
        pp = test(pimg,ckptfile,batch_size,can_idx)
        print('test over')
        nm = np.argwhere(pp[:,1]>0.5)
        zb = can_idx[nm,:]
        zb = np.squeeze(zb)
        
        
        
        zb = zb-28
        zb[:,1]=imgvox.shape[1]-zb[:,1]-1
        zb =zb+1
#        zb1 = np.hstack((zb,pp[nm,1]))
#        print(zb1.shape)
        savemarker(film[i],zb)
#        savemarker1(film[i] +'qqq.marker',zb1)
#    img = img*255
#    marker = np.argwhere(img)
#    marker = marker - 28
#
#
#
#    marker[:,1] = imgvox.shape[1] -  marker[:,1]
#    
#    savemarker('/media/ttt/E81AA8261AA7EFAC/0000/ms3_b.marker',marker)
#    
##    writetiff3d('/media/ttt/E81AA8261AA7EFAC/0000/shuju/testm7.tif',img.astype('uint8'))





    



 
