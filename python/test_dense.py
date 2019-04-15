#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:43:07 2018

@author: tyhhnu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:26:06 2018

@author: ttt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 13:03:04 2018

@author: ttt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:39:40 2018

@author: ttt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:56:47 2018

@author: ttt
"""

import numpy as np
import tensorflow as tf
import time
import DENSE3D4 as model
import globals3d as g_
import h5py
from io1 import loadimg,writetiff3d
#import os
#import hickle as hkl
#import os.path as osp
#from glob import glob
#import sklearn.metrics as metrics
#from input import Dataset



#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#sys.path.append(parentdir)


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/media/ttt/LANKEXIN/shuju/5_7.tif',
                           """h5file of trian data """
                           """and checkpoint.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('epoch', 50,
                            """Number of training epoch.""")
#tf.app.flags.DEFINE_string('val_dir', '/home/ttt/hdf/hdf5/train17.hdf5', 
#                            """h5file of val data""")
#'/media/tyhhnu/tyh1/gz/train_data/dense-model/model.cpkt-2970'
#'/media/tyhhnu/Elements/TanYinghui/model2.cpkt-8910'
#/home/ttt/model/model3.cpkt-5170
#/home/ttt/model/model3.cpkt-41000
#/media/ttt/Elements/TanYinghui/model4.cpkt-32000
#'/media/ttt/Elements/TanYinghui/model5/model5.cpkt-8900'
tf.app.flags.DEFINE_string('model', '/home/ttt/model/model3.cpkt-41000', 
                            """finetune with a model converted by tensorflow""")

np.set_printoptions(precision=3)


def train(trainpath,  ckptfile):
    immm = loadimg(trainpath)
    print ('testing() called')
    V = 64
    vz = 32
    margin = 32
    re_filed = 0
#    data_size = get_datasize(trainpath)
    
#    print ('training size:', data_size)


    with tf.Graph().as_default():
                 
        # placeholders for graph input
        view_ = tf.placeholder('float32', shape=(None, V, V, vz,1), name='im0')
#        y_ = tf.placeholder('int64', shape=(N`-one,V-16,V-16,V-16), name='y')
        keep_prob_ = tf.placeholder('float32')

        # graph outputs
        fc8 = model.inference_multiview(view_, g_.NUM_CLASSES, keep_prob_)
#        pr=tf.nn.softmax(fc8)
#        loss = model.loss(fc8, y_)
#        train_op = model.train(loss, global_step, data_size)
        prediction = model.classify(fc8)




        # must be after merge_all_summaries
#        validation_loss = tf.placeholder('float32', shape=(), name='validation_loss')
#        validation_summary = tf.summary.scalar('validation_loss', validation_loss)
#        validation_acc = tf.placeholder('float32', shape=(), name='validation_accuracy')
#        validation_acc_summary = tf.summary.scalar('validation_accuracy', validation_acc)

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1000)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        
        saver.restore(sess, ckptfile)
        print ('restore variables done')


        immm = padding(immm,margin)
#        zmmm = np.zeros(immm.shape)
        imum = get_cen(immm.shape,V,V,vz,margin,re_filed)
        
        for num in range(len(imum)):
                print(num)
                center_point = imum[num]
#                center_point[0] = center_point[0] +margin
#                center_point[1] = center_point[1] +margin
#
#                center_point[2] = center_point[2] +margi/home/ttt/model/model3.cpkt-41000n

                #print('center point:', center_point)
                image = immm[center_point[0]-V//2:center_point[0]+V//2,
                             center_point[1]-V//2:center_point[1]+V//2,
                             center_point[2]-vz//2:center_point[2]+vz//2]
                if image.shape !=(V,V,vz):
                    break
    
                image = np.expand_dims(image, axis = 0)
                image = np.expand_dims(image, axis = 4)
                        
#                        start_time = time.time()
                feed_dict = {view_: image,
                             keep_prob_: 0.5 }
        #                feed_dict_1 = {view_: batch_x,
        #                             keep_prob_: 0.5 }
        #                p_fc,p_softmax = sess.run(
        #                        [fc8,pr],
        #                        feed_dict=feed_dict_1)
        #                print(p_fc,p_softmax)
                        
                pred = sess.run(
                        prediction,
                        feed_dict=feed_dict)
#                pred = np.argmax(pred,-1)
                pred = pred[:,:,:,:,1]
                pred = np.array(pred)
                pred = np.squeeze(pred)
                bnn = np.argwhere(pred)
                pred = pred>0.5
                print(bnn.shape[0])
                immm[center_point[0]-V//2:center_point[0]+V//2,
                     center_point[1]-V//2:center_point[1]+V//2,
                     center_point[2]-vz//2:center_point[2]+vz//2] = pred
        immm = depadding(immm,margin)
        immm[immm>1] = 0
        immm = immm*255
        writetiff3d('/media/ttt/Elements/TanYinghui/TP/5_7.tif',immm.astype('uint8'))


def get_cen(shape,rx,ry,rz,margin,re_filed):
    x0_num = shape[0]
    y0_num = shape[1]
    z0_num = shape[2]
    x1_num = (x0_num // (rx-re_filed))  #12 
    y1_num = (y0_num // (ry-re_filed))   #7
    z1_num = (z0_num // (rz-re_filed))+1  #3
    cen_location = []
    for k in range(z1_num):
        for j in range(y1_num):
            for i in range(x1_num):
    #            cen_pixel[i, j, k] = orgimg[30+60*i, 30+60*j, 17+34*k]
                cen_location.append([int((rx-re_filed)/2+(rx-re_filed)*i), int((ry-re_filed)/2+(ry-re_filed)*j), int((rz-re_filed)/2+(rz-re_filed)*k)])
    return cen_location


def padding(image,margin):
    pimg = np.zeros((image.shape[0]+2*margin,
                     image.shape[1]+2*margin,
                     image.shape[2]+2*margin))
    pimg[margin:margin + image.shape[0], 
         margin:margin + image.shape[1], 
         margin:margin + image.shape[2]] = image
    return pimg  
def depadding(image,margin):
#    pimg = np.zeros((image.shape[0]-2*margin,
#                     image.shape[1]-2*margin,
#                     image.shape[2]-2*margin))
    pimg = image[margin:image.shape[0] - margin, 
         margin:image.shape[1]-margin, 
         margin:image.shape[2]-margin] 
    return pimg 

def main():
    st = time.time()
    
    print ('start loading data')

#    listfiles_train, labels_train = read_lists(g_.TRAIN_LOL)
#    listfiles_val, labels_val = read_lists(g_.VAL_LOL)
#    dataset_train = Dataset(listfiles_train, labels_train, subtract_mean=False, V=g_.NUM_VIEWS)
#    dataset_val = Dataset(listfiles_val, labels_val, subtract_mean=False, V=g_.NUM_VIEWS)
    trainpath = FLAGS.train_dir
#    valpath = FLAGS.val_dir
#    train_x,train_y = loaddata(trainpath)
    
#    val_x,val_y = loaddata(valpath)
    
    
#    train_y = np.squeeze(train_y[random_idx, :])
#    print(train_y.shape)

    print ('done loading data, time=', time.time() - st)

    train(trainpath, FLAGS.model)

#def loaddata(path):
#    imfile = h5py.File(path, 'a')
#    nimg = len([k for k in imfile.keys() if k != 'cache'])
#    trainidx = [i for i in range(nimg)]
##    trainidx = [i for i in trainidx if i not in testidx]
#
#    if nimg==0:
#        raise Exception('There are one or more test idx out of bound')
#
#    train_x = []
#    train_y = []
#
#    for i, idx in enumerate(trainidx):
#        print('== Collect patches from image %d/%d' %(i, len(trainidx)))
#        x, y, c = select_patches_from(imfile,idx)
#        train_x.append(x)
#        train_y.append(y)
#    imfile.close()
#    print('== All the patches collected')
#
#    train_x = np.concatenate(train_x, axis=0)
#    train_y = np.concatenate(train_y, axis=0)
#
##    print('nsample_each:%d\ttrain_x:%d' %
##                  (nsample_each, train_x.shape[0]))
#
#            # Shuffle the indices
#    random_idx = np.arange(train_y.shape[0])
#    np.random.shuffle(random_idx)
#    train_x = train_x[random_idx, :, :, :, :]
#    train_x = train_x.transpose((0,4,2,3,1))
#    
#    train_y = train_y[random_idx, :]
#    return train_x,train_y
def get_datasize(path):
    size = 0
    f=h5py.File(path, "r")
    img_names = [k for k in f['/'].keys() if k != 'cache']
    for i in range(len(img_names)):
        x = f[img_names[i]]['data']['x']
        n, nrotate, nscale, kernelsz, _, _ = x.shape
        size = size + n
    f.close
    return size
def select_patches_from(imfile,idx):
    
    img_names = [k for k in imfile['/'].keys() if k != 'cache']
    x = imfile[img_names[idx]]['data']['x']

#    if binary:
    y = imfile[img_names[idx]]['data']['label']
#    else:
#        y = imfile[img_names[idx]]['data']['dist']
#    c = imfile[img_names[idx]]['data']['c']  # Total number of locations
#    n, nrotate, nscale, kernelsz, _, _ = x.shape
#
#    # Sample half with zeros
#    y_np = np.squeeze(np.array(y))
#    nsample_each = len(y_np)
#    # Convert y to numpy array
#    zero_idx = np.argwhere(y_np == 0)
#    nonzero_idx = np.argwhere(y_np > 0)
#    np.random.shuffle(zero_idx)
#    np.random.shuffle(nonzero_idx)
#
##    if binary:
#    nsample_each_cls = int(np.floor(nsample_each / 2))
#    sample_idx = np.concatenate(
#            (zero_idx[:nsample_each_cls if zero_idx.size > nsample_each_cls
#                          else zero_idx.size],
#                 nonzero_idx[:nsample_each_cls if zero_idx.size >
#                             nsample_each_cls else zero_idx.size]))
##    else:
##        nsample_nonzero = int(np.floor(nsample_each * 1 / 4))
##        nsample_zero = int(np.floor(nsample_each * 3 / 4))
##        sample_idx = np.concatenate(
##            (zero_idx[:nsample_zero
##                      if zero_idx.size > nsample_zero else zero_idx.size],
##                 nonzero_idx[:nsample_nonzero if zero_idx.size >
##                             nsample_nonzero else zero_idx.size]))
#
#    # np.random.shuffle(sample_idx)
#    sample_idx = np.squeeze(sample_idx)
#    nsample_each = sample_idx.size
#    print('sample_idx', sample_idx.shape)
#
#    # Determine the depth of the block
#    #    if self._patch_type == '25d':
#    #        block_depth = 3
#    #    elif self._patch_type == 'nov':
#    #        block_depth = 9
#    #    elif self._patch_type == '3d':
#    #        block_depth = 3
#    
#    block_depth = g_.NUM_VIEWS
#
#    # Claim memory for the patches
#    patches = np.zeros(
#        (nsample_each, 1, kernelsz, kernelsz, block_depth))
#    groundtruth = np.zeros((nsample_each, 1))
#    coords = np.zeros((nsample_each, 3))
#
#    # for i, idx in enumerate(tqdm(sample_idx)):
#    sample_idx = np.sort(sample_idx)
#    patch_idx = np.arange(len(sample_idx))
#    patches[patch_idx, :, :, :, :] = x[sample_idx, 0, :, :, :, :]
#    groundtruth[patch_idx, :] = y[sample_idx, :]
#    coords[patch_idx] = c[sample_idx, :]
#
#    print('Nonzeros: %d/%d' % (np.count_nonzero(groundtruth), groundtruth.size))
#
    return x, y, 


#def read_lists(list_of_lists_file):
#    listfile_labels = np.loadtxt(list_of_lists_file, dtype=str).tolist()
#    listfiles, labels  = zip(*[(l[0], int(l[1])) for l in listfile_labels])
#    return listfiles, labels
    


if __name__ == '__main__':
    main()


