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
from datetime import datetime
import model2 as model
import globals as g_
import h5py
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
tf.app.flags.DEFINE_string('train_dir', '/media/ttt/Elements/TanYinghui/lung1/train1.hdf5',
                           """h5file of trian data """
                           """and checkpoint.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('epoch', 300,
                            """Number of training epoch.""")
#tf.app.flags.DEFINE_string('val_dir', '/home/ttt/hdf/hdf5/train17.hdf5', 
#                            """h5file of val data""")
tf.app.flags.DEFINE_string('model', '', 
                            """finetune with a model converted by tensorflow""")

np.set_printoptions(precision=3)


def train(trainpath,  ckptfile=''):
    print ('train() called')
    is_finetune = bool(ckptfile)
    batch_size = 200
    data_size = get_datasize(trainpath)
    print ('training size:', data_size)


    with tf.Graph().as_default():
        startstep = 0 if not is_finetune else int(ckptfile.split('-')[-1])
        global_step = tf.Variable(startstep, trainable=False)
         
        # placeholders for graph input
        view_ = tf.placeholder('float32', shape=(None, 3, 3, 21, 21), name='im0')
        y_ = tf.placeholder('int64', shape=(None), name='y')
        keep_prob_ = tf.placeholder('float32')

        # graph outputs
        fc8 = model.inference_multiview(view_, g_.NUM_CLASSES, keep_prob_)
#        pr=tf.nn.softmax(fc8)
        loss = model.loss(fc8, y_)
        train_op = model.train(loss, global_step, data_size)
        prediction = model.classify(fc8)

        # build the summary operation based on the F colection of Summaries
        summary_op = tf.summary.merge_all()


        # must be after merge_all_summaries
#        validation_loss = tf.placeholder('float32', shape=(), name='validation_loss')
#        validation_summary = tf.summary.scalar('validation_loss', validation_loss)
#        validation_acc = tf.placeholder('float32', shape=(), name='validation_accuracy')
#        validation_acc_summary = tf.summary.scalar('validation_accuracy', validation_acc)

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1000)

        init_op = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        
        if is_finetune:
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

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                               graph=sess.graph) 

        step = startstep
        for epoch in range(FLAGS.epoch):
            print ('epoch:', epoch)
            imfile = h5py.File(FLAGS.train_dir, 'a')
            
            nimg = len([k for k in imfile.keys() if k != 'cache'])
            trainidx = [i for i in range(nimg)]
            for i, idx in enumerate(trainidx):
                print('== <<<<trainning on  image %d/%d' %(i+1, len(trainidx)))
                x, y, c = select_patches_from(imfile,idx)
                num, nrotate, nscale, kernelsx, kernelsy, n_views = x.shape
#                num =len(x)
                print('the number of the image is %d' % num)
            
                batch_start = 0 
                batch_end = 0 
                while batch_start < num:
                    step += 1
                    batch_end = batch_start + batch_size 
                    if batch_end <= num :
                        
                        nsample_each = batch_end - batch_start
                        batch_x = np.zeros((nsample_each, nscale, kernelsx, kernelsy, n_views))
#                        xuhao = np.arange(batch_x.shape[0])
#                        np.random.shuffle(xuhao)
                        batch_y = np.zeros((nsample_each, 1))
                        print(batch_y.shape)
    #                    coords = np.zeros((nsample_each, 3))             
                        batch_x[:, :, :, :, :] = x[batch_start:batch_end,0,:,:,:,:]
                        batch_x = batch_x.transpose((0,4,1,2,3))
                        batch_y = np.squeeze(y[batch_start:batch_end,:])
                        print(batch_y.shape)
                        batch_y = batch_y.astype('int64')
#                        batch_x = batch_x[xuhao,:,:,:,:]
#                        batch_y = batch_y[xuhao]
                        print(batch_y)
                        
                        start_time = time.time()
                        feed_dict = {view_: batch_x,
                                     y_ : batch_y,
                                     keep_prob_: 0.5 }
        #                feed_dict_1 = {view_: batch_x,
        #                             keep_prob_: 0.5 }
        #                p_fc,p_softmax = sess.run(
        #                        [fc8,pr],
        #                        feed_dict=feed_dict_1)
        #                print(p_fc,p_softmax)
                        
                        _, pred, loss_value,ffc = sess.run(
                                [train_op, prediction,  loss,fc8],
                                feed_dict=feed_dict)
        #                print(ffc)
                        print(pred)
        #                print(loss_value)
        
                        duration = time.time() - start_time
        
                        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
        
                        # print training information
                        if step % 10 == 0 or step - startstep <= 30:
                            sec_per_batch = float(duration)
                            print ('%s: step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)' \
                                 % (datetime.now(), step, loss_value,
                                            FLAGS.batch_size/duration, sec_per_batch))
        
        
        
                        if step % 100 == 0:
                            # print 'running summary'
                            summary_str = sess.run(summary_op, feed_dict=feed_dict)
                            summary_writer.add_summary(summary_str, step)
                            summary_writer.flush()
        
                        if step % g_.SAVE_PERIOD == 0 and step > startstep:
        #                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                            saver.save(sess, '/media/ttt/Elements/TanYinghui/lung1/model2d/model2.cpkt', global_step=step)
                    batch_start = batch_end
#                    predictions = np.array([])
#                    val_y = []
#        val_batch_start = 0
#        val_batch_end = val_x.shape[0] + 1
##                    while batch_start < dataset_train.shape[0]:
##                        step += 1
##                        batch_end = batch_start + batch_size if batch_end <= dataset_train.shape[0] else dataset_train.shape[0]
#                                
#        val_batch_x = val_x[val_batch_start:val_batch_end,:,:,:,:]
##                        val_batch_y = np.squeeze(val_y[batch_start:batch_end,:])
##                        batch_start = batch_end
#        val_feed_dict = {view_: val_batch_x,
#                         keep_prob_: 1.0 }
#        pred = sess.run( prediction, feed_dict=val_feed_dict)
#        predictions = np.hstack((predictions, pred))
#        p_n = np.argwhere(predictions>0)
#        print(p_n.shape[0])



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
    c = imfile[img_names[idx]]['data']['c']  # Total number of locations
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
    return x, y, c


#def read_lists(list_of_lists_file):
#    listfile_labels = np.loadtxt(list_of_lists_file, dtype=str).tolist()
#    listfiles, labels  = zip(*[(l[0], int(l[1])) for l in listfile_labels])
#    return listfiles, labels
    


if __name__ == '__main__':
    main()


