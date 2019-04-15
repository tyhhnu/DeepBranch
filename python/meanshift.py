#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 18:16:23 2018

@author: tyh
"""

from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
from io1 import  loadmarker,savemarker,loadimg
X = loadmarker('/media/ttt/LANKEXIN/shuju/fjp/flylight_janelia_part1_25_sub.tif_s.marker')
marker_r=[]
#bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=1067)   
bandwidth=8
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)   
ms.fit(X)   
labels = ms.labels_   
cluster_centers = ms.cluster_centers_   
  
labels_unique = np.unique(labels)   
n_clusters_ = len(labels_unique)   
new_X = np.column_stack((X, labels))   
#savemarker('/media/tyh/3E5618CD561887B3/test/amop1.-1.marker',marker_r)
a=new_X[:,3]
b=np.max(a)

for i in range(int(b)+1):
    c=np.array(np.where(a==i))
    if c.shape[1]>1:
        marker_r.append(cluster_centers[i,0:3])
marker_r=np.array(marker_r)
savemarker('/media/ttt/LANKEXIN/shuju/fjp/flylight_janelia_part1_25_sub_s.tifm.marker',marker_r)
savemarker('/media/ttt/LANKEXIN/shuju/fjp/flylight_janelia_part1_25_sub_s.tifmc.marker',cluster_centers)
    
    
    
    