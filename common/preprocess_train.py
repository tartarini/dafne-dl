# -*- coding: utf-8 -*-

import numpy as np
import os
import skimage
from skimage.morphology import square
import padorcut
from scipy.ndimage import zoom
import math


def to_mask(categorical_mask,dim,ch):  ##ch = 13 for thigh, 7 for leg
    segmentation_mask=np.zeros((dim,dim))
    for i in range(categorical_mask.shape[1]):
        for j in range(categorical_mask.shape[1]):
            segmentation_mask[i,j]=categorical_mask[i,j,1:ch+1].argmax()
    return segmentation_mask


def split_mirror(path,card,ch):
    '''
    Function to split the training data into left and mirrored right parts cropped and zoomed/padorcutted. 
    '''
    MODEL_SIZE = (270, 270) 
    from skimage.filters import threshold_local
    for j in range(1,card+1):
        arr=np.load(os.path.join(path,'train_'+str(j)+'.npy'))
        image=arr[:,:,0]
        if arr.shape[2]==2:
            roi=arr[:,:,1] 
        else:
            roi=to_mask(arr,432,ch)
        block_size = 15
        local_thresh = threshold_local(image, block_size, offset=10)
        binary_local = image > local_thresh
        binary_local=binary_local==0.0
        binary_local=skimage.morphology.area_opening(binary_local,area_threshold=20)
        binary_local=skimage.morphology.area_closing(binary_local,area_threshold=20)
        mountainsc=binary_local[:,:].sum(axis=0)
        mountainsr=binary_local[:,:].sum(axis=1)
        s=0
        p=0
        ii=0 
        jj=0
        a1=0
        a2=0
        a3=0
        a4=0
        b1=0
        b2=0
        while s!=4:
            if mountainsc[ii]>0 and s==0:
                s=1
                a1=ii
            if mountainsc[ii]==0 and s==1:
                s=2
                a2=ii
            if mountainsc[ii]>0 and s==2:
                s=3
                a3=ii
            if mountainsc[ii]==0 and s==3:
                s=4
                a4=ii
            ii+=1
            if ii==432 and s==2:
                s=4
                a4=a2
                a2=np.ceil((a4-a1)/2)+a1
                a3=np.ceil((a4-a1)/2)+a1
        while p!=2:
            if mountainsr[jj]>0 and p==0:
                p=1
                b1=jj
            if mountainsr[jj]==0 and p==1:
                p=2
                b2=jj
            jj+=1
        left=image[int(b1):int(b2),int(a1):int(a2)]
        left=padorcut(left, MODEL_SIZE)
        right=image[int(b1):int(b2),int(a3):int(a4)]
        right=right[::1,::-1]
        right=padorcut(right, MODEL_SIZE)
        roileft=roi[int(b1):int(b2),int(a1):int(a2)]
        roileft=padorcut(roileft, MODEL_SIZE)
        roiright=roi[int(b1):int(b2),int(a3):int(a4)]
        roiright=roiright[::1,::-1]
        roiright=padorcut(roiright, MODEL_SIZE)
        conc_left=np.stack((left,roileft),axis=-1)
        np.save(os.path.join(path,'train_'+str(j)),conc_left)
        conc_right=np.stack((right,roiright),axis=-1)
        np.save(os.path.join(path,'train_'+str(card+j)),conc_right)

 
def compute_class_frequencies(path,dim,card,ch):
    classes=[0]*ch
    images_containing_class=[0]*ch
    for j in range(1,card+1):
        arr=np.load(os.path.join(path,'train_'+str(j)+'.npy'))
        if arr.shape[2]==2:
            seg=arr[:,:,1] 
        else:
            seg=to_mask(arr,dim,ch)
        for class_ in range(ch):
            n_pixels=(seg==float(class_)).sum()
            if n_pixels>0:
                classes[class_]+=n_pixels
                images_containing_class[class_]+=dim**2
    return classes,images_containing_class


def categorical_and_weight(img,seg,av,freq,dim,band,ch): 
    from skimage.filters import threshold_otsu
    cate=np.zeros((seg.shape[0],seg.shape[1],ch),dtype='float32')
    W=np.zeros((seg.shape[0],seg.shape[1]),dtype='float32')
    global_thresh = threshold_otsu(img)
    binary_global = img > global_thresh
    binary_mask=binary_global==1.0
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            cate[i,j,int(seg[i,j])]=1.0
            if (binary_mask[i,j]>0):
                h=0
                k=0
                d1=0
                d2=0
                aa=seg[i,j]
                l=1
                while h!=1 or k!=1:
                    if (seg[i-l,j]!=aa and int(seg[i-l,j]!=0)) and h==0:
                        d1=l
                        h=1
                        bb=seg[i-l,j]
                    if (seg[min(i+l,seg.shape[0]-1),j]!=aa and int(seg[min(i+l,seg.shape[0]-1),j])!=0) and h==0:
                        d1=l
                        h=1
                        bb=seg[min(i+l,seg.shape[0]-1),j]
                    if (seg[i,j-l]!=aa and int(seg[i,j-l])!=0) and h==0:
                        d1=l
                        h=1
                        bb=seg[i,j-l]
                    if (seg[i,min(j+l,seg.shape[0]-1)]!=aa and int(seg[i,min(j+l,seg.shape[0]-1)])!=0) and h==0:
                        d1=l
                        h=1
                        bb=seg[i,min(j+l,seg.shape[0]-1)]
                    if ((seg[i-l,j]!=aa and seg[i-l,j]!=bb and int(seg[i-l,j]!=0)) or (seg[min(i+l,seg.shape[0]-1),j]!=aa and seg[min(i+l,seg.shape[0]-1),j]!=bb and int(seg[min(i+l,seg.shape[0]-1),j])!=0) or (seg[i,j-l]!=aa and seg[i,j-l]!=bb and int(seg[i,j-l])!=0) or (seg[i,min(j+l,seg.shape[0]-1)]!=aa and seg[i,min(j+l,seg.shape[0]-1)]!=bb and int(seg[i,min(j+l,seg.shape[0]-1)])!=0)) and h==1 and k==0:
                        d2=l
                        k=1
                    if l==seg.shape[0]-1:
                        d1=seg.shape[0]
                        h=1
                        k=1
                    l+=1
                W[i,j]=av/freq[int(seg[i,j])]+10*math.exp(-((d1+d2)**2)/(2*band))
    W=np.reshape(W,(dim,dim,1))
    return np.concatenate([cate,W],axis=-1)

def input_creation(path,card,dim,band,ch):
    '''
    Creates the training data with labels categorization and creation of weights maps.
    '''
    classes,images=compute_class_frequencies(path,dim,card,ch)
    frequencies=[]
    for cla,ima in zip(classes,images):
        frequencies.append(cla/ima)
    av=sum(frequencies)/ch
    for j in range(1,card+1):
        arr=np.load(os.path.join(path,'train_'+str(j)+'.npy'))
        img=arr[:,:,0]
        if arr.shape[2]==2:
           seg=arr[:,:,1] 
        else:
           seg=to_mask(arr,dim,ch)
        categ=categorical_and_weight(img,seg,av,frequencies,dim,band,ch)
        arr=np.concatenate([np.reshape(img,(dim,dim,1)),categ],axis=-1);
        np.save(os.path.join(path,'train_'+str(j)+'.npy'),arr)

