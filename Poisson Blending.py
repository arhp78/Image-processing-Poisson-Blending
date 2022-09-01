# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 23:36:50 2021

@author: hatam
"""

import numpy as np
import cv2
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
lengh=735
witdth=324

                 
def poison(target,mask_source1,source):
    x,y=np.where(mask_source1[:,:]>=100)
    kernel = np.array([[0,-1,0], [-1,4,-1], [0,-1,0]])
   
    mask_target_laplace = cv2.filter2D(source, -1, kernel)
    matrix_A=np.zeros((int(source[:,:].size),int(source[:,:].size)),dtype='int8')
    matrix_B=np.zeros((int(source[:,:].size),1))
    matrix_X=np.zeros((int(source[:,:].size),1))
    
    for i in range(len(source)):
        for j in range(len(source[0])):
            a=np.where(x==lengh+i)
            b=np.where(y==witdth+j)
            #exists1= common(a,b)
            exists1=len(set(a[0][:])&set(b[0][:])) == 0
            num_of_row=i*len(source[0])+j-1
            if( not exists1 ):
                matrix_A[num_of_row,num_of_row]=4
                matrix_A[num_of_row,num_of_row-1]=-1
                matrix_A[num_of_row,num_of_row+1]=-1
                matrix_A[num_of_row,num_of_row-len(source[0])]=-1
                matrix_A[num_of_row,num_of_row+len(source[0])]=-1
                matrix_B[num_of_row,0]=mask_target_laplace[i,j]
            else:
                matrix_A[num_of_row,num_of_row]=1
                matrix_B[num_of_row,0]=target[lengh+i,witdth+j]
                
    A = csc_matrix(matrix_A)
    matrix_X= spsolve(A, matrix_B)




    for i in range(len(source)):
        for j in range(len(source[0])):
            num_of_row=i*len(source[0])+j
            target[lengh+i,witdth+j]=matrix_X[num_of_row]
    
    
    return target


#read image
target=cv2.imread("2.target.jpg")
source=cv2.imread("1.source.jpg")

source1=cv2.imread("mask-source.jpg")
mask_source=np.zeros_like(target)
mask_source[lengh:lengh+len(source),witdth:witdth+len(source[0]),:]=source1

x,y=np.where(mask_source[:,:,0]>=100)
target=target.astype("float32")
mask_source=mask_source.astype("float32")
source=source.astype("float32")

target[:,:,0]=poison(target[:,:,0],mask_source[:,:,0],source[:,:,0])
target[:,:,1]=poison(target[:,:,1],mask_source[:,:,1],source[:,:,1])
target[:,:,2]=poison(target[:,:,2],mask_source[:,:,2],source[:,:,2])



target[x,y,0]=target[x,y,0]+abs(target[x,y,0].min())
target[x,y,0]=target[x,y,0]/(target[x,y,0].max()-target[x,y,0].min())
target[x,y,0]=target[x,y,0]*250

target[x,y,1]=target[x,y,1]+abs(target[x,y,1].min())
target[x,y,1]=target[x,y,1]/(target[x,y,1].max()-target[x,y,1].min())
target[x,y,1]=target[x,y,1]*250

target[x,y,2]=target[x,y,2]+abs(target[x,y,2].min())
target[x,y,2]=target[x,y,2]/(target[x,y,2].max()-target[x,y,2].min())
target[x,y,2]=target[x,y,2]*250

'''
len(set(a)&set(b)) == 0
#x1,y1=np.where(target[x,y,0]<0)
#target[x1,y1,0]=0
x1,y1=np.where(target[x,y,0]>255)
target[x1,y1,0]=255

#x1,y1=np.where(target[x,y,1]<0)
#target[x1,y1,1]=0
x1,y1=np.where(target[x,y,1]>255)
target[x1,y1,1]=255

#x1,y1=np.where(target[x,y,2]<0)
#target[x1,y1,2]=0
x1,y1=np.where(target[x,y,2]>255)
target[x1,y1,2]=255
'''

target=target.astype("uint8")
cv2.imwrite("res1.jpg", target)
  
