import sys
import caffe
import numpy as np
import cv2
import math
import os

def EditFcnProto(templateFile, height, width):
    with open(templateFile, 'r') as ft:
        template = ft.read()
        outFile = 'DehazeNetFcn.prototxt'
        with open(outFile, 'w') as fd:
            fd.write(template.format(height_15=height+15, width_15=width+15,
                height_11=height+11, width_11=width+11))


def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz,1)
    imvec = im.reshape(imsz,3)
    indices = darkvec.argsort()
    indices = indices[imsz-numpx::]
    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
        atmsum = atmsum + imvec[indices[ind]]
    A = atmsum / numpx
    return A

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p
    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I
    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I
    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))
    q = mean_a*im + mean_b
    return q

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray,et,r,eps)
    return t

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype)
    t = cv2.max(t,tx)
    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]
    return res

if __name__ == '__main__':
    if not len(sys.argv) == 2:
        print 'Usage: python DeHazeNet.py haze_imgdir_path'
        exit()
    else:
        img_dir_path = sys.argv[1]
        
        npad, net_full_conv, transformers = initial_setup()

        if not os.path.exists(img_dir_path + "Dehazed/"):
            os.mkdir(img_dir_path + "Dehazed/")
        templateFile = 'DehazeFcnTemplate.prototxt'
        for i in os.listdir(img_dir_path):
                if not i.endswith('jpg'):
                    continue
                im_path = img_dir_path + i
                src = cv2.imread(im_path)
                height = src.shape[0]
                width = src.shape[1]
                EditFcnProto(templateFile, height, width)
                I = src/255.0
                dark = DarkChannel(I,15)
                A = AtmLight(I,dark)
                te = TransmissionEstimate(npad, net_full_conv, transformers, im_path, height, width)
                t = TransmissionRefine(src,te)
                J = Recover(I,t,A,0.1)
                #cv2.imshow('TransmissionEstimate',te)
                #cv2.imshow('TransmissionRefine',t)
                #cv2.imshow('Origin',src)
                #cv2.imshow('Dehaze',J)
                #cv2.waitKey(0)
                print('Processed Image written to '+img_dir_path+'Dehazed/'+i)
                cv2.imwrite(img_dir_path+'Dehazed/'+i,J*255)
