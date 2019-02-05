# -*- coding: utf-8 -*-
import numpy as np
import cv2
import random
import os
import glob
import socket
import time

peopleDir = '../images/people_crops/'

class BatchAllSet:
    def __init__(self, image_list, mean_file, image_size, crop_size, batchSize=100, num_pos=10, isTraining=True, isOverfitting=False):
        self.image_size = image_size
        self.crop_size = crop_size
        self.isTraining = isTraining
        self.isOverfitting = isOverfitting

        self.meanFile = mean_file
        meanIm = np.load(self.meanFile)

        if meanIm.shape[0] == 3:
            meanIm = np.moveaxis(meanIm, 0, -1)

        self.meanImage = cv2.resize(meanIm, (self.crop_size[0], self.crop_size[1]))

        #img = img - self.meanImage
        if len(self.meanImage.shape) < 3:
            self.meanImage = np.asarray(np.dstack((self.meanImage, self.meanImage, self.meanImage)))

        self.numPos = num_pos
        self.batchSize = batchSize

        self.classes = {}

        for im in image_list:
            split_im = im.split('/')
            image_id = int(split_im[-1].split('.')[0])
            source = split_im[-2]
            cls = int(split_im[-3])
            chain = int(split_im[-4])
            if not cls in self.classes:
                self.classes[cls] = {}
                self.classes[cls]['images'] = []
                self.classes[cls]['sources'] = []
            self.classes[cls]['images'].append(im)
            self.classes[cls]['sources'].append(source)

        for cls in self.classes.keys():
            if len(self.classes[cls]['images']) < self.numPos:
                self.classes.pop(cls)

        self.people_crop_files = glob.glob(os.path.join(peopleDir,'*.png'))


    def getBatch(self):
        numClasses = self.batchSize/self.numPos
        classes = np.random.choice(self.classes.keys(),numClasses,replace=False)
        labels = np.zeros([self.batchSize],dtype='int')
        ims = []

        ctr = 0
        for cls in classes:
            cls = int(cls)
            clsPaths = self.classes[cls]['ims']
            clsSources = np.array(self.classes[cls]['sources'])
            traffickcamInds = np.where(clsSources=='traffickcam')[0]
            exInds = np.where(clsSources=='travel_website')[0]
            if len(traffickcamInds) >= self.numPos/2 and len(exInds) >= self.numPos/2:
                numtraffickcam = self.numPos/2
                numEx = self.numPos - numtraffickcam
            elif len(traffickcamInds) >= self.numPos/2 and len(exInds) < self.numPos/2:
                numEx = len(exInds)
                numtraffickcam = self.numPos - numEx
            else:
                numtraffickcam = len(traffickcamInds)
                numEx = self.numPos - numtraffickcam

            random.shuffle(traffickcamInds)
            random.shuffle(exInds)

            for j1 in np.arange(numtraffickcam):
                imPath = self.classes[cls]['ims'][traffickcamInds[j1]]
                labels[ctr] = cls
                ims.append(imPath)
                ctr += 1

            for j2 in np.arange(numEx):
                imPath = self.classes[cls]['ims'][exInds[j2]]
                labels[ctr] = cls
                ims.append(imPath)
                ctr += 1

        batch = self.getProcessedImages(ims)

        return batch, labels, ims

    def getBatchFromImageList(self,image_list):
        batch = np.zeros([len(image_list), self.crop_size[0], self.crop_size[1], 3])
        for ix in range(0,len(image_list)):
            img = self.getProcessedImage(image_list[ix])
            batch[ix,:,:,:] = img
        return batch

    def getProcessedImage(self, image_file):
        img = cv2.imread(image_file)
        if img is None:
            return None

        if self.isTraining and not self.isOverfitting and random.random() > 0.5:
            img = cv2.flip(img,1)

        # if self.isTraining:
        #     img = doctor_im(img)

        img = cv2.resize(img, (self.image_size[0], self.image_size[1]))

        if self.isTraining and not self.isOverfitting:
            top = np.random.randint(self.image_size[0] - self.crop_size[0])
            left = np.random.randint(self.image_size[1] - self.crop_size[1])
        else:
            top = int(round((self.image_size[0] - self.crop_size[0])/2))
            left = int(round((self.image_size[1] - self.crop_size[1])/2))

        img = img[top:(top+self.crop_size[0]),left:(left+self.crop_size[1]),:]
        img = img - self.meanImage

        return img

    def getProcessedImages(self,ims):
        numIms = len(ims)
        imgs = np.array([cv2.resize(cv2.imread(image_file),(self.image_size[0], self.image_size[1])) for image_file in ims])

        if self.isTraining and not self.isOverfitting and random.random() > 0.5:
            imgs = np.array([cv2.flip(img,1) if random.random() > 0.5 else img for img in imgs])

        if self.isTraining and not self.isOverfitting:
            top = np.random.randint(0,self.image_size[0] - self.crop_size[0],numIms)
            left = np.random.randint(0,self.image_size[1] - self.crop_size[1],numIms)
        else:
            top = int(round((self.image_size[0] - self.crop_size[0])/2))
            left = int(round((self.image_size[1] - self.crop_size[1])/2))

        imgs = np.array([imgs[ix,top[ix]:(top[ix]+self.crop_size[0]),left[ix]:(left[ix]+self.crop_size[1]),:]-self.meanImage for ix in range(numIms)])
        return imgs

    def getPeopleMasks(self):
        which_inds = random.sample(np.arange(0,len(self.people_crop_files)),self.batchSize)

        people_crops = np.zeros([self.batchSize,self.crop_size[0],self.crop_size[1]])
        for ix in range(0,self.batchSize):
            people_crops[ix,:,:] = self.getImageAsMask(self.people_crop_files[which_inds[ix]])

        people_crops = np.expand_dims(people_crops, axis=3)

        return people_crops

    def getImageAsMask(self, image_file):
        img = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # how much of the image should the mask take up
        scale = np.random.randint(30,70)/float(100)
        resized_img = cv2.resize(img,(int(self.crop_size[0]*scale),int(self.crop_size[1]*scale)))

        # where should we put the mask?
        top = np.random.randint(0,self.crop_size[0]-resized_img.shape[0])
        left = np.random.randint(0,self.crop_size[1]-resized_img.shape[1])

        new_img = np.ones((self.crop_size[0],self.crop_size[1]))*255.0
        new_img[top:top+resized_img.shape[0],left:left+resized_img.shape[1]] = resized_img

        new_img[new_img<255] = 0
        new_img[new_img>1] = 1

        return new_img

class NonTripletSet(BatchAllSet):
    def __init__(self, image_list, mean_file, image_size, crop_size, batchSize=100, num_pos=10, isTraining=True, isOverfitting=False):
        self.image_size = image_size
        self.crop_size = crop_size
        self.isTraining = isTraining
        self.isOverfitting = isOverfitting

        self.meanFile = mean_file
        meanIm = np.load(self.meanFile)

        if meanIm.shape[0] == 3:
            meanIm = np.moveaxis(meanIm, 0, -1)

        self.meanImage = cv2.resize(meanIm, (self.crop_size[0], self.crop_size[1]))

        #img = img - self.meanImage
        if len(self.meanImage.shape) < 3:
            self.meanImage = np.asarray(np.dstack((self.meanImage, self.meanImage, self.meanImage)))

        self.numPos = num_pos
        self.batchSize = batchSize

        self.classes = {}
        for im in image_list:
            split_im = im.split('/')
            image_id = int(split_im[-1].split('.')[0])
            source = split_im[-2]
            cls = int(split_im[-3])
            chain = int(split_im[-4])
            if not cls in self.classes:
                self.classes[cls] = {}
                self.classes[cls]['images'] = []
                self.classes[cls]['sources'] = []
            self.classes[cls]['images'].append(im)
            self.classes[cls]['sources'].append(source)

        self.people_crop_files = glob.glob(os.path.join(peopleDir,'*.png'))
