from __future__ import print_function
import csv, multiprocessing, cv2, os
import numpy as np
import urllib
import urllib.request
from itertools import islice

# the SSL certificates for hotels50K are expired now. proceed with caution!
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0"

opener = AppURLopener()
def url_to_image(url):
    resp = opener.open(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    return image

# chain,hotel,im_source,im_id,im_url
def download_and_resize(imList):
    for im in imList:
        try:
            saveDir = os.path.join('./images/train/',im[0],im[1],im[2])
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)

            savePath = os.path.join(saveDir,str(im[3])+'.'+im[4].split('.')[-1])

            if not os.path.isfile(savePath):
                img = url_to_image(im[4])
                if img.shape[1] > img.shape[0]:
                    width = 640
                    height = round((640 * img.shape[0]) / img.shape[1])
                    img = cv2.resize(img,(width, height))
                else:
                    height = 640
                    width = round((640 * img.shape[1]) / img.shape[0])
                    img = cv2.resize(img,(width, height))
                cv2.imwrite(savePath,img)
                print('Good: ' + savePath)
            else:
                print('Already saved: ' + savePath)
        except:
            print('Bad: ' + savePath)

def download_and_resize_into_same_dir(imList):
    for im in imList:
        try:
            saveDir = os.path.join('./images/train/')
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)

            savePath = os.path.join(saveDir,str(im[3])+'.'+im[4].split('.')[-1])

            if not os.path.isfile(savePath):
                img = url_to_image(im[4])
                if img.shape[1] > img.shape[0]:
                    width = 640
                    height = round((640 * img.shape[0]) / img.shape[1])
                    img = cv2.resize(img,(width, height))
                else:
                    height = 640
                    width = round((640 * img.shape[1]) / img.shape[0])
                    img = cv2.resize(img,(width, height))
                cv2.imwrite(savePath,img)
                print('Good: ' + savePath)
            else:
                print('Already saved: ' + savePath)
        except:
            print('Bad: ' + savePath)

def main():
    hotel_f = open('./input/dataset/hotel_info.csv','r')
    hotel_reader = csv.reader(hotel_f)
    hotel_headers = next(hotel_reader,None)
    hotel_to_chain = {}
    for row in hotel_reader:
        hotel_to_chain[row[0]] = row[2]

    train_f = open('./input/dataset/train_set.csv','r')
    train_reader = csv.reader(train_f)
    train_headers = next(train_reader,None)

    images = []
    for im in train_reader:
        im_id = im[0]
        im_url = im[2]
        im_source = im[3]
        hotel = im[1]
        chain = hotel_to_chain[hotel]
        images.append((chain,hotel,im_source,im_id,im_url))

    pool = multiprocessing.Pool()
    NUM_THREADS = multiprocessing.cpu_count()

    imDict = {}
    for cpu in range(NUM_THREADS):
        pool.apply_async(download_and_resize_into_same_dir,[images[cpu::NUM_THREADS]])
    pool.close()
    pool.join()

if __name__ == '__main__':
    retcode = main()