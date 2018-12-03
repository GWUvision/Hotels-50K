import csv, multiprocessing, cv2, os
import numpy as np
import urllib
import json

def url_to_image(url):
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    return image

def resize(imList):
    for im in imList:
        try:
            imUrl = im[2]
            saveDir = os.path.join('./images/train/',im[0],im[1])
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)

            savePath = os.path.join(saveDir,imUrl.split('/')[-1])

            if not os.path.isfile(savePath):
                img = url_to_image(imUrl)
                if img.shape[1] > img.shape[0]:
                    width = 640
                    height = (640 * img.shape[0]) / img.shape[1]
                    img = cv2.resize(img,(width, height))
                else:
                    height = 640
                    width = (640 * img.shape[1]) / img.shape[0]
                    img = cv2.resize(img,(width, height))
                cv2.imwrite(savePath,img)
                print 'Good: ' + savePath
            else:
                print 'Already saved: ' + savePath
        except:
            print 'Bad: ' + savePath

def main():
    jsonDataPath = './input/train_set.json'
    with open(jsonDataPath) as f:
        data = json.load(f)

    imList = []
    for hotel in data.keys():
        info = data[hotel]
        for source in info['ims'].keys():
            for im in info['ims'][source].keys():
                im_path = info['ims'][source][im]['path']
                imList.append((hotel,source,im_path))

    pool = multiprocessing.Pool()
    NUM_THREADS = multiprocessing.cpu_count()

    imDict = {}
    for cpu in range(NUM_THREADS):
        pool.apply_async(resize,[imList[cpu::NUM_THREADS]])
    pool.close()
    pool.join()

if __name__ == '__main__':
    retcode = main()
