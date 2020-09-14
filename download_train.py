from __future__ import print_function
import csv, multiprocessing, cv2, os
import numpy as np
import urllib
#import urllib.request
#import urllib2
import time, random
import requests
import http.client
requests.packages.urllib3.disable_warnings()


def url_to_image(url):
    proxy_addr = getproxy()
    proxy = {'http': proxy_addr}
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36',\
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',\
               'Accept-Encoding': 'gzip, deflate, br',\
               'Accept-Language': 'en-US,en;q=0.9',\
               'Connection': 'keep-alive'}
    try:
        resp = requests.get(
                    url=url,
                    headers=headers,
                    timeout=5,
                    verify=False)
    except (http.client.IncompleteRead) as e:
        resp = e.partial
    except requests.Timeout:
        pass
    if(resp.status_code == 200):
        image = np.asarray(bytearray(resp.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    else:
        image = None 
    return image

def write_to_log(im,err_msg):
    with open('./error_log.csv','a') as csvfile:
        log = csv.writer(csvfile)
        log.writerow([im[2],im[3],im[4],err_msg])
        csvfile.close()

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
                    height = int(round((640 * img.shape[0]) / img.shape[1]))
                    img = cv2.resize(img,(width, height))
                else:
                    height = 640
                    width = int(round((640 * img.shape[1]) / img.shape[0]))
                    img = cv2.resize(img,(width, height))
                cv2.imwrite(savePath,img)
                print('Good: ' + savePath)
                #time.sleep(0.5)
            else:
                print('Already saved: ' + savePath)
        except Exception as e:
            write_to_log(im,e.message)
            print('Bad: ' + e.message)
            #print(e)
            #raise Exception(e.message)
            
def download_images(images):
    try:
        pool = multiprocessing.Pool()
        NUM_THREADS = multiprocessing.cpu_count()

        imDict = {}
        for cpu in range(NUM_THREADS):
            pool.apply_async(download_and_resize,[images[cpu::NUM_THREADS]])
    except Exception as e:
        print(e.message)
    pool.close()
    pool.join()

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
    images_TC = []
    count= 0
    for im in train_reader:
        im_id = im[0]
        im_url = im[2]
        im_source = im[3]
        hotel = im[1]
        chain = hotel_to_chain[hotel]
        if (im_source != 'traffickcam'):
            #hotel website images
            images.append((chain,hotel,im_source,im_id,im_url))
        else:
            #traffickcam images
            images_TC.append((chain,hotel,im_source,im_id,im_url))

    download_images(images_TC)  
    download_images(images)

if __name__ == '__main__':
    retcode = main()
