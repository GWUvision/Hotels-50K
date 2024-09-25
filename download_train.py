from __future__ import print_function
import csv, multiprocessing, cv2, os
import numpy as np
import urllib.request
import ssl
import cv2
import tempfile

# Create a custom context that ignores SSL certificate errors
context = ssl._create_unverified_context()

class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0"

opener = AppURLopener()

def url_to_image(url):
    # Open the URL with the custom SSL context to ignore certificate errors
    resp = urllib.request.urlopen(url, context=context)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    return image

def download_and_resize(imList, temp_bad_file):
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
        except Exception as e:
            with open(temp_bad_file, 'a') as f:
                f.write(f"{savePath}\n")
            print(f'Bad: {savePath} ({e})')

def main():
    hotel_f = open('./input/dataset/hotel_info.csv','r')
    hotel_reader = csv.reader(hotel_f)
    hotel_headers = next(hotel_reader, None)
    hotel_to_chain = {}
    for row in hotel_reader:
        hotel_to_chain[row[0]] = row[2]

    train_f = open('./input/dataset/train_set.csv','r')
    train_reader = csv.reader(train_f)
    train_headers = next(train_reader, None)

    images = []
    for im in train_reader:
        im_id = im[0]
        im_url = im[2]
        im_source = im[3]
        hotel = im[1]
        chain = hotel_to_chain.get(hotel, 'Unknown')
        images.append((chain, hotel, im_source, im_id, im_url))

    pool = multiprocessing.Pool()
    NUM_THREADS = multiprocessing.cpu_count()

    # Create temp files for each process to write bad files
    temp_bad_files = [tempfile.NamedTemporaryFile(delete=False, mode='w').name for _ in range(NUM_THREADS)]

    # Distribute tasks across processes
    for cpu in range(NUM_THREADS):
        pool.apply_async(download_and_resize, [images[cpu::NUM_THREADS], temp_bad_files[cpu]])
    
    pool.close()
    pool.join()

    # Concatenate all bad files into one
    with open('./bad_files.txt', 'w') as outfile:
        for temp_file in temp_bad_files:
            with open(temp_file, 'r') as infile:
                outfile.write(infile.read())
            os.remove(temp_file)  # Remove temp file after concatenation

    print("Bad files are saved in 'bad_files.txt'")

if __name__ == '__main__':
    retcode = main()