import tensorflow as tf
from classfile import *
import numpy as np
import tensorflow.contrib.slim as slim
from nets import resnet_v2
import faiss
import h5py
import sys
import glob

def save_h5(data_description,data,data_type,path):
    h5_feats=h5py.File(path,'w')
    h5_feats.create_dataset(data_description, data=data, dtype=data_type)
    h5_feats.close()

def load_h5(data_description,path):
    with h5py.File(path, 'r') as hf:
        data = hf[data_description][:]
    return data

def main():
    output_dir = './features/'
    pretrained_net = './baseline_snapshot/baseline_snapshot'
    train_images = glob.glob('../images/train/*/*/*/*.jpg')

    img_size = [256, 256]
    crop_size = [224, 224]
    batch_size = 120
    output_size = 256
    mean_file = '../input/meanIm.npy'
    image_batch = tf.placeholder(tf.float32, shape=[None, crop_size[0], crop_size[0], 3])

    print("Preparing network...")
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        _, layers = resnet_v2.resnet_v2_50(image_batch, num_classes=output_size, is_training=False)

    featLayer = 'resnet_v2_50/logits'
    feat = tf.squeeze(tf.nn.l2_normalize(layers[featLayer],3))
    c = tf.ConfigProto()
    c.gpu_options.visible_device_list=str(3) # specify which gpu you want to run on
    sess = tf.Session(config=c)
    saver = tf.train.Saver()
    saver.restore(sess, pretrained_net)

    occlusion_levels = ['unoccluded','low_occlusions','medium_occlusions','high_occlusions']
    for occlusion in occlusion_levels:
        test_output_dir = os.path.join('./features/',occlusion)
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)

        test_images = glob.glob(os.path.join('../images/test',occlusion,'*/*/*/*.jpg'))
        test_data = NonTripletSet(test_images, mean_file, img_size, crop_size, isTraining=False)
        test_ims = []
        test_classes = []
        for cls in test_data.classes.keys():
            for im in test_data.classes[cls]['images']:
                test_ims.append(int(im.split('/')[-1].split('.')[0])) # convert image path to image id
                test_classes.append(cls)

        test_ims = np.array(test_ims)
        test_classes = np.array(test_classes)
        test_feats = np.zeros((test_ims.shape[0],output_size))
        for ix in range(0,test_ims.shape[0],batch_size):
            image_list = test_ims[ix:ix+batch_size]
            batch = test_data.getBatchFromImageList(image_list)
            ff = sess.run(feat,{image_batch:batch})
            test_feats[ix:ix+ff.shape[0],:] = ff
            print 'Test features: ',ix+ff.shape[0], ' out of ' , test_feats.shape[0]
            save_h5('test_ims',test_ims,'i8',os.path.join(test_output_dir,'testIms.h5'))
            save_h5('test_classes',test_classes,'i8',os.path.join(test_output_dir,'testClasses.h5'))
            save_h5('test_feats',test_feats,'f',os.path.join(test_output_dir,'testFeats.h5'))

    train_data = NonTripletSet(train_images, mean_file, img_size, crop_size, isTraining=False)
    train_ims = []
    train_classes = []
    for cls in train_data.classes.keys():
        for im in train_data.classes[cls]['images']:
            train_ims.append(int(im.split('/')[-1].split('.')[0]))  # convert image path to image id
            train_classes.append(cls)

    train_ims = np.array(train_ims)
    train_classes = np.array(train_classes)
    train_feats = np.zeros((train_ims.shape[0],output_size))
    for ix in range(0,train_ims.shape[0],batch_size):
        image_list = train_ims[ix:ix+batch_size]
        batch = train_data.getBatchFromImageList(image_list)
        ff = sess.run(feat,{image_batch:batch})
        train_feats[ix:ix+ff.shape[0],:] = ff
        print 'Train features: ', ix+ff.shape[0], ' out of ' , train_feats.shape[0]

    save_h5('train_ims',train_ims,'i8',os.path.join(output_dir,'trainIms.h5'))
    save_h5('train_classes',train_classes,'i8',os.path.join(output_dir,'trainClasses.h5'))
    save_h5('train_feats',train_feats,'f',os.path.join(output_dir,'trainFeats.h5'))

if __name__ == "__main__":
    main()
