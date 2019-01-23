import faiss
import h5py
import os
import numpy as np

def load_h5(data_description,path):
    with h5py.File(path, 'r') as hf:
        data = hf[data_description][:]
    return data

def main():
    output_dir = './features/'

    train_ims = load_h5('train_ims',os.path.join(output_dir,'trainIms.h5'))
    train_classes = load_h5('train_classes',os.path.join(output_dir,'trainClasses.h5'))
    train_feats = load_h5('train_feats',os.path.join(output_dir,'trainFeats.h5'))

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 3 # specify which GPU to use

    gpu_index = faiss.GpuIndexFlatIP(res, train_feats.shape[1],flat_config)
    for feat in train_feats:
        gpu_index.add(np.expand_dims(feat,0))

    csv_dir = './csv_output'
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    occlusion_levels = ['unoccluded','low_occlusions','medium_occlusions','high_occlusions']
    for occlusion in occlusion_levels:
        with open(os.path.join(csv_dir,occlusion+'.csv'),'wb') as csv_file:
            test_output_dir = os.path.join('./features/',occlusion)
            test_ims = load_h5('test_ims',os.path.join(test_output_dir,'testIms.h5'))
            test_feats = load_h5('test_feats',os.path.join(test_output_dir,'testFeats.h5'))
            for imId,ft in zip(test_ims,test_feats):
                print occlusion, imId
                result_dists, result_inds = gpu_index.search(np.expand_dims(ft,0).astype('float32'),100)
                result_im_inds = train_ims[result_inds[0]]
                csv_line = str(imId) + ',' + ','.join([str(r) for r in result_im_inds]) +'\n'
                csv_file.writelines(csv_line)

if __name__ == "__main__":
    main()
