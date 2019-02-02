from __future__ import print_function
# TODO: Update to not expect all 50k classes in the input csv
# instead, for each query, create the 50k vector, and use hotel_class_ids to index into it.

import csv
import numpy as np
import sys
from sklearn.metrics import log_loss
from utils import id_to_class_parser

def id_to_class_parser(dataset_file):
    id_to_class = {}
    with open(dataset_file) as f:
        csv_reader = csv.reader(f,delimiter=',')
        lnNum = 0
        for row in csv_reader:
            if lnNum == 0:
                pass
            else:
                id_to_class[int(row[0])] = int(row[1])
            lnNum += 1
    return id_to_class

# csv_file = '../baseline_implementation/class_prob_output/unoccluded.csv'
def main(csv_file):
    dirname = os.path.dirname(__file__)
    test_id_to_class = id_to_class_parser(os.path.join(dirname,'..','input/dataset/test_set.csv'))
    train_id_to_class = id_to_class_parser(os.path.join(dirname,'..','input/dataset/train_set.csv'))
    hotel_class_ids = np.unique(train_id_to_class.values())

    losses = np.array((0))
    with open(csv_file) as cf:
        csv_reader = csv.reader(cf,delimiter=',')
        lnNum = 0
        for row in csv_reader:
            if lnNum % 1000 == 0 and lnNum != 0:
                print('Computed log loss for rows ' + str(lnNum-1000) + ' through ' + str(lnNum))
            if lnNum == 0:
                print('Starting to compute log loss. This may take a while!')
            query_image_id = int(row[0])
            result_probs = np.zeros((hotel_class_ids.shape[0]))
            try:
                query_class = test_id_to_class[query_image_id]
            except:
                print("Query image ID ("+row[0]+") in row " + str(lnNum) + " is unknown.")
                break

            # check which classes are in the result file, look up what the class index is for that class
            result_class_inds = np.array([np.where(hotel_class_ids==int(r))[0][0] for r in row[1::2]])
            # set the values in result_probs for each class in the result file
            result_probs[result_class_inds] = np.array([float(r) for r in row[2::2]])
            # get the index for the correct hotel
            query_class_ind = np.where(hotel_class_ids==query_class)[0][0]

            # compute the log loss
            ll = log_loss([query_class_ind],[result_probs],labels=np.arange(hotel_class_ids.shape[0]))
            losses = np.vstack((losses,ll))
            lnNum += 1

    print('Log loss for ' + csv_file)
    print('%0.2f' % (np.mean(losses)))

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        print('Expected input parameters: csv_file')
    csv_file = args[1]
    main(csv_file)
