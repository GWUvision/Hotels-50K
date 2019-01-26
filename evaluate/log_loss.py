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

def main(csv_file):
    test_id_to_class = id_to_class_parser('../input/dataset/test_set.csv')
    train_id_to_class = id_to_class_parser('../input/dataset/train_set.csv')
    hotel_class_ids = np.unique(train_id_to_class.values())

    with open(csv_file) as cf:
        csv_reader = csv.reader(cf,delimiter=',')
        lnNum = 0
        for row in csv_reader:
            query_image_id = int(row[0])
            if len(row) != 100001:
                print "Expected each row to contain a query image ID and 50000 class ID, probability pairs. Failed at line: " + str(lnNum)
                break
            try:
                query_class = test_id_to_class[query_image_id]
            except:
                print "Query image ID ("+row[0]+") in row " + str(lnNum) + " is unknown."
                break
