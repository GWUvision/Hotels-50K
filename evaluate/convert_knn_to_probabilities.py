import csv
import numpy as np
import sys

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

def main(input_file,output_file):
    test_id_to_class = id_to_class_parser('../input/dataset/test_set.csv')
    train_id_to_class = id_to_class_parser('../input/dataset/train_set.csv')

    hotel_class_ids = np.unique(train_id_to_class.values())

    with open(input_file) as in_f:
        with open(output_file,'wb') as out_f:
            csv_reader = csv.reader(in_f,delimiter=',')
            lnNum = 0
            for row in csv_reader:
                query_image_id = int(row[0])
                class_probabilities = np.zeros(hotel_class_ids.shape[0])
                if len(row) != 101:
                    print "Expected each row to contain a query image ID and 100 result image IDs. Failed at line: " + str(lnNum)
                    break
                for idx in range(len(row[1:])):
                    try:
                        result_class = train_id_to_class[int(row[1+idx])]
                        result_class_ind = np.where(hotel_class_ids==result_class)[0][0]
                        class_probabilities[result_class_ind] += 1.
                    except:
                        print "Result  image ID ("+row[1+idx]+") in row " + str(lnNum) + " is unknown."
                        break

                class_probabilities = class_probabilities / 100.
                non_zero_inds = np.where(class_probabilities>0)[0]
                prob_str = row[0] + ','
                prob_str += ','.join(['%i,%0.3f' % (c,p) for c,p in zip(hotel_class_ids[non_zero_inds],class_probabilities[non_zero_inds])])
                prob_str += '\n'

                out_f.writelines(prob_str)
                lnNum += 1

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 3:
        print 'Expected input parameters: input_knn_results_file_path, output_class_probabilities_file_path'
    input_path = args[1]
    output_path = args[2]
    main(input_path,output_path)
