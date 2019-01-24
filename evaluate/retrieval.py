# KNN Retrieval Code. Takes in a csv file, computes average top1, top10, top100 retrieval accuracy by hotel instance
# and average top1, top3, top5 chain accuracy.

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

def class_to_chain_parser(hotel_info_file):
    class_to_chain = {}
    with open(hotel_info_file) as f:
        csv_reader = csv.reader(f,delimiter=',')
        lnNum = 0
        for row in csv_reader:
            if lnNum == 0:
                pass
            else:
                class_to_chain[int(row[0])] = int(row[2])
            lnNum += 1
    return class_to_chain

def main(csv_file):
    test_id_to_class = id_to_class_parser('../input/dataset/test_set.csv')
    train_id_to_class = id_to_class_parser('../input/dataset/train_set.csv')
    classes_to_chain = class_to_chain_parser('../input/dataset/hotel_info.csv')

    result_dict = {}
    # Check to make sure there are no invalid image IDs in the results file.
    # Assuming everything is valid, create a dict with image IDs in the class,
    # and query class and result class values.
    with open(csv_file) as cf:
        csv_reader = csv.reader(cf,delimiter=',')
        lnNum = 0
        for row in csv_reader:
            query_image_id = int(row[0])
            if len(row) < 101:
                print "Expected each row to contain a query image ID and 100 result image IDs. Failed at line: " + str(lnNum)
                break
            try:
                query_class = test_id_to_class[query_image_id]
                result_dict[query_image_id] = {}
                result_dict[query_image_id]['query_class'] = query_class
                result_dict[query_image_id]['query_chain'] = classes_to_chain[query_class]
                result_dict[query_image_id]['result_classes'] = np.zeros(100,dtype='int')
                result_dict[query_image_id]['result_chains'] = np.zeros(100,dtype='int')
            except:
                print "Query image ID ("+row[0]+") in row " + str(lnNum) + " is unknown."
                break
            for idx in range(len(row[1:])):
                try:
                    result_class = train_id_to_class[int(row[1+idx])]
                    result_dict[query_image_id]['result_classes'][idx] = result_class
                    result_dict[query_image_id]['result_chains'][idx] = classes_to_chain[result_class]
                except:
                    print "Result  image ID ("+row[1+idx]+") in row " + str(lnNum) + " is unknown."
                    break
            lnNum += 1

    num_queries = len(result_dict.keys())
    top_k_instance = np.zeros((num_queries,100),dtype='int')
    for idx in range(num_queries):
        query_image_id = result_dict.keys()[idx]
        query_class = result_dict[query_image_id]['query_class']
        result_classes = result_dict[query_image_id]['result_classes']
        correct_results = np.where(result_classes==query_class)[0]
        if len(correct_results) > 0:
            top_hit = correct_results[0]
            top_k_instance[idx,top_hit:] = 1

    average_instance_retrieval_accuracy = np.mean(top_k_instance,axis=0)

    top_k_chain = np.zeros((num_queries,100),dtype='int')
    known_chain_inds = []
    for idx in range(num_queries):
        query_image_id = result_dict.keys()[idx]
        query_chain = result_dict[query_image_id]['query_chain']
        if query_chain > -1:
            known_chain_inds.append(idx)
        result_chains = result_dict[query_image_id]['result_chains']
        correct_results = np.where(result_chains==query_chain)[0]
        if len(correct_results) > 0:
            top_hit = correct_results[0]
            top_k_chain[idx,top_hit:] = 1

    only_queries_with_chains = top_k_chain[np.array(known_chain_inds),:]
    average_chain_retrieval_accuracy = np.mean(only_queries_with_chains,axis=0)

    print 'Hotel instance retrieval accuracy for ' + csv_file
    print 'Top-1: %0.2f' % (average_instance_retrieval_accuracy[0]*100)
    print 'Top-10: %0.2f' % (average_instance_retrieval_accuracy[9]*100)
    print 'Top-100: %0.2f' % (average_instance_retrieval_accuracy[99]*100)

    print '---'
    print 'Hotel chain retrieval accuracy for ' + csv_file
    print 'Top-1: %0.2f' % (average_chain_retrieval_accuracy[0]*100)
    print 'Top-3: %0.2f' % (average_chain_retrieval_accuracy[2]*100)
    print 'Top-5: %0.2f' % (average_chain_retrieval_accuracy[4]*100)


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        print 'Expected input parameters: csv_file'
    csv_file = args[1]
    main(csv_file)
