import csv

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
