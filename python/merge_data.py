import os

def merge_data(input_path1, input_path2, output_path):
    print("Begin merging data ...")
    if not os.path.isfile(input_path1):
        print("ERROR: file not exist. " + input_path1)
        exit(-1)
    if not os.path.isfile(input_path2):
        print("ERROR: file not exist. " + input_path2)
        exit(-1)
    fi1 = open(input_path1, 'r')
    fi2 = open(input_path2, 'r')
    fo = open(output_path, 'w')
    for line in fi1:
        fo.write(line)
    for line in fi2:
        fo.write(line)
    fi1.close()
    fi2.close()
    fo.close()
    print("Merging done.")

id = 2261
if __name__ == '__main__':
    data_folder = "../../make-ipinyou-data/"
    train_path = data_folder + str(id) + "/train.yzx.txt"
    test_path = data_folder + str(id) + "/test.yzx.txt"
    merge_path = data_folder + str(id) + "/all.yzx.txt"

    merge_data(train_path, test_path, merge_path)