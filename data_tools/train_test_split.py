import argparse
import csv
import os
import random

test_percent = 0.2
output_dir = "gt-split"


# Split the ground truth into train, test sets
def split_gt(groundtruth, test_percent=0.2, data_num=None):
    with open(groundtruth, "r") as fd:
        data = fd.read()
        data = data.split('\n')
        data = [x.split('\t') for x in data]
        random.shuffle(data)
        if data_num:
            assert sum(data_num) < len(data)
            return data[:data_num[0]], data[data_num[0]:data_num[0] + data_num[1]]
        test_len = round(len(data) * test_percent)
        return data[test_len:], data[:test_len] # train, test


def write_tsv(data, path):
    with open(path, "w") as fd:
        writer = csv.writer(fd, delimiter="\t")
        writer.writerows(data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--test-percent",
        dest="test_percent",
        default=test_percent,
        type=float,
        help="Percent of data to use for test [Default: {}]".format(test_percent)
    )
    parser.add_argument(
        "-n",
        "--data_num",
        nargs=2,
        type=int,
        help="Number of train data and test data",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        required=True,
        type=str,
        help="Path to input ground truth file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        default=output_dir,
        type=str,
        help="Directory to save the split ground truth files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    train_gt, test_gt = split_gt(options.input, options.test_percent, options.data_num)
    if not os.path.exists(options.output_dir):
        os.makedirs(options.output_dir)
    write_tsv(train_gt, os.path.join(options.output_dir, "train.txt"))
    write_tsv(test_gt, os.path.join(options.output_dir, "test.txt"))
