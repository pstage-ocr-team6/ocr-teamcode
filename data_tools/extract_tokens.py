import argparse
import csv
import sys


def parse_symbols(truth):
    unique_symbols = set(truth.split())
    return unique_symbols


def create_tokens(groundtruth, output="tokens.txt"):
    with open(groundtruth, "r") as fd:
        data = fd.read()

    unique_symbols = set()
    data = data.split("\n")
    data = [x.split("\t") for x in data]
    for _, truth in data:
        truth_symbols = parse_symbols(truth)
        unique_symbols = unique_symbols.union(truth_symbols)

    symbols = list(unique_symbols)
    symbols.sort()
    with open(output, "w") as output_fd:
        writer = csv.writer(output_fd, delimiter="\n")
        writer.writerow(symbols)


if __name__ == "__main__":
    """
    extract_tokens path/to/groundtruth.tsv [-o OUTPUT]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        default="tokens.txt",
        help="Output path of the tokens text file",
    )
    parser.add_argument("groundtruth", nargs=1, help="Ground truth TXT file")
    args = parser.parse_args()
    create_tokens(args.groundtruth[0], args.output)
