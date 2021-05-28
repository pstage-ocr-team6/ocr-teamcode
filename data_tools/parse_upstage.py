"""
input folder/
|_ batch_01/
    |_ images
    |_ data.json
|_ batch_02/
...

output folder/
|_ gt.txt
|_ images/
|_ academic_level.txt
"""


import argparse
import json
import os
import sys
import cv2
import glob
import numpy as np


def convert(args):

    # Read JSON files
    json_files = glob.glob(os.path.join(args.input, "batch*", args.json))
    json_files.sort()

    jsons = []
    for file in json_files:
        jsons += json.load(open(file, "r"))

    # Check JSON data validity
    assert len(jsons) > 0  # list of dicts

    # Create output directory if not exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(os.path.join(args.output, "images")):
        os.makedirs(os.path.join(args.output, "images"))
    with open(os.path.join(args.output, "gt.txt"), "w") as f:
        f.write("")
    if args.academic_level:
        with open(os.path.join(args.output, "academic_level.txt"), "w") as f:
            f.write("")

    image_index = args.starting_index
    for file in jsons:
        parse_upstage_part(file, args, image_index)
        image_index += 1

    print("Parsing complete for {} images".format(image_index))


def parse_upstage_part(file, args, image_index):
    assert "formula_area" in file.keys()
    assert "formula" in file.keys()
    coords = file["formula_area"]
    assert all(x in coords for x in ["x1", "x2", "x3", "x4", "y1", "y2", "y3", "y4"])
    if all(
        len(coords[x]) != len(file["formula"]) in coords
        for x in ["x1", "x2", "x3", "x4", "y1", "y2", "y3", "y4"]
    ):
        sys.exit("ERROR with {}".format(file["filename"]))

    # Prepare image
    img = cv2.imread(os.path.join(args.input, file["filename"]))

    # Loop through formula roi
    for i in range(len(file["formula"])):
        formula_ext = os.path.splitext(file["filename"])[1]
        formula_filename = "{}_{}{}".format(image_index, i, formula_ext)
        formula_coord = [
            coords["x1"][i],
            coords["y1"][i],
            coords["x2"][i],
            coords["y2"][i],
            coords["x3"][i],
            coords["y3"][i],
            coords["x4"][i],
            coords["y4"][i],
        ]
        formula_char = file["formula"][i]["latex"]

        # Crop image with coords
        # transform = TPS_SpatialTransformerNetwork(
        #     F=20,  # number of fiducial points of TPS-STN
        #     I_size=img.shape[:2],
        #     I_r_size=img.shape[:2],
        #     I_channel_num=img.shape[2],
        # )
        xmin = min(
            [formula_coord[0], formula_coord[2], formula_coord[4], formula_coord[6]]
        )
        xmax = max(
            [formula_coord[0], formula_coord[2], formula_coord[4], formula_coord[6]]
        )
        ymin = min(
            [formula_coord[1], formula_coord[3], formula_coord[5], formula_coord[7]]
        )
        ymax = max(
            [formula_coord[1], formula_coord[3], formula_coord[5], formula_coord[7]]
        )
        cropped_img = img[ymin:ymax, xmin:xmax]
        cv2.imwrite(
            os.path.join(args.output, "images", formula_filename),
            cropped_img,
        )

        convert_recognition(formula_char, formula_filename)

        if args.academic_level:
            level = file["formula"][i]["academic_level"]
            create_academic_level(level, formula_filename)


def convert_recognition(chars, filename):
    if type(chars) == list:
        chars = " ".join(chars)
    GTstring = filename + "\t" + chars + "\n"
    outputPath = os.path.join(args.output, "gt.txt")
    with open(outputPath, "a") as f:
        f.write(GTstring)


def create_academic_level(level, filename):
    GTstring = filename + "\t" + str(level) + "\n"
    outputPath = os.path.join(args.output, "academic_level.txt")
    with open(outputPath, "a") as f:
        f.write(GTstring)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts JSON ground truth data to ICDAR15 format"
    )

    parser.add_argument(
        "-i", "--input", help="Folder with multiple batches", required=True
    )
    parser.add_argument("-o", "--output", help="Output folder name", required=True)
    parser.add_argument(
        "-j",
        "--json",
        help="Name of individual JSON file",
        required=False,
        default="data.json",
    )
    parser.add_argument(
        "-idx",
        "--starting_index",
        help="Image name starting index",
        default=1,
        required=False,
        type=int,
    )
    parser.add_argument(
        "-academic_level",
        help="Extract academic level from Upstage GT",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()

    convert(args=args)
