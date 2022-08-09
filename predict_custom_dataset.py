import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import os

import wasr.models as models
from predict_single import predict

BATCH_SIZE = 12
ARCHITECTURE = 'wasr_resnet101_imu'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="WaSR Network MaSTr1325 Inference")
    parser.add_argument("--image_folder", type=str,
                        help="Path to the image to run inference on.")
    parser.add_argument("--output_folder", type=str,
                        help="Path to the file, where the output prediction will be saved.")
    parser.add_argument("--dataset", type=str,
                        help="Name of the dataset used in order to use the correct directory structure (MODS, TODO:list of choices).")                    
    parser.add_argument("--imu_mask", type=str, default=None,
                        help="Path to the corresponding IMU mask (if needed by the model).")
    parser.add_argument("--architecture", type=str, choices=models.model_list, default=ARCHITECTURE,
                        help="Model architecture.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the model weights or a model checkpoint.")
    return parser.parse_args()


def predict_mods(args):
    count = 0
    for root, dirs, files in os.walk(args.image_folder):
        for dir in dirs:
            for frame in os.listdir(os.path.join(root, dir, "imus")): #iterate through imus to get only those frames that have a coresponding imu image
               args.image = os.path.join(root, dir, "frames", frame)
               args.imu_mask = os.path.join(root, dir, "imus", frame)
               args.output = os.path.join(args.output_folder, dir, frame)
               predict(args)
               count += 1
               if count == 10:
                return
    


def main():
    args = get_arguments()
    print(args)
    if args.dataset == "MODS":
        predict_mods(args)


if __name__ == '__main__':
    main()
