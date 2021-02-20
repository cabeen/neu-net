#!/usr/bin/env python
################################################################################
#
#  A program for running the training phase
#
#  Author: Ryan Cabeen
#
################################################################################

'''
Train the model
'''

import os, sys, argparse, glob
from argparse import ArgumentDefaultsHelpFormatter as formatter

import nibabel as nib
import unetseg 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=formatter)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('--images', type=str, required=True, help='the training images directory')
    required.add_argument('--masks', type=str, required=True, help='the training mask directory')
    required.add_argument('--output', type=str, required=True, help='the output Directory')
    optional.add_argument('--labels', type=int, default=1, help='the number of output segmentation labels')
    optional.add_argument('--augment', type=int, default=0, help='augment the dataset this many times')
    optional.add_argument('--raw', action="store_true", default=False, help='use the raw intensities as input')
    optional.add_argument('--largest', action="store_true", default=False, help='keep the largest connected component')
    optional.add_argument('--init', type=str, help='the initial model (not required)')
    optional.add_argument('--epochs', type=int, default=40, help='number of training epoch')
    optional.add_argument('--rate', type=float, default=0.0001, help='the learning rate')
    optional.add_argument('--rescale', type=int, default=256, help='the size of the u-net input image')
    optional.add_argument('--kernel', type=int, default=16, help='the convolution kernel size')
    optional.add_argument('--batches', type=int, default=20, help='the number of batches')
    parser._action_groups.append(optional)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    
    for check in [args.masks, args.images]:
        if not os.path.exists(check): 
            print("Invalid input: %s" % check)
            sys.exit(2)

    args.channels = 1
    fn = glob.glob("%s/**.nii.gz" % args.images)[0]
    img = nib.load(fn)
    if len(img.shape) == 4:
      args.channels = img.shape[3]
    print("detected channels: %d" % args.channels)

    unetseg.train_main(unetseg.Settings(vars(args)), args.init, args.images, args.masks, args.output)

################################################################################
# End
################################################################################
