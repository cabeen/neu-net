#!/usr/bin/env python
################################################################################
#
#  A program for running the testing phase
#
#  Author: Ryan Cabeen
#
################################################################################

'''
Test the model
'''

import os, sys, argparse
from argparse import ArgumentDefaultsHelpFormatter as formatter

import unetseg

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=formatter)
    optional=parser._action_groups.pop()
    required=parser.add_argument_group('required arguments')

    required.add_argument('--images', type=str, required=True, help='the test image directory')
    required.add_argument('--masks', type=str, required=True, help='the test mask directory')
    required.add_argument('--model', type=str, required=True, help='the model to apply')
    required.add_argument('--output', type=str, required=True, help='the output directory')
    parser._action_groups.append(optional)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    for check in [args.masks, args.images, args.model]:
        if not os.path.exists(check):
            print("Invalid input: %s" % check)
            sys.exit(2)

    unetseg.test_main(args.model, args.images, args.masks, args.output) 

################################################################################
# End
################################################################################
