#!/usr/bin/env python
################################################################################
#
#  A program for running the validation phase
#
#  Author: Ryan Cabeen
#
################################################################################

'''
Validate the model
'''

import os, sys, argparse
from argparse import ArgumentDefaultsHelpFormatter as formatter

import unetseg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=formatter)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional.add_argument('--models', type=str, required=True, help='the training models directory')
    required.add_argument('--images', type=str, required=True, help='the validation image directory')
    required.add_argument('--masks', type=str, required=True, help='the validation mask directory')
    required.add_argument('--output', type=str, required=True, help='the output directory')
    parser._action_groups.append(optional)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    for check in [args.masks, args.images, args.models]:
        if not os.path.exists(check): 
            print("Invalid input: %s" % check)
            sys.exit(2)

    unetseg.validate_main(args.models, args.images, args.masks, args.output)

################################################################################
# End
################################################################################
