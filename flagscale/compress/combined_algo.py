import os
import sys
import copy
import pdb

def prepare_compress_methods(compress_cfg):
    recipes = copy.deepcopy(compress_cfg)
    return recipes