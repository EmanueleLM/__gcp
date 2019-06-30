# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 15:13:16 2019

@author: Emanuele
"""


def write_tree(src):
    
    """
       src:string, must be in the format '/path/to/seeds_files/<name>_<number>.csv', 
        where each file contains as first element the list of seeds of the element
        you want to extract the vector parameters. <number> must start from 0;
    """
    
    import os, re
    num_files = len(next(os.walk(src))[2])  # dir is your directory path as string
    assert num_files > 0 
    filename = src.split('/')[-1].split('_')[0]  # extract filename
    
    for i in range(num_files):
        
        file_ = filename + str(i) + '.csv'
        f = open(file_, "r")
        list_seeds = re.search(r"\[([A-Za-z0-9_]+)\]", f.read())
        list_seeds = list_seeds.split(',')  