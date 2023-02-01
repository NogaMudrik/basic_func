# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 07:58:02 2023

@author: noga mudrik
"""

import sys
import basic_functions as b_func

def output_help_to_file(filepath, func):

    with open(filepath,"w") as archi:
        t = sys.stdout
        sys.stdout = archi
        help(func)
        sys.stdout = t
        
sys.path.insert(0, '/path/to/application/app/folder')

output_help_to_file(r'description_text.txt', b_func)