from evaluationmcu import gerwav
#! /usr/bin/env python3
import numpy as np
import time
#import chirplet as ch
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path

def reform_file(root_dir):
    #root_dir = 'D:/ICML2013Bird/'
    files = sorted(Path(root_dir).glob('*.wav'))
    index = 0
    classname = []
    for x in files:
        oldpath = os.path.join( root_dir , x)
        print("oldpath:", oldpath)
        x = str(x)
        label = x.split("\\")[-1]
        classname.append(label)
        print("classname:", classname[index])
        index = index + 1
        newaudiopath = root_dir + str(index) + '/'

        print("filenew:", newaudiopath)
        if not os.path.exists(newaudiopath):
            print("Can't find new files!")
            os.makedirs(newaudiopath)
        newpath = newaudiopath + label
        print("newpath:", newpath)
        shutil.move(oldpath,newpath)

    return classname


if __name__ == '__main__':
    root_dir = 'D:/ICML2013Bird/'
    #classname = reform_file(root_dir)
    #print("classname:", classname)
    gerwav(root_dir)