import os
import shutil


def mkdir(path, clean=True):
    if os.path.exists(path) and clean:
        shutil.rmtree(path)
    os.makedirs(path)
