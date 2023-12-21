import os
import shutil

def remove_ccramic_caches(directory):
    """
    Remove any ccramic caches from the specified directory
    """
    if os.access(directory, os.R_OK):
        # TODO: establish cleaning the tmp dir for any sub directory that has ccramic cache in it
        subdirs = [x[0] for x in os.walk(directory) if 'ccramic_cache' in x[0]]
        # remove any parent directory that has a ccramic cache in it
        for dir in subdirs:
            if os.access(os.path.dirname(dir), os.R_OK) and os.access(dir, os.R_OK):
                shutil.rmtree(os.path.dirname(dir))
