"""Module containing utility functions and classes for decorating nested session callbacks
"""

import time
from functools import wraps
from rakaia.io.session import create_download_dir

class DownloadDirGenerator:
    """
    Decorate a callback function to create a download directory for file export
    Assumes that the destination directory is passed as keyword `dest_dir` or as the first positional argument
    to the wrapped function

    :param dest_dir: Directory path to the desired output download directory.
    :return: result
    """
    def __init__(self, dest_dir=None):
        self.dest_dir = dest_dir
    def __call__(self, func):
        def inner_create_download_dir(*args, **kwargs):
            # create the temporary download directory, then pass the arg to the download function
            create_download_dir(self.dest_dir)
            if 'dest_dir' in kwargs:
                kwargs['dest_dir'] = self.dest_dir
            else:
                # should be the first positional argument if not provided as keyword
                args = list(args)
                args[0] = self.dest_dir
            result = func(*args, **kwargs)
            return result
        return inner_create_download_dir



def time_taken_callback(func=None, show_output=True, *decorator_args, **decorator_kwargs):
    """
    Decorator to print execution time of a callback/function.
    Supports both:
    @time_taken_callback
    def f(...):
    and
    time_taken_callback(f, x=1, y=2)()
    """

    if func is None: return None

    @wraps(func)
    def wrapper_function(*args, **kwargs):

        # merge decorator-time kwargs with call-time kwargs
        merged_kwargs = {**decorator_kwargs, **kwargs}

        begin = time.time()

        result = func(*decorator_args, *args, **merged_kwargs)

        end = time.time()

        if show_output:
            print(f"Total time taken in {func.__name__}: {end - begin:.4f} sec")

        return result

    return wrapper_function
