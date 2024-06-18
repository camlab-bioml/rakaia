import time
from ccramic.io.session import create_download_dir

class DownloadDirGenerator:
    """
    Decorate a callback function to create a download directory for file export
    Assumes that the destination directory is passed as keyword `dest_dir` or as the first positional argument
    to the wrapped function
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

def time_taken_callback(func, show_output: bool=True, *args, **kwargs):
    """
    Will print the execution time of the callback
    """
    def wrapper_function():
        begin = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        if show_output:
            print("Total time taken in : ", func.__name__, (end - begin))
        return result
    return wrapper_function
