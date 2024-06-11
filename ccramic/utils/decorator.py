import time
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
