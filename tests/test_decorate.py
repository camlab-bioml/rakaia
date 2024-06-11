from ccramic.utils.decorator import time_taken_callback

def test_time_taken(capfd):

    def add(x, y):
        return x + y

    assert add(2, 3) == 5
    out = time_taken_callback(add, x=2, y=3)()
    assert out == 5
    captured = capfd.readouterr()
    assert 'Total time taken' in captured.out

def test_time_taken_empty(capfd):

    def add(x, y):
        return x + y

    assert add(2, 3) == 5
    out = time_taken_callback(add, show_output=False, x=2, y=3)()
    assert out == 5
    captured = capfd.readouterr()
    assert not captured.out
