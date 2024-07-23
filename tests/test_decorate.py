from rakaia.utils.decorator import (
    time_taken_callback,
    DownloadDirGenerator)
import os
import tempfile

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

def test_download_dir_decorate():
    with tempfile.TemporaryDirectory() as tmpdirname:
        download_dir = os.path.join(tmpdirname, "gibdaiffafisdf", 'downloads')
        final_file = os.path.join(download_dir, "fake.txt")
        @DownloadDirGenerator(download_dir)
        def write_fake_text(dest_dir, filename="fake.txt"):
            f = open(str(os.path.join(dest_dir, filename)), "w")
            f.write("decorator test")
            f.close()
        write_fake_text("path", filename="fake.txt")
        assert os.path.isfile(final_file)
        if os.access(final_file, os.W_OK):
            os.remove(final_file)

        write_fake_text(dest_dir="path", filename="fake.txt")
        assert os.path.isfile(final_file)
        if os.access(final_file, os.W_OK):
            os.remove(final_file)
