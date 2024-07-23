
import dash_uploader as du
from rakaia.io.readers import DashUploaderFileReader

def test_drag_drop_readers():

    status = du.UploadStatus(n_total = 3, uploaded_files=["file_1.txt", "file_2.txt", "file_3.txt"],
                             total_size_mb=100, uploaded_size_mb=100)
    reader = DashUploaderFileReader(status)
    assert reader.return_filenames() == ['file_1.txt', 'file_2.txt', 'file_3.txt']

    status_2 = du.UploadStatus(n_total=100, uploaded_files=['file_1.txt', 'file_2.txt', 'file_3.txt'],
                             total_size_mb=50, uploaded_size_mb=100)
    reader_2 = DashUploaderFileReader(status_2)
    assert reader_2.return_filenames() is None
