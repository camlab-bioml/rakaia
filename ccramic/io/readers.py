import dash_uploader as du

class DashUploaderFileReader:
    """
    A basic reader for the dash uploader component. Reads an upload status object and creates a list
    of input files based on the progress status requested (default is 1)
    """
    def __init__(self, uploader: du.UploadStatus, progress: float = 1.0):
        self.uploader = uploader
        self.progress = progress

    def return_filenames(self):
        filenames = [str(x) for x in self.uploader.uploaded_files]
        # IMP: ensure that the progress is up to 100% in the float before beginning to process
        if filenames and float(self.uploader.progress) == self.progress:
            return filenames
        else:
            return None
