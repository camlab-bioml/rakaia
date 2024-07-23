import dash_uploader as du

class DashUploaderFileReader:
    """
    A basic reader for the dash uploader component. Reads an upload status object and creates a list
    of input files based on the progress status requested (default is 1)

    :param uploader: `dash_uploader` UploadStatus for a specific dash_uploader` component
    :param progress: Float specifying the percentage of file contents to be read before the filenames are to be parsed.
    By default, all files should be read before parsing.

    :return: None
    """
    def __init__(self, uploader: du.UploadStatus, progress: float = 1.0):
        self.uploader = uploader
        self.progress = progress

    def return_filenames(self):
        """
        Get a list of filenames parsed from a single `dash_uploader` component HTTP request

        :return: filename list
        """
        filenames = [str(x) for x in self.uploader.uploaded_files]
        # IMP: ensure that the progress is up to 100% in the float before beginning to process
        if filenames and float(self.uploader.progress) == self.progress:
            return filenames
        return None
