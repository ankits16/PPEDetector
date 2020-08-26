"""
All the exceptions observed in PPE detection work flow
"""
class Error(Exception):
    """Base class for other exceptions"""
    pass


class VideoFileNotFoundError(Error):
    """Raised when the video file is not found at provided location"""
    pass