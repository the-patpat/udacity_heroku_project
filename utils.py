"""
Some basic utilities
"""
import os
import errno
import logging 

def mkdir_p(path):
    """https://stackoverflow.com/questions/20666764/python-logging-how-to-ensure-logfile-directory-is-created"""
    """http://stackoverflow.com/a/600612/190597 (tzot)"""
    try:
        os.makedirs(path, exist_ok=True)  # Python>3.2
    except TypeError:
        try:
            os.makedirs(path)
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise

class MakeFileHandler(logging.FileHandler):
    """https://stackoverflow.com/questions/20666764/python-logging-how-to-ensure-logfile-directory-is-created"""
    def __init__(self, filename, mode='a', encoding=None, delay=0):
        mkdir_p(os.path.dirname(filename))
        logging.FileHandler.__init__(self, filename, mode, encoding, delay)