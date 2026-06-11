import logging
from datetime import datetime
import pytz

class TimezoneFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, tzinfo=pytz.UTC):
        super().__init__(fmt, datefmt)
        self.tzinfo = tzinfo

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, self.tzinfo)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")

def setup_logging(log_file, level, include_host=False):
    shanghai_tz = pytz.timezone("Asia/Shanghai")
    if include_host:
        import socket
        hostname = socket.gethostname()
        # formatter = logging.Formatter(f'%(asctime)s |  {hostname} | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
        formatter = TimezoneFormatter('%(asctime)s - %(hostname) - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S", tzinfo=shanghai_tz)
    else:
        # formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
        formatter = TimezoneFormatter('%(asctime)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S", tzinfo=shanghai_tz)

    logging.root.setLevel(level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)

