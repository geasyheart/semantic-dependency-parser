# -*- coding: utf8 -*-
#
import logging

from tqdm import tqdm

from src.config import DATA_PATH


class TqdmHandler(logging.StreamHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


def get_logger():
    logger = logging.getLogger('semantic-dependency-parser')
    logger.setLevel(logging.INFO)
    fmt = '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s'

    file_handler = logging.FileHandler(filename=DATA_PATH.joinpath('run.log'))
    tqdm_handler = TqdmHandler()
    file_handler.setFormatter(logging.Formatter(fmt))
    tqdm_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)
    logger.addHandler(tqdm_handler)

    return logger


logger = get_logger()
