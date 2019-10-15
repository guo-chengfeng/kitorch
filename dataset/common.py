# codes are from paddlepaddle
from __future__ import print_function

import requests
import hashlib
import os
import errno
import shutil
import six
import sys
import numpy as np

__all__ = [
    'DATA_HOME',
    'download',
    'md5file',
]

DATA_HOME = os.path.expanduser('~/.cache/minitotch/dataset')


# When running unit tests, there could be multiple processes that
# trying to create DATA_HOME directory simultaneously, so we cannot
# use a if condition to check for the existence of the directory;
# instead, we use the filesystem as the synchronization mechanism by
# catching returned errors.
def must_mkdirs(path):
    try:
        os.makedirs(DATA_HOME)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


must_mkdirs(DATA_HOME)


def md5file(fname):
    hash_md5 = hashlib.md5()
    f = open(fname, "rb")
    for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()


def download(url, module_name, md5sum, save_name=None):
    dirname = os.path.join(DATA_HOME, module_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = os.path.join(dirname,
                            url.split('/')[-1]
                            if save_name is None else save_name)

    retry = 0
    retry_limit = 3
    while not (os.path.exists(filename) and md5file(filename) == md5sum):
        if os.path.exists(filename):
            sys.stderr.write("file %s  md5 %s" % (md5file(filename), md5sum))
        if retry < retry_limit:
            retry += 1
        else:
            raise RuntimeError("Cannot download {0} within retry limit {1}".
                               format(url, retry_limit))
        sys.stderr.write("Cache file %s not found, downloading %s" %
                         (filename, url))
        r = requests.get(url, stream=True)
        total_length = r.headers.get('content-length')

        if total_length is None:
            with open(filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        else:
            with open(filename, 'wb') as f:
                dl = 0
                total_length = int(total_length)
                for data in r.iter_content(chunk_size=4096):
                    if six.PY2:
                        data = six.b(data)
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stderr.write("\r[%s%s]" % ('=' * done,
                                                   ' ' * (50 - done)))
                    sys.stdout.flush()
    sys.stderr.write("\n")
    sys.stdout.flush()
    return filename


def batch(data, batch_size, shuffle=True):
    num_data = len(data)
    if shuffle:
        idx = list(range(num_data))
        np.random.shuffle(idx)

    num_iter = num_data // batch_size

    for i in range(num_iter):
        start = i * batch_size
        if num_data - (i + 1) * batch_size < batch_size:
            end = num_data
        else:
            end = (i + 1) * batch_size

        yield data[idx[start:end], :]
