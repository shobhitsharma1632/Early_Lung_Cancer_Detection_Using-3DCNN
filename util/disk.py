import gzip 

from diskcache import FanoutCache, Disk
from diskcache.core import MODE_BINARY, io
from io import BytesIO
from cassandra.cqltypes import BytesType

import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
log.setLevel(logging.INFO)
# log.setLevel(logging.WARN)


class GzipDisk(Disk):
    def store(self, value, read, key=None):
        """
        Override from base calss diskcahce.Disk.
        
        :param value: value to convert
        :param bool read: True when value is file-like object
        :return: (size, mode, filename, value) tuple for Cache table
        """
        if type(value) is BytesType:
            if read:
                value = value.read()
                read = False
            
            str_io = BytesIO()
            gz_file = gzip.GzipFile(mode='wb', compresslevel=1, fileobj=str_io)

            # The commented code below is for python < 2.7.13 which does not 
            # support writing data > 2GB to a GzipFile object in one go.
            # for offset in range(0, len(value), 2**30):
            #     gz_file.write(value[offset:offset+2**30])
            gz_file.write(value)
            gz_file.close()

            value = str_io.getvalue()
        return super().store(value, read)
    

    def fetch(self, mode, filename, value, read):
        """
        Override from base class diskcache.Disk.
        
        :param int mode: value mode raw, binary, text, or pickle
        :param str filename: filename of corresponding value
        :param value: database value
        :param bool read: when True, return an open file handle
        :return: corresponding Python value
        """
        value = super().fetch(mode, filename, value, read)

        if mode == MODE_BINARY:
            str_io = BytesIO(value)
            gz_file = gzip.GzipFile(mode='rb', fileobj=str_io)
            
            value = gz_file.read()

        return value
    
def getCache(scope_str):
    return FanoutCache('data-unversioned/cache/' + scope_str,
                       disk=GzipDisk,
                       shards=64,
                       timeout=1,
                       size_limit=3e11,
                       )