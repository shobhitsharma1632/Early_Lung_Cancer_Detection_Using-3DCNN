import collections 
import copy
import datetime
import gc
import time

import numpy as np

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
# log.setLevel(logging.INFO)
# log.setLevel(logging.WARN)

IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = collections.namedtuple('XyzTuple', ['x', 'y', 'z'])

def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1]
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    return XyzTuple(*coords_xyz)

def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    cri_a = np.round(cri_a)
    return IrcTuple(*(cri_a[2]), int(cri_a[1]), int(cri_a[0]))

def importstr(module_str, from_=None):
    """
    >>> importstr('os')
    <module 'os' from '.../os.pyc'>
    >>> importstr ('math', 'fabs')
    <built-in function fabs>
    """
    if from_ is None and ':' in module_str:
        module_str, from_ = module_str.rsplit(':')
    
    module = __import__(module_str)
    for sub_str in module_str.split('.')[1:]:
        module = getattr(module, sub_str)
    
    if from_:
        try:
            return getattr(module, from_)
        except:
            raise ImportError('{}.{}'.format(module_str, from_))
    return module 

def prhist(ary, prefix_str=None, **kwargs):
    if prefix_str is None:
        prefix_str = ''
    else:
        prefix_str += ' '
    
    count_ary, bins_ary = np.histogram(ary, **kwargs)
    for i in range(count_ary.shape[0]):
        print("{}{:-8.2f}".format(prefix_str, bins_ary[i]), "{:-10}".format(count_ary[i]))
    print("{}{:-8.2f}".format(prefix_str, bins_ary[-1]))

def enumerateWithEstimate(
        iter,
        desc_str,
        start_ndx=0,
        print_ndx=4,
        backoff=None,
        iter_len=None,
):
    """
    Enhanced enumerate function with progress tracking and time estimation.
    
    Provides the same iteration functionality as the built-in enumerate() function,
    but adds intelligent progress logging with estimated completion times. This is
    particularly useful for long-running loops where you need visibility into
    progress and performance metrics.
    
    Args:
        iter: The iterable object to enumerate over. Can be any iterable such as
            lists, tuples, generators, etc.
            
        desc_str (str): Human-readable description of the operation being performed.
            Used in log messages to identify the current process. Examples:
            "Training epoch 5", "Processing CT scans", "Validating model".
            
        start_ndx (int, optional): Number of initial iterations to exclude from
            timing calculations. Useful when early iterations have setup overhead
            (caching, initialization) that would skew average timing metrics.
            Defaults to 0.
            
        print_ndx (int, optional): Iteration number at which to begin progress
            logging. Must be >= start_ndx * backoff. Allows the timing to
            stabilize before reporting begins. Defaults to 4.
            
        backoff (int, optional): Multiplier for spacing between log messages.
            After each log message, the interval until the next message is
            multiplied by this factor to reduce log frequency over time.
            Auto-selected based on iter_len: 2 for small datasets (<= 1000),
            4 for larger datasets.
            
        iter_len (int, optional): Total number of items in the iterable.
            If not provided, will attempt to determine using len(iter).
            Required for accurate progress estimation.
    
    Yields:
        tuple: (index, item) pairs identical to enumerate(), where index starts
            from 0 and item is the current element from the iterable.
    
    Note:
        When start_ndx > 0, the displayed duration excludes time spent on
        skipped iterations. This should be considered when using timing
        data for performance analysis or scheduling.
    
    Example:
        >>> data = range(1000)
        >>> for i, item in enumerateWithEstimate(data, "Processing data"):
        ...     process(item)
        # Logs: "Processing data: 4/1000 items, 0.5s elapsed, ~2m remaining"
    """
    if iter_len is None:
        iter_len = len(iter)
    
    if backoff is None:
        backoff = 2
        while backoff ** 10 < iter_len:
            backoff *= 2
    
    # Ensure first log happens right after timing starts
    print_ndx = max(print_ndx, start_ndx + 1)

    log.warning("{} -----/{}, starting".format(desc_str, iter_len))
    start_ts = time.time()
    for (current_ndx, item) in enumerate(iter):
        yield (current_ndx, item)
        if current_ndx == print_ndx:
            duration_sec = ((time.time() - start_ts)
                            / (current_ndx - start_ndx + 1)
                            * (iter_len-start_ndx)
                            )
            
            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)

            log.info("{} {:-4}/{}, done at {}, {}".format(
                desc_str,
                current_ndx, 
                iter_len,
                str(done_dt).rsplit('.', 1)[0],    # Remove microseconds
                str(done_td).rsplit('.', 1)[0],    # Remove microseconds
            ))

            print_ndx *= backoff
        
        if current_ndx + 1 == start_ndx:
            start_ts = time.time()

    log.warning("{} -----/{}, all done at {}".format(desc_str, iter_len, str(datetime.datetime.fromtimestamp(start_ts)).rsplit('.', 1)[0])) 
