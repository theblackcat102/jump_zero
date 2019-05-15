import resource
import sys
import signal
import time

def memory_limit(soft_limit=-1):
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if soft_limit != -1:
        soft = soft_limit
    else:
        soft = int(get_memory() * 1024 / 2)
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    resource.setrlimit(resource.RLIMIT_RSS, (soft//10, hard))


def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

def time_expired(n, stack):
    raise SystemExit('(time ran out)')

def time_limit(time=1200):
    signal.signal(signal.SIGXCPU, time_expired)
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    resource.setrlimit(resource.RLIMIT_CPU, (time, hard))
