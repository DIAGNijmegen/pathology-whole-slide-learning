import time
import sys

nvm_init = False
try:
    from pynvml import nvmlInit,nvmlDeviceGetHandleByIndex,nvmlDeviceGetMemoryInfo,nvmlDeviceGetCount
    nvm_init_failed = False
except:
    print('importing pynvml failed')
    nvm_init_failed = True
    print(sys.exc_info())

def _init_nmv():
    global nvm_init  # add this line!
    global nvm_init_failed
    if not nvm_init and not nvm_init_failed:
        print('initializing nvml...')
        nvmlInit()
        nvm_init = True


def gpu_mem(print_info=False):
    """ returns used, free and total gpu memory in GB """
    global nvm_init_failed
    if nvm_init_failed:
        return 0, 0, 0

    try:
        _init_nmv()

        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        # print("Total memory:", info.total)
        # print("Free memory:", info.free)
        # print("Used memory:", info.used)
        used, free, total = info.used/1e9, info.free/1e9, info.total/1e9
        if print_info:
            print('used: %.2f, free=%.2f, total=%.2f' % (used, free, total))
        return used, free, total
    except:
        print('failed using nvml:', sys.exc_info())
        nvm_init_failed = True
        return 0, 0, 0

def count_gpus():
    # print('counting gpus with nvml...')
    _init_nmv()
    return nvmlDeviceGetCount()

if __name__ == '__main__':
    start = time.time()
    used, free, total = gpu_mem()
    t = time.time()-start
    print('Used %.3fGB/%.3fGB, free: %.3GBf, time: %s' % (used, total, free, str(t)))

    n_gpus = count_gpus()
    print('%d gpus' % n_gpus)