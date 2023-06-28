import ray

from wsilearn.utils.docker_utils import count_docker_cpus
from wsilearn.utils.gpu_utils import count_gpus
from logging import DEBUG

def init_ray_on_docker(n_cpus=None, n_gpus=None):
    if n_cpus is None:
        n_cpus = max(2, count_docker_cpus())
    if n_gpus is None:
        try:
            n_gpus = count_gpus()
        except Exception as ex:
            print('counting gpus failed with %s' % str(ex))
            n_gpus = 0
    if not ray.is_initialized():
        print('initializing ray with %d cpus and %d gpus' % (n_cpus, n_gpus))
        ray.init(num_cpus=n_cpus, num_gpus=n_gpus, #include_webui=False,
                 # configure_logging=False,
                 logging_level=DEBUG,
                 # log_to_driver=True
                 )
    return n_cpus, n_gpus

@ray.remote(num_cpus=1, num_gpus=1)
class RayCallWrapper1(object):
    def __init__(self, calee):
        self.callee = calee

    def __call__(self, *args, **kwargs):
        return self.callee(*args, **kwargs)

@ray.remote(num_cpus=2, num_gpus=2)
class RayCallWrapper2(object):
    def __init__(self, calee):
        self.callee = calee

    def __call__(self, *args, **kwargs):
        return self.callee(*args, **kwargs)

class CallRayWrapper(object):
    def __init__(self, callee, n_gpus=1):
        if n_gpus==1:
            self.rcw = RayCallWrapper1.remote(callee)
        elif n_gpus==2:
            self.rcw = RayCallWrapper2.remote(callee)
        else:
            raise ValueError('todo %d gpus...' % n_gpus)

    def __call__(self, *args, **kwargs):
        call_id = self.rcw.__call__.remote(*args, **kwargs)
        val = ray.get(call_id)
        return val

class RayDummyTest(object):
    def __call__(self, *args):
        print(args)

if __name__ == '__main__':
    init_ray_on_docker(4, 1)
    rdt = RayDummyTest()
    CallRayWrapper(rdt)

    rdt('test')