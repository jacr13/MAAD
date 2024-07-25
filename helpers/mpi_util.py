import warnings


class FakeCommWorld(object):
    barred = False
    _bcast_data = None

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def Allreduce(self, local_var, global_var, op=None):
        if op == "SUM":
            global_var += local_var
        else:
            warnings.warn("We are using a fake instance of MPI.COMM_WORLD.Allreduce()")
        return global_var

    def Barrier(self, *args, **kwargs):
        self.barred = not self.barred

    def bcast(self, data):
        if data is None:
            return self._bcast_data
        else:
            self._bcast_data = data

    def allgather(self, data, root=None):
        return data

    def gather(self, data, root=None):
        return data

    def Abort(self):
        return


class FakeMPI(object):
    def __init__(self):
        warnings.warn(" We are using a fake instance of MPI!")
        self.COMM_WORLD = FakeCommWorld()
        self.SUM = "SUM"


try:
    from mpi4py import MPI
except ImportError:
    MPI = FakeMPI()
