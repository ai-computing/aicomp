"""Linux inherited-fd host KV slab for a synchronous, single-flight worker.

This is CPU shared memory, not device DMA. The controller must wait for the
producer's request-bound completion before reading and finish decode before
submitting another write. All array views must be discarded before close().
"""

import fcntl
import mmap
import os


class SharedKVSlab:
    def __init__(self, fd, shape, *, writable=False):
        import numpy as np

        if len(shape) != 4 or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError("positive cache/head/token/dimension shape required")
        self.shape = tuple(shape)
        self.nbytes = int(np.prod(shape)) * 4
        if os.fstat(fd).st_size != self.nbytes:
            raise ValueError("shared KV slab size mismatch")
        self.fd = os.dup(fd)
        try:
            self.mapping = mmap.mmap(
                self.fd,
                self.nbytes,
                access=mmap.ACCESS_WRITE if writable else mmap.ACCESS_READ,
            )
        except BaseException:
            os.close(self.fd)
            raise
        self.closed = False

    @classmethod
    def create(cls, shape):
        import math

        if len(shape) != 4 or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError("positive cache/head/token/dimension shape required")
        fd = os.memfd_create("native-pd-kv", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        try:
            os.ftruncate(fd, math.prod(shape) * 4)
            fcntl.fcntl(
                fd,
                fcntl.F_ADD_SEALS,
                fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL,
            )
            return cls(fd, shape)
        finally:
            os.close(fd)

    def view(self):
        import numpy as np

        if self.closed:
            raise RuntimeError("shared KV slab closed")
        return np.ndarray(self.shape, dtype=np.float32, buffer=self.mapping)

    def close(self):
        if not self.closed:
            self.mapping.close()
            os.close(self.fd)
            self.closed = True
