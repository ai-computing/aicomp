"""ATOM cache writer for an exclusively owned block and synchronous runtime.

The caller supplies validated SDK-specific staging and BF16-bit encoding. This
backend does not claim asynchronous DMA or accept an unvalidated wire format.
"""
import numpy as np
import torch


class AtomUpload:
    def __init__(self, packet, target):
        host=packet.host
        if (host.device.type!='cpu' or host.dtype!=torch.bfloat16 or host.ndim!=4
                or host.shape[0]<2 or host.shape[0]%2 or any(n<1 for n in host.shape)):
            raise ValueError('expected CPU BF16 [K/V slots, positions, heads, dim]')
        if len(target.metas)!=host.shape[0] or len(target.staging)!=host.shape[0]:
            raise ValueError('cache slot count differs')
        for meta,buf in zip(target.metas,target.staging):
            shape=tuple(meta['shape'])
            if (len(shape)!=4 or buf.shape!=shape or buf.dtype!=np.uint16
                    or shape[1]!=host.shape[2] or shape[3]!=host.shape[3]
                    or type(target.block) is not int or not 0<=target.block<shape[0]
                    or type(target.start) is not int or not 0<=target.start<=shape[2]-host.shape[1]):
                raise ValueError('ATOM block geometry differs')
        self.packet,self.target=packet,target
        self.seen=set();self.closed=False;self.failed=False

    def stage(self,first,last):
        if self.closed or self.failed:raise RuntimeError('upload unavailable')
        if (type(first) is not int or type(last) is not int
                or not 0<=first<last<=self.packet.host.shape[0]
                or self.seen.intersection(range(first,last))):
            raise ValueError('invalid or repeated slots')
        t=self.target;n=self.packet.host.shape[1]
        try:
            bits=self.packet.host[first:last].permute(0,2,1,3).contiguous().view(torch.int16).numpy().view(np.uint16)
            encoded=t.encode(bits)
            if encoded.shape!=bits.shape or encoded.dtype!=np.uint16:
                raise ValueError('invalid encoded KV')
            for slot,data in zip(range(first,last),encoded):
                buf=t.staging[slot]
                buf[t.block,:,t.start:t.start+n]=data
                t.runtime.set_kv_cache_for_layer(t.metas[slot]['name'],t.block,t.start,n,buf)
            self.seen.update(range(first,last))
        except BaseException:
            self.failed=True
            raise

    def finish(self):
        if self.closed or self.failed:raise RuntimeError('upload unavailable')
        if len(self.seen)!=self.packet.host.shape[0]:raise ValueError('incomplete upload')
        return self.target.block

    def close(self):
        # Supported runtime writes are blocking. Caller retires a partially
        # written block on cancellation/failure and never exposes it to decode.
        self.closed=True;self.packet=None


def import_atom_span(blocks, lane, start, keys, values):
    """Adapter for optimus-rm AtomKVBlocks; same owner/thread as its scheduler."""
    from types import SimpleNamespace
    from opt_prime.accel.platforms.atom.batching import bf16_to_dlfloat
    if keys.shape!=values.shape or keys.ndim!=4:
        raise ValueError('expected equal [layers, heads, positions, dim] KV')
    host=torch.stack((keys,values),dim=1).flatten(0,1).permute(0,2,1,3).detach().to('cpu',torch.bfloat16).contiguous()
    target=SimpleNamespace(runtime=blocks.io_rt,metas=blocks.metas,staging=blocks._staging(),
        block=blocks.lane_block[lane],start=start,encode=bf16_to_dlfloat)
    upload=AtomUpload(SimpleNamespace(host=host),target)
    try:
        for first in range(0,host.shape[0],4):upload.stage(first,min(first+4,host.shape[0]))
        return upload.finish()
    finally:upload.close()
