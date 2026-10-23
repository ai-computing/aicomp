"""Validated, bounded-by-coverage notifications for a reserved KV slab.

This tracks source writes only. A consumer must finish its own transfers before
publishing the cache or releasing the reservation; source completion is not a fence.
"""
from threading import Lock


class LayerProgress:
    def __init__(self, layers, start, end):
        if any(type(x) is not int for x in (layers,start,end)) or layers<=0 or not 0<=start<=end:
            raise ValueError('invalid layer-progress geometry')
        self.layers,self.start,self.end=layers,start,end
        self._coverage=[[] for _ in range(layers)]
        self._pending=[]
        self._closed=False
        self._lock=Lock()

    def publish(self, layer_start, layer_end, token_start, token_end):
        span=(layer_start,layer_end,token_start,token_end)
        with self._lock:
            if self._closed:
                raise ValueError('layer progress is closed')
            if (any(type(x) is not int for x in span)
                or not 0<=layer_start<layer_end<=self.layers
                or not self.start<=token_start<token_end<=self.end):
                raise ValueError('invalid layer-progress span')
            for layer in range(layer_start,layer_end):
                if any(token_start<b and a<token_end for a,b in self._coverage[layer]):
                    raise ValueError('overlapping layer-progress span')
            # Validate the complete message before changing any layer's coverage.
            for layer in range(layer_start,layer_end):
                self._coverage[layer].append((token_start,token_end))
            self._pending.append(span)

    def drain(self):
        with self._lock:
            pending,self._pending=self._pending,[]
            return pending

    def finish(self):
        with self._lock:
            if self._closed:
                raise ValueError('layer progress is closed')
            for intervals in self._coverage:
                position=self.start
                for start,end in sorted(intervals):
                    if start!=position:
                        raise ValueError('incomplete layer-progress coverage')
                    position=end
                if position!=self.end:
                    raise ValueError('incomplete layer-progress coverage')
            self._closed=True
