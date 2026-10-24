"""Separate diagnostic entrypoint; leaves the baseline HTTP server unchanged."""
import json
import atom_server270_http as server
from stage_probe270 import StageProbe
from timeline270 import TimelineProbe

probes = []
timelines = []
original_observer = server.observe_decode_batches


def observe_with_probe(batcher):
    histogram = original_observer(batcher)
    probe = StageProbe()
    probe.install(batcher)
    probes.append(probe)
    timeline = TimelineProbe()
    timeline.install(batcher)
    timelines.append(timeline)
    return histogram


if __name__ == '__main__':
    server.observe_decode_batches = observe_with_probe
    try:
        server.main()
    finally:
        # server.main closes the producer before return, settling completion callbacks.
        for probe in probes:
            print('ATOM270_HOST_STAGE_TIMES ' + json.dumps(probe.snapshot()), flush=True)

        for timeline in timelines:
            print("ATOM270_HOST_TIMELINE " + json.dumps(timeline.snapshot()), flush=True)
