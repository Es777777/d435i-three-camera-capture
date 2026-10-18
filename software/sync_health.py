"""Evidence from frame delivery, not a GPIO continuity detector."""
import numpy as np


def frame_health(samples, now_ns, fps):
    result = {}
    for name in ('ir1', 'ir2', 'rgb'):
        rows = list(samples.get(name, []))
        if not rows:
            result[name] = dict(state='no_frames', fps=0.)
            continue
        numbers, host, device = np.asarray(rows, dtype=float).T
        age = (now_ns-host[-1])/1e9
        intervals = np.diff(device)
        measured = (len(host)-1)*1e9/(host[-1]-host[0]) if len(host)>1 and host[-1]>host[0] else 0.
        result[name] = dict(state='stale' if age>max(1., 5/fps) else 'receiving',
                            fps=float(measured), age_s=float(age),
                            period_ms=float(np.median(intervals)) if len(intervals) else None,
                            period_std_ms=float(np.std(intervals)) if len(intervals) else None,
                            frame_gaps=int(np.maximum(np.diff(numbers)-1, 0).sum()))
    return dict(streams=result, cable_connected=None, exposure_synchronized=None,
                evidence='SDK exposes sync mode, not cable continuity or external pulse lock')
