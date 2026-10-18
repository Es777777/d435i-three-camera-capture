"""Audio-assisted timeline estimation. This does not measure camera exposure lock."""
import json
import wave
from pathlib import Path

import numpy as np
from scipy import signal


def read_audio(path):
    with wave.open(str(path), 'rb') as wav:
        if wav.getsampwidth()!=2:
            raise ValueError('Audio must be PCM16 WAV')
        rate, channels = wav.getframerate(), wav.getnchannels()
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), '<i2').reshape(-1,channels)
    return pcm.astype(np.float32).mean(axis=1)/32768., rate


def envelope(pcm, rate, hz=1000):
    block = rate//hz
    if rate % hz:
        pcm = signal.resample_poly(pcm, hz*48, rate)
        block = 48
    pcm = pcm[:len(pcm)//block*block]
    energy = np.sqrt(np.mean(pcm.reshape(-1,block)**2,axis=1))
    return np.log1p(energy*100).astype(float)


def estimate_audio_map(reference, other, hz=1000):
    """Return ref_seconds = scale * other_seconds + offset_seconds."""
    n = min(6000, len(other)//4, len(reference)//3)
    if n < 1000:
        raise ValueError('至少需要约 4 秒有共同声音的录音；建议 15 秒以上')
    points, scores = [], []
    for start in np.linspace(0, len(other)-n, 5).astype(int):
        pattern = other[start:start+n].copy()
        pattern -= pattern.mean()
        norm = np.linalg.norm(pattern)
        if norm < .01:
            continue
        cross = signal.correlate(reference, pattern, mode='valid', method='fft')
        sums = np.r_[0., np.cumsum(reference)]
        sq = np.r_[0., np.cumsum(reference**2)]
        variance = np.maximum(sq[n:]-sq[:-n]-(sums[n:]-sums[:-n])**2/n, 1e-12)
        corr = cross/(norm*np.sqrt(variance))
        peak = int(np.argmax(corr))
        score = float(corr[peak])
        rivals = corr.copy()
        rivals[max(0,peak-250):peak+251] = -1
        rival = float(rivals.max()) if len(rivals) else -1
        if score < .45 or score-rival < .08:
            continue
        points.append([(start+n/2)/hz, (peak+n/2)/hz])
        scores.append(score)
    if len(points)<2:
        raise ValueError('音频匹配不唯一或共同声音不足；使用视频共同事件手动对齐')
    points = np.asarray(points)
    # End windows may extend beyond the shared recording; reject those matches
    # using consensus across independent windows, not a relaxed residual limit.
    best = None
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            slope = (points[j,1]-points[i,1])/(points[j,0]-points[i,0])
            if abs(slope-1) > .005:
                continue
            intercept = points[i,1]-slope*points[i,0]
            mask = np.abs(points[:,1]-slope*points[:,0]-intercept) <= .015
            if best is None or mask.sum() > best.sum():
                best = mask
    if best is None or best.sum() < max(2, len(points)//2+1):
        raise ValueError('音频窗口无法形成一致时间映射')
    points = points[best]
    scores = np.asarray(scores)[best].tolist()
    scale, offset = np.polyfit(points[:,0],points[:,1],1)
    residual = points[:,1]-(scale*points[:,0]+offset)
    if abs(scale-1)>.005 or np.max(np.abs(residual))>.015:
        raise ValueError('音频漂移/匹配残差过大，不能自动合成；请用共同事件校正')
    return dict(scale=float(scale), offset_s=float(offset),
                residual_ms=(residual*1000).tolist(), scores=scores,
                matched_windows_s=points.tolist(),
                note='Audio delay includes acoustic propagation; does not certify exposure synchronization')


def audio_host_map(folder):
    rows = [json.loads(s) for s in (Path(folder)/'audio_blocks.jsonl').read_text(encoding='utf8').splitlines()]
    good = [r for r in rows if r.get('first_sample_host_ns') is not None and not r.get('status')]
    if len(good)<2:
        raise ValueError('缺少可靠音频 ADC 时间记录')
    summary = json.loads((Path(folder)/'audio.json').read_text(encoding='utf8'))
    if not summary.get('complete'):
        raise ValueError('音轨记录不完整：'+str(summary.get('error')))
    x = np.array([r['sample_start']/summary['rate'] for r in good])
    base = good[0]['first_sample_host_ns']/1e9
    y = np.array([r['first_sample_host_ns']/1e9-base for r in good])
    slope, offset = np.polyfit(x,y,1)
    residual = y-(slope*x+offset)
    if np.max(np.abs(residual))>.02 or abs(slope-1)>.005:
        raise ValueError('音频 ADC 与主机时钟映射不稳定')
    return dict(slope=float(slope), intercept_s=float(base+offset),
                residual_ms=float(np.std(residual)*1000))


def camera_host_map(records):
    x = np.array([r['device_ts_ms']/1000 for r in records])
    y = np.array([r['host_monotonic_ns']/1e9 for r in records])
    if len(x)<3 or np.any(np.diff(x)<=0):
        raise ValueError('视频时间戳不足或不单调')
    base_x, base_y = x[0], y[0]
    slope, offset = np.polyfit(x-base_x,y-base_y,1)
    residual = y-base_y-(slope*(x-base_x)+offset)
    return dict(slope=float(slope), intercept_s=float(base_y+offset-slope*base_x),
                callback_residual_ms=float(np.std(residual)*1000),
                note='USB callback latency is not measured exposure latency')


def aligned_audio(sessions, maps, start_s, end_s, output, cancel=lambda:False):
    """Keep three tracks as separate WAV files; resample onto the reference timebase."""
    rate_out = 48000
    for i, (folder, mapping) in enumerate(zip(sessions,maps)):
        pcm, rate = read_audio(Path(folder)/'audio.wav')
        count = max(0,int((end_s-start_s)*rate_out))
        with wave.open(str(Path(output)/f'aligned_audio_{i+1}.wav'),'wb') as wav:
            wav.setparams((1,2,rate_out,0,'NONE','not compressed'))
            for begin in range(0,count,rate_out):
                if cancel():
                    raise InterruptedError('Cancelled')
                t = start_s+np.arange(begin,min(count,begin+rate_out))/rate_out
                sample = (t-mapping['offset_s'])/mapping['scale']*rate
                # Only a one-second output chunk and its neighboring source samples are allocated.
                lo, hi = max(0,int(np.floor(sample.min()))-1), min(len(pcm),int(np.ceil(sample.max()))+2)
                value = np.interp(sample,np.arange(lo,hi),pcm[lo:hi],left=0,right=0) if hi>lo else np.zeros(len(t))
                wav.writeframes((np.clip(value,-1,1)*32767).astype('<i2').tobytes())
