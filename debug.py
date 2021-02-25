import numpy as np
wlen = 3200
shift = 160
signal = np.random.random(3000)
if len(signal) < wlen:
    segs = np.concatenate([signal, np.zeros(wlen-len(signal))], axis=0)
    segs = np.expand_dims(segs, axis=0)
else:
    segs = []
    for st in range(0, int(len(signal)-wlen), int(shift)):
        seg = signal[st:st+wlen]
        segs.append(seg)
segs = np.array(segs)
print(segs.shape)
print(segs)