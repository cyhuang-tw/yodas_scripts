import json
import numpy as np
from pathlib import Path


def filter_score(samples, percentages=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    scores = []
    for s in samples:
        scores.extend(s['confidences'])

    thresholds = np.quantile(scores, percentages)

    res = {}
    for p, t in zip(percentages, thresholds):
        res[p] = []
        for s in samples:
            if np.all(np.array(s['confidences']) >= t):
                res[p].append(s)
        print(f"Percentage {p}, threshold {t}: {len(res[p])} samples")

    return res


valid_utts = []
with open("data_reseg/lid_remaining.txt", 'r') as f:
    for line in f:
        valid_utts.append(line.strip())
valid_utts = set(valid_utts)
print(f"There are {len(valid_utts)} valid utts")

lang2samples = {}
with open("data_reseg/all.jsonl", 'r') as f:
    for line in f:
        sample = json.loads(line.strip())
        if sample['utt_id'] in valid_utts:
            lang = sample['lang'][1:-1]
            if lang not in lang2samples:
                lang2samples[lang] = []
            lang2samples[lang].append(sample)


outroot = "data_reseg/data"
# percentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
percentages = [0.15]
for lang, samples in lang2samples.items():
    print(f"-------- Processing {lang} with {len(samples)} samples --------")
    outdir = Path(outroot) / lang
    outdir.mkdir(parents=True, exist_ok=True)

    res = filter_score(samples, percentages)

    for p, s in res.items():
        with open(outdir / f"quantile_{p:.2f}.jsonl", 'w') as f:
            for sample in s:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
