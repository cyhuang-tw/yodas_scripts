import json
import numpy as np
from pathlib import Path


def get_duration(file):
    res = 0
    with open(file, 'r') as f:
        for line in f:
            sample = json.loads(line.strip())
            res += sample['end_time'] - sample['start_time']

    return res / 3600


if __name__ == "__main__":
    root = Path("data_reseg/data")
    # percentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    percentages = [0.15]
    stats = {}
    for percentage in percentages:
        for file in root.glob(f"*/quantile_{percentage:.2f}.jsonl"):
            lang = file.parent.name
            duration = get_duration(file)

            if lang not in stats:
                stats[lang] = []

            stats[lang].append(duration)

    with open(f"data_reseg/data/durations.tsv", 'w') as fout:
        fout.write("lang\t" + "\t".join([f"quantile_{p:.2f}" for p in percentages]) + "\n")
        for lang, durations in sorted(stats.items(), key=lambda x: x[0], reverse=True):
            fout.write(lang + "\t" + "\t".join([f"{d:.2f}" for d in durations]) + "\n")
