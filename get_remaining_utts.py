import json
from pathlib import Path


def is_valid(text: str) -> bool:
    invalid_syms = ["<s>", "</s>", "#0"]
    for sym in invalid_syms:
        if text.find(sym) != -1:
            return False
    return True


if __name__ == "__main__":
    quantile = "0.9"
    outfile = f"data_reseg/data/remaining_utts_{quantile}.txt"

    fout = open(outfile, 'w')
    for file in Path("data_reseg/data").glob(f"*/quantile_{quantile}.jsonl"):
        with open(file, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                utt_id = sample['utt_id']

                if is_valid(sample['text_with_time']) and is_valid(sample['asr_text']) and is_valid(sample['prev_text']):
                    fout.write(f"{utt_id}\n")
                else:
                    print(f"Invalid: {utt_id}")
                    print(f"Text with time: {sample['text_with_time']}")
                    print(f"ASR text: {sample['asr_text']}")
                    print(f"Prev text: {sample['prev_text']}")
    fout.close()
