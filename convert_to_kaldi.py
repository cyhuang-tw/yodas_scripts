import json
from pathlib import Path


def is_valid(text: str) -> bool:
    invalid_syms = ["<s>", "</s>", "#0"]
    for sym in invalid_syms:
        if text.find(sym) != -1:
            return False
    return True


if __name__ == "__main__":
    filename = "quantile_0.0.jsonl"
    # outroot = "/work/hdd/bbjs/peng6/espnet-owsm-train-20240205/egs2/owsm_v4/s2t1/data/yodas0.00"
    outroot = "/work/nvme/bbjs/peng6/DeltaAI/espnet-owsm-ft/egs2/owsm_v4/s2t1/data/yodas0.00"

    f_text = open(f"{outroot}/text", 'w')
    f_textctc = open(f"{outroot}/text.ctc", 'w')
    f_textprev = open(f"{outroot}/text.prev", 'w')
    f_utt2spk = open(f"{outroot}/utt2spk", 'w')
    f_wavscp = open(f"{outroot}/wav.scp", 'w')
    f_segments = open(f"{outroot}/segments", 'w')

    for file in Path("data_reseg/data").glob(f"*/{filename}"):
        with open(file, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                utt_id = sample['utt_id']
                wav_id = sample['wav_id']

                if is_valid(sample['text_with_time']) and is_valid(sample['asr_text']) and is_valid(sample['prev_text']):
                    f_text.write(f"{utt_id} {sample['lang']}{sample['task']}{sample['text_with_time']}\n")
                    f_textctc.write(f"{utt_id} {sample['asr_text']}\n")
                    f_textprev.write(f"{utt_id} {sample['prev_text']}\n")
                    f_utt2spk.write(f"{utt_id} {utt_id}\n")
                    f_segments.write(f"{utt_id} {wav_id} {sample['start_time']} {sample['end_time']}\n")
                    f_wavscp.write(f"{wav_id} sox {sample['wav_path']} -t wav -r 16k -c 1 - |\n")
                else:
                    print(f"Invalid: {utt_id}")
                    print(f"Text with time: {sample['text_with_time']}")
                    print(f"ASR text: {sample['asr_text']}")
                    print(f"Prev text: {sample['prev_text']}")

    f_text.close()
    f_textctc.close()
    f_textprev.close()
    f_utt2spk.close()
    f_wavscp.close()
    f_segments.close()
