import json
from tqdm import tqdm
from pathlib import Path
from utils import (
    Utterance,
    LongUtterance,
    generate_long_utterances,
    TO_ISO_LANGUAGE_CODE
)


def construct_data_from_file(file):
    lang = Path(file).resolve().parent.parent.name[:2]  # two-letter language code
    if lang == "iw":
        lang = "he"
    lang = TO_ISO_LANGUAGE_CODE[lang]   # convert to three-letter language code

    long_utts = []
    with open(file, 'r') as f:
        for line in f:
            recording = json.loads(line.strip())

            short_utts = []
            audio_id = recording['audio_id']
            wav_path = recording['wav_path']

            for utt_id, start_time, end_time, confidence, cleaned_text, raw_text in recording['utts']:
                short_utts.append(
                    Utterance(
                        utt_id=utt_id,
                        wav_id=audio_id,
                        wav_path=wav_path,
                        start_time=start_time,
                        end_time=end_time,
                        lang=f"<{lang}>",
                        task="<asr>",
                        text=cleaned_text,
                        asr_text=cleaned_text,
                        confidence=confidence,
                    )
                )
            
            long_utts.extend(generate_long_utterances(short_utts))

    return long_utts


if __name__ == "__main__":
    all_files = list(Path("/work/hdd/bbjs/shared/corpora/yodas2/data").glob("*/text_reseg/*.jsonl"))
    with open("data_reseg/all.jsonl", 'w') as fout:
        for file in tqdm(all_files):
            long_utts = construct_data_from_file(file)
            for long_utt in long_utts:
                fout.write(json.dumps(long_utt.__dict__, ensure_ascii=False) + '\n')
