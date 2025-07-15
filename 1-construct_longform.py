import json
import emoji
from tqdm import tqdm
from pathlib import Path
from utils import (
    Utterance,
    LongUtterance,
    generate_long_utterances,
)


def clean_text(text: str) -> str:
    # remove special spaces
    text = " ".join(text.strip().split())

    # remove emojis
    text = emoji.replace_emoji(text, replace="")

    return text


def construct_data_from_file(file):
    audio_dir = Path(file).parent.parent / 'audio'
    lang = Path(file).resolve().parent.parent.name[:2]

    video2text = {}
    json_obj_lst = json.loads(open(file, 'r').read())
    for json_obj in json_obj_lst:
        video_id = json_obj['audio_id']
        short_utts = []
        for k, v in sorted(json_obj['text'].items()):
            v = clean_text(v)

            fields = k.split('-')
            start_timestamp = float(fields[-2]) / 100
            end_timestamp = float(fields[-1]) / 100
            short_utts.append(
                Utterance(
                    utt_id=k,
                    wav_id=video_id,
                    wav_path=str(audio_dir / f'{video_id}.flac'),
                    start_time=start_timestamp,
                    end_time=end_timestamp,
                    lang="<" + lang + ">",
                    task="<asr>",
                    text=v,
                    asr_text=v,
                )
            )

        long_utts = generate_long_utterances(short_utts)
        video2text[video_id] = long_utts

    return video2text


def construct_all(in_dir, out_file):
    with open(out_file, 'w') as fout:
        for file in tqdm(list(Path(in_dir).glob("*/text/*.json"))):
            video2text = construct_data_from_file(file)
            for video_id, long_utts in video2text.items():
                for long_utt in long_utts:
                    fout.write(json.dumps(long_utt.__dict__) + '\n')


if __name__ == "__main__":
    construct_all("/work/hdd/bbjs/shared/corpora/yodas2/data", "/work/hdd/bbjs/peng6/yodas/data/all.jsonl")
