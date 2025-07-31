import os
import re
import argparse
import json
import tempfile
import torch
import librosa
import emoji
import chinese_converter
from pathlib import Path
from typing import List
from tqdm import tqdm
# from espnet2.bin.s2t_ctc_align import CTCSegmentation
from s2t_ctc_align import CTCSegmentation
from espnet_model_zoo.downloader import ModelDownloader
from utils import TO_ISO_LANGUAGE_CODE


owsm_langs = [
    "abk",
    "afr",
    "amh",
    "ara",
    "asm",
    "ast",
    "aze",
    "bak",
    "bas",
    "bel",
    "ben",
    "bos",
    "bre",
    "bul",
    "cat",
    "ceb",
    "ces",
    "chv",
    "ckb",
    "cmn",
    "cnh",
    "cym",
    "dan",
    "deu",
    "dgd",
    "div",
    "ell",
    "eng",
    "epo",
    "est",
    "eus",
    "fas",
    "fil",
    "fin",
    "fra",
    "frr",
    "ful",
    "gle",
    "glg",
    "grn",
    "guj",
    "hat",
    "hau",
    "heb",
    "hin",
    "hrv",
    "hsb",
    "hun",
    "hye",
    "ibo",
    "ina",
    "ind",
    "isl",
    "ita",
    "jav",
    "jpn",
    "kab",
    "kam",
    "kan",
    "kat",
    "kaz",
    "kea",
    "khm",
    "kin",
    "kir",
    "kmr",
    "kor",
    "lao",
    "lav",
    "lga",
    "lin",
    "lit",
    "ltz",
    "lug",
    "luo",
    "mal",
    "mar",
    "mas",
    "mdf",
    "mhr",
    "mkd",
    "mlt",
    "mon",
    "mri",
    "mrj",
    "mya",
    "myv",
    "nan",
    "nep",
    "nld",
    "nno",
    "nob",
    "npi",
    "nso",
    "nya",
    "oci",
    "ori",
    "orm",
    "ory",
    "pan",
    "pol",
    "por",
    "pus",
    "quy",
    "roh",
    "ron",
    "rus",
    "sah",
    "sat",
    "sin",
    "skr",
    "slk",
    "slv",
    "sna",
    "snd",
    "som",
    "sot",
    "spa",
    "srd",
    "srp",
    "sun",
    "swa",
    "swe",
    "swh",
    "tam",
    "tat",
    "tel",
    "tgk",
    "tgl",
    "tha",
    "tig",
    "tir",
    "tok",
    "tpi",
    "tsn",
    "tuk",
    "tur",
    "twi",
    "uig",
    "ukr",
    "umb",
    "urd",
    "uzb",
    "vie",
    "vot",
    "wol",
    "xho",
    "yor",
    "yue",
    "zho",
    "zul",
]


def clean_text(text: str) -> str:
    # remove special spaces
    text = " ".join(text.strip().split())

    # remove emojis
    text = emoji.replace_emoji(text, replace="")

    # remove [xx] or (xx)
    text = re.sub("[\(\[].*?[\)\]]", "", text)

    text = " ".join(text.strip().split())

    return text


def align_audio(text, wav_path, aligner):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        speech = librosa.load(wav_path, sr=16000)[0]
        res = aligner(
            speech,
            text
        )

    output = []
    start_err = 0.
    end_err = 0.
    for utt_id, (start_time, end_time, confidence) in zip(res.utt_ids, res.segments):
        fields = utt_id.split('-')
        ori_start_time = float(fields[-2]) / 100
        ori_end_time = float(fields[-1]) / 100

        start_err += abs(start_time - ori_start_time)
        end_err += abs(end_time - ori_end_time)

        output.append((utt_id, start_time, end_time, confidence))

    return output, start_err / len(res.utt_ids), end_err / len(res.utt_ids)

def chunkize(file: Path, tmp_dir: Path) -> List[Path]:
    cmd = f"sox {file} {tmp_dir}/{file.stem}%5n.wav trim 0 1800 : newfile : restart"
    os.system(cmd)
    return sorted(list(tmp_dir.iterdir()))

def parse_text(kaldi_text: str) -> List[str]:
    texts = kaldi_text.split("\n")
    texts = [t for t in texts if len(t) > 0]
    chunks = [[]]
    count = 0
    for t in texts:
        idf = t.split(" ")[0]
        # st, ed are in ms, not sec.
        _, idx, st, ed = idf.strip().rsplit("-", 3)
        st = int(st)
        ed = int(ed)
        if st >= (count + 1) * 30 * 60 * 1000:
            count += 1
            assert len(chunks) == count
            chunks.append([])
        chunks[-1].append(t)
    ret = []
    for ch in chunks:
        new_text = "\n".join(ch) + "\n"
        ret.append(new_text)
    return ret

def align_audio_by_chunk(text, wav_path, aligner):
    tmp_dir = Path(tempfile.mkdtemp())
    chunk_files = chunkize(wav_path, tmp_dir)
    res_list = []
    text_chunks = parse_text(text)
    for sub_idx, sub_file in enumerate(chunk_files):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            speech = librosa.load(sub_file, sr=16000)[0]
            sub_res = aligner(speech, text_chunks[sub_idx])
            res_list.append(sub_res)
    utt_ids = [uid for sub in res_list for uid in sub.utt_ids]
    bias = 30 * 60 # 30 minutes in seconds
    segments = [(seg[0] + sub_idx * bias, seg[1] + sub_idx * bias, *seg[2:])
                for sub_idx, sub in enumerate(res_list)
                for seg in sub.segments]
    
    output = []
    start_err = 0.
    end_err = 0.
    for utt_id, (start_time, end_time, confidence) in zip(utt_ids, segments):
        fields = utt_id.split('-')
        # I think it should be / 1000 because we are converting ms to sec.
        ori_start_time = float(fields[-2]) / 1000
        ori_end_time = float(fields[-1]) / 1000

        start_err += abs(start_time - ori_start_time)
        end_err += abs(end_time - ori_end_time)

        output.append((utt_id, start_time, end_time, confidence))

    return output, start_err / len(utt_ids), end_err / len(utt_ids)


def find_full_filename(base_name: str, directory: str = '.'):
    """
    Search `directory` for a file whose name (without extension) matches `base_name`.
    Returns the first match (including its extension), or None if not found.
    """
    dir_path = Path(directory)
    # this will match files like "audio.wav", "audio.mp3", etc.
    match = next(dir_path.glob(f'{base_name}.*'), None)
    return match.name if match else None

def process_json(file, out_dir, aligner, root_dir):
    """Segment all audios in the input json file and 
    write the new jsonl file to the output directory.
    """
    fout = open(out_dir / (Path(file).stem + ".jsonl"), 'w')

    # audio_dir = Path(file).parent.parent / 'audio'
    audio_dir = Path(root_dir)
    json_id = Path(file).stem
    curr_dir = audio_dir / json_id

    json_obj_lst = json.loads(open(file, 'r').read())
    for json_obj in tqdm(json_obj_lst, disable=False, mininterval=30, maxinterval=300):
        audio_id = json_obj['audio_id']
        # wav_path = str(audio_dir / json_id / f'{audio_id}.flac')
        wav_name = find_full_filename(audio_id, curr_dir)
        assert wav_name is not None
        wav_path = curr_dir / wav_name
        if "error" in json_obj:
            tqdm.write(f"**** Skipping {audio_id} because error flag was found. ****")
            continue
        lang = re.search(r"<(.*?)>", json_obj["lang"]).group(1)
        aligner.lang_sym = json_obj["lang"]
        additional_text_cleaners = []
        if lang == "zho":
            additional_text_cleaners.append(chinese_converter.to_simplified)
            tqdm.write("**** Additional text cleaner: chinese_converter.to_simplified")

        kaldi_text = ""
        raw_texts = []
        cleaned_texts = []
        for k, v in sorted(json_obj['text'].items()):
            ori = str(v)    # save the original text

            for cleaner in additional_text_cleaners:
                v = cleaner(v)

            v = clean_text(v)

            if len(v) > 0:
                kaldi_text += f"{k} {v}\n"
                raw_texts.append(ori)
                cleaned_texts.append(v)

        if len(raw_texts) > 0:
            try:
                # segments, ave_start_err, ave_end_err = align_audio(kaldi_text, wav_path, aligner)
                segments, ave_start_err, ave_end_err = align_audio_by_chunk(kaldi_text, wav_path, aligner)
                

                sample = {
                    'audio_id': audio_id,
                    'wav_path': str(wav_path),
                    'ave_start_err': ave_start_err,
                    'ave_end_err': ave_end_err,
                    'utts': [
                        (utt_id, start_time, end_time, confidence, cleaned_text, raw_text) for (
                                utt_id, start_time, end_time, confidence
                            ), cleaned_text, raw_text in zip(segments, cleaned_texts, raw_texts)
                    ]
                }
                fout.write(json.dumps(sample, ensure_ascii=False) + '\n')

            except Exception as e:
                tqdm.write(f"Failed to process {audio_id}: {e}")

    fout.close()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_list", type=str, required=True)
    parser.add_argument("-r", "--root_dir", type=str, required=True)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    d = ModelDownloader()
    downloaded = d.download_and_unpack("espnet/owsm_ctc_v3.2_ft_1B")
    # downloaded = d.download_and_unpack("espnet/owsm_ctc_v3.1_1B")

    aligner = CTCSegmentation(
        **downloaded,
        fs=16000,
        ngpu=1,
        batch_size=32,    # batched parallel decoding; reduce it if your GPU memory is smaller
        kaldi_style_text=True,
        time_stamps="auto",
        lang_sym="<eng>",
        task_sym="<asr>",
        context_len_in_secs=2,  # left and right context in buffered decoding
    )

    with open(args.file_list, 'r') as fin:
        for file in tqdm(fin):
            file = file.strip()

            '''
            lang = Path(file).parent.parent.name[:2]
            assert TO_ISO_LANGUAGE_CODE[lang] in owsm_langs
            aligner.lang_sym = f"<{TO_ISO_LANGUAGE_CODE[lang]}>"
            tqdm.write("------ Current file: " + file)
            tqdm.write("------ Current lang: " + aligner.lang_sym)

            additional_text_cleaners = []
            if lang == "zh":
                additional_text_cleaners.append(chinese_converter.to_simplified)
                tqdm.write("**** Additional text cleaner: chinese_converter.to_simplified")
            '''

            out_dir = Path(file).parent.parent / 'text_reseg'
            out_dir.mkdir(exist_ok=True)

            process_json(file, out_dir, aligner, args.root_dir)
