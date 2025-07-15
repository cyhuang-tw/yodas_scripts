import torch
import json
from collections import defaultdict
import librosa
import numpy as np
from pathlib import Path
import random
# from espnet2.bin.s2t_ctc_align import CTCSegmentation
from s2t_ctc_align import CTCSegmentation
from espnet_model_zoo.downloader import ModelDownloader
import whisper
from speechbrain.inference.classifiers import EncoderClassifier

from utils import TO_ISO_LANGUAGE_CODE
from ctc_seg import owsm_langs


def search_valid_audio(root, lang, lid_model, cnt=10):
    if lang == "he":
        lang = "iw"

    all_paths = list(Path(root).glob(f"{lang}*/audio/*.flac"))
    random.seed(42)
    random.shuffle(all_paths)
    paths = []
    for wav_path in all_paths:
        lang_pred = lid_model.classify_batch(
            torch.tensor(librosa.load(wav_path, sr=16000, offset=0.0, duration=30)[0], device="cuda")
        )[3][0].split(":")[0]

        if lang_pred == lang:
            paths.append(wav_path)
            if len(paths) == cnt:
                break
        else:
            print(wav_path, lang, lang_pred)

    return paths


def compare_time(whisper_model, owsm_model, wav_path, lang):
    wav_path = str(wav_path)

    whisper_res = whisper_model.transcribe(wav_path, language=lang)
    # print(whisper_res["text"])

    kaldi_text = ""
    for seg in whisper_res["segments"]:
        utt_id = str(seg['id'])
        kaldi_text += f"{utt_id} {seg['text']}\n"

    owsm_res = owsm_model(
        librosa.load(wav_path, sr=16000)[0],
        kaldi_text
    )

    start_err = 0.
    end_err = 0.
    for (start_time, end_time, confidence), whisper_seg in zip(owsm_res.segments, whisper_res["segments"]):
        start_err += abs(start_time - whisper_seg['start'])
        end_err += abs(end_time - whisper_seg['end'])

    return start_err / len(owsm_res.utt_ids), end_err / len(owsm_res.utt_ids)




# wav_path = "test/cPAMhrApF-k.wav"


# model = whisper.load_model("turbo")
# result = model.transcribe(wav_path)
# print(result["text"])

# kaldi_text = ""
# with open("test/whisper.txt", "w") as f:
#     for seg in result["segments"]:
#         utt_id = str(seg['id'])
#         f.write(f"{utt_id} {seg['start']} {seg['end']} {seg['text']}\n")
#         kaldi_text += f"{utt_id} {seg['text']}\n"


# d = ModelDownloader()
# downloaded = d.download_and_unpack("espnet/owsm_ctc_v3.2_ft_1B")
# # downloaded = d.download_and_unpack("espnet/owsm_ctc_v3.1_1B")

# aligner = CTCSegmentation(
#     **downloaded,
#     fs=16000,
#     ngpu=1,
#     batch_size=16,    # batched parallel decoding; reduce it if your GPU memory is smaller
#     kaldi_style_text=True,
#     time_stamps="auto",
#     lang_sym="<eng>",
#     task_sym="<asr>",
#     context_len_in_secs=2,  # left and right context in buffered decoding
# )

# res = aligner(
#     librosa.load(wav_path, sr=16000)[0],
#     kaldi_text
# )
# with open("test/owsm_ctc_3.2.txt", "w") as f:
#     f.write(str(res))



if __name__ == "__main__":
    whisper_model = whisper.load_model("turbo")

    d = ModelDownloader()
    downloaded = d.download_and_unpack("espnet/owsm_ctc_v3.2_ft_1B")
    # downloaded = d.download_and_unpack("espnet/owsm_ctc_v3.1_1B")

    aligner = CTCSegmentation(
        **downloaded,
        fs=16000,
        ngpu=1,
        batch_size=16,    # batched parallel decoding; reduce it if your GPU memory is smaller
        kaldi_style_text=True,
        time_stamps="auto",
        lang_sym="<eng>",
        task_sym="<asr>",
        context_len_in_secs=2,  # left and right context in buffered decoding
    )

    sb_model = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir="models/speechbrain",
        run_opts={"device": "cuda"}
    )

    root = "/work/hdd/bbjs/shared/corpora/yodas2/data"
    # languages = ["en", "es", "fr", "de", "nl", "it", "pt", "pl", "zh", "ko", "ja", "ru", "ro", "sk"]
    # languages = ["en", "ja", "ru", "it", "ro", "sk"]
    # languages = ["he"]
    languages = ["nl", "pl"]

    stats = defaultdict(list)

    for lang_label in languages:
        paths = search_valid_audio(root, lang_label, sb_model)

        assert TO_ISO_LANGUAGE_CODE[lang_label] in owsm_langs
        aligner.lang_sym = f"<{TO_ISO_LANGUAGE_CODE[lang_label]}>"

        for wav_path in paths:
            wav_path = str(wav_path)

            try:
                ave_start_err, ave_end_err = compare_time(whisper_model, aligner, wav_path, lang_label)

                stats[lang_label].append((wav_path, ave_start_err, ave_end_err))

                print(f"*******{lang_label}: {wav_path} - {ave_start_err}, {ave_end_err}")

            except Exception as e:
                print(f"Failed to process {wav_path}: {e}")

    with open("test/owsm3.2_vs_whisper.json", "w") as f:
        json.dump(stats, f, indent=4)

    print("=" * 50)
    # compute min, median of ave_start_err, ave_end_err for each language
    for lang_label, values in stats.items():
        start_errs = [v[1] for v in values]
        end_errs = [v[2] for v in values]

        print(f"{lang_label}: {min(start_errs):.2f}, {np.median(start_errs):.2f} - {min(end_errs):.2f}, {np.median(end_errs):.2f}")
