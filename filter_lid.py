import json
import numpy as np
from tqdm import tqdm
from utils import TO_ISO_LANGUAGE_CODE


def norm_speech_lang(lang):
    lang_mappings = {
        "iw": "he",
        "jw": "jv",
    }
    if lang in lang_mappings:
        lang = lang_mappings[lang]
    
    if lang in TO_ISO_LANGUAGE_CODE:
        return TO_ISO_LANGUAGE_CODE[lang]

    return lang


def norm_text_lang(lang):
    if len(lang) == 2 and lang in TO_ISO_LANGUAGE_CODE:
        return TO_ISO_LANGUAGE_CODE[lang]

    return lang


if __name__ == "__main__":
    in_file = "data_reseg/lid.jsonl"
    out_file = "data_reseg/lid_remaining.txt"

    with open(in_file, 'r') as fin, open(out_file, 'w') as fout:
        for line in tqdm(fin):
            sample = json.loads(line.strip())
            lang = sample['lang']
            speech_lang = norm_speech_lang(sample['speech_pred'])
            text_lang = norm_text_lang(sample['text_pred'])
            prevtext_lang = norm_text_lang(sample['prev_text_pred'])

            if prevtext_lang == "<na>":
                prevtext_lang = text_lang

            if lang == text_lang and lang == prevtext_lang and lang == speech_lang:
                fout.write(sample['utt_id'] + '\n')
