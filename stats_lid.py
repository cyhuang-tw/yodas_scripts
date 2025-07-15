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


class SampleStats:
    def __init__(self):
        self.durations = []
        self.langs = []
        self.text_langs = []
        self.prevtext_langs = []
        self.speech_langs = []

    def add(self, sample):
        # utt_id: AOpQHv-gvyU_000167435_000195682_fra_asr
        fields = sample['utt_id'].split('_')
        duration = (float(fields[-3]) - float(fields[-4])) / 1000
        self.durations.append(duration)
        self.langs.append(sample['lang'])   # three-letter code
        self.text_langs.append(norm_text_lang(sample['text_pred']))
        self.prevtext_langs.append(norm_text_lang(sample['prev_text_pred']))    # can be "<na>"
        self.speech_langs.append(norm_speech_lang(sample['speech_pred']))

    def compute(self):
        tot_hours = np.sum(self.durations) / 3600

        match_speech = 0.
        match_text = 0.
        match_all = 0.
        for dur, lang, text_lang, prevtext_lang, speech_lang in zip(self.durations, self.langs, self.text_langs, self.prevtext_langs, self.speech_langs):
            if prevtext_lang == "<na>":
                prevtext_lang = text_lang

            if lang == speech_lang:
                match_speech += dur
            
            if lang == text_lang and lang == prevtext_lang:
                match_text += dur

            if lang == text_lang and lang == prevtext_lang and lang == speech_lang:
                match_all += dur

        self.tot_hours = tot_hours
        self.match_speech = match_speech / 3600
        self.match_text = match_text / 3600
        self.match_all = match_all / 3600

        return self.tot_hours, self.match_speech, self.match_text, self.match_all


if __name__ == "__main__":
    in_file = "data_reseg/lid.jsonl"
    out_file = "data_reseg/lid_stats.csv"
    # in_file = "logdir/lid_reseg/lid.1000"
    # out_file = "tmp.txt"

    all_stats = SampleStats()
    lang2stats = {}
    with open(in_file, 'r') as fin:
        for line in tqdm(fin):
            sample = json.loads(line.strip())
            all_stats.add(sample)
            if sample['lang'] not in lang2stats:
                lang2stats[sample['lang']] = SampleStats()
            lang2stats[sample['lang']].add(sample)

    all_stats.compute()

    print(f"Total hours: {all_stats.tot_hours:.2f}")
    print(f"Speech match hours: {all_stats.match_speech:.2f}, percentage: {all_stats.match_speech / all_stats.tot_hours:.2%}")
    print(f"Text match hours: {all_stats.match_text:.2f}, percentage: {all_stats.match_text / all_stats.tot_hours:.2%}")
    print(f"All match hours: {all_stats.match_all:.2f}, percentage: {all_stats.match_all / all_stats.tot_hours:.2%}")
    print("-" * 80)


    for lang, stats in tqdm(lang2stats.items()):
        stats.compute()
    # Sort languages by total hours and print stats
    sorted_langs = sorted(lang2stats.items(), key=lambda x: x[1].tot_hours, reverse=True)
    for lang, stats in sorted_langs:
        print(f"{lang}: total={stats.tot_hours:.2f} hours, "
            f"speechmatch={stats.match_speech:.2f} hours ({stats.match_speech / stats.tot_hours:.2%}), "
            f"textmatch={stats.match_text:.2f} hours ({stats.match_text / stats.tot_hours:.2%}), "
            f"allmatch={stats.match_all:.2f} hours ({stats.match_all / stats.tot_hours:.2%})")


    if out_file:
        with open(out_file, 'w') as fout:
            fout.write("lang,total_hours,speechmatch_hours,speechmatch_percentage,textmatch_hours,textmatch_percentage,allmatch_hours,allmatch_percentage\n")
            fout.write(f"all,{all_stats.tot_hours:.2f},"
                    f"{all_stats.match_speech:.2f},{all_stats.match_speech / all_stats.tot_hours:.2%},"
                    f"{all_stats.match_text:.2f},{all_stats.match_text / all_stats.tot_hours:.2%},"
                    f"{all_stats.match_all:.2f},{all_stats.match_all / all_stats.tot_hours:.2%}\n")
            for lang, stats in sorted_langs:
                fout.write(f"{lang},{stats.tot_hours:.2f},"
                        f"{stats.match_speech:.2f},{stats.match_speech / stats.tot_hours:.2%},"
                        f"{stats.match_text:.2f},{stats.match_text / stats.tot_hours:.2%},"
                        f"{stats.match_all:.2f},{stats.match_all / stats.tot_hours:.2%}\n")
