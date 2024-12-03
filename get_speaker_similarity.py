# inspired by https://github.com/alphacep/vosk-tts/blob/master/extra/tts-test/ru/eval_similarity.py
import traceback
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)

from datasets import load_dataset
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

from models import *


def process_protocol(enroll_wavs_dict, test_wavs_dict, proto_path, speaker_model):
    utt2model = {}
    scores = []
    with open(proto_path) as in_file:
        lines = in_file.readlines()
        total_num = len(lines)
        for line in tqdm(lines, total=total_num):
            try:
                test_id, enroll_id = line.strip().split()
                enroll_id = enroll_id.split('.')[0]
                test_id = test_id.split('.')[0]

                if enroll_id not in utt2model:
                    utt2model[enroll_id] = speaker_model(enroll_wavs_dict[enroll_id]).cpu().numpy()
                enroll_emb = utt2model[enroll_id]

                if test_id not in utt2model:
                    utt2model[test_id] = speaker_model(test_wavs_dict[test_id]).cpu().numpy()
                test_emb = utt2model[test_id]

                score = np.dot(enroll_emb, test_emb.T) / (np.linalg.norm(enroll_emb) * np.linalg.norm(test_emb))
                scores.append(score[0][0])
            except Exception as e:
                traceback.print_exc()
                continue
    nscores = np.array(scores)
    return np.mean(nscores), np.min(nscores)


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--conversion-dir",
        type=str,
        help="Directory with converted audio. If not provided, the original audio will be used.",
    )
    parser.add_argument(
        "--model-name",
        default="WavLM-TDNN",
        choices=["WavLM-TDNN", "wespeaker"],
        type=str,
        help="The speaker model. (Default: 'WavLM-TDNN')",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    if args.model_name == 'WavLM-TDNN':
        speaker_model = WavLM_TDNN()
    elif args.model_name == 'wespeaker':
        speaker_model = WespeakerModel()
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    ds = load_dataset("FluentaAI/cv-ru-test-speaker",
                      cache_dir='./data')
    enroll_wavs_dict = {l['path'].split('.')[0]: l['array'] for l in ds['tests']['audio']}
    if args.conversion_dir:
        test_wavs_dict = {}
        for l in list(Path(args.conversion_dir).rglob('*.wav')):
            audio, sr = torchaudio.load(str(l))
            if sr != 16000:
                audio = torchaudio.transforms.Resample(sr, speaker_model.sr)(audio)
            test_wavs_dict.update({l.stem: audio.numpy()})
    else:
        test_wavs_dict = {l['path'].split('.')[0]: l['array'] for l in ds['tests']['audio']}

    _proto_path = './eer_trials.txt'

    mean_score, min_score = process_protocol(enroll_wavs_dict=enroll_wavs_dict,
                           test_wavs_dict=test_wavs_dict,
                           proto_path=_proto_path,
                           speaker_model=speaker_model
                           )

    print(f"{args.model_name} mean cosine score {mean_score:.3f}"
          f" min cosine score {min_score:.3f}")