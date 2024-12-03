import os
import traceback
import sklearn.metrics
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)

from tqdm import tqdm
from datasets import load_dataset
from argparse import ArgumentParser
from pathlib import Path

from models import *


def process_protocol(enroll_wav_dict, test_wav_dict, proto_path, speaker_model):
    utt2model = {}
    pred_lst = []
    label_lst = []

    with open(proto_path) as in_file:
        lines = in_file.readlines()
        total_num = len(lines)
        for line in tqdm(lines, total=total_num):
            # try:
            enroll_id, test_id, lab = line.strip().split()
            if enroll_id not in utt2model:
                utt2model[enroll_id] = speaker_model(enroll_wav_dict[enroll_id]).cpu().numpy()
            enroll_emb = utt2model[enroll_id]

            if test_id not in utt2model:
                utt2model[test_id] = speaker_model(test_wav_dict[test_id]).cpu().numpy()
            test_emb = utt2model[test_id]

            cos_sim = np.dot(enroll_emb, test_emb.T) / (np.linalg.norm(enroll_emb) * np.linalg.norm(test_emb))
            # except Exception as e:
                # traceback.print_exc()
                # continue

            if lab == 'imp':
                label_lst.append(0)
            elif lab == 'tar':
                label_lst.append(1)
            else:
                raise ValueError
            pred_lst.append(cos_sim[0][0])

    eer = compute_eer(label=label_lst,
                      pred=pred_lst)

    return eer


def compute_eer(label, pred, positive_label=1):
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred, pos_label=positive_label)
    fnr = 1 - tpr

    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = (eer_1 + eer_2) / 2

    return eer


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--conversion-dir',
        type=str,
        help='Directory with converted audio. If not provided, the original audio will be used.',
    )
    parser.add_argument(
        '--model-name',
        default='WavLM-TDNN',
        choices=['WavLM-TDNN', 'wespeaker'],
        type=str,
        help='The speaker model. (Default: WavLM-TDNN)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    if args.model_name == 'WavLM-TDNN':
        speaker_model = WavLM_TDNN()
    elif args.model_name == 'wespeaker':
        speaker_model = WespeakerModel()
    else:
        raise ValueError('Unknown model name: {}'.format(args.model_name))

    ds = load_dataset('FluentaAI/cv-ru-test-speaker',
                      cache_dir='./data')

    enroll_wav_dict = {item['path'].split('.')[0]: item['array'] for item in ds['enrolls']['audio']}

    if args.conversion_dir:
        test_wav_dict = {}
        for wav in os.listdir(args.conversion_dir):
            wav_path = os.path.join(args.conversion_dir, wav)
            audio, sr = torchaudio.load(str(wav_path))
            if sr != 16000:
                audio = torchaudio.transforms.Resample(sr, speaker_model.sr)(audio)
            test_wav_dict.update({wav.split('.')[0]: audio.squeeze().numpy()})
    else:
        test_wav_dict = {item['path'].split('.')[0]: item['array'] for item in ds['tests']['audio']}

    _proto_path = './meta/cv-ru-test-speaker/eer_protocol.txt'

    res = process_protocol(enroll_wav_dict=enroll_wav_dict,
                           test_wav_dict=test_wav_dict,
                           proto_path=_proto_path,
                           speaker_model=speaker_model
                           )
    print('{} EER: {:.2f}'.format(args.model_name,
                                  res))
