# download model WESPEAKER from https://wenet.org.cn/downloads?models=wespeaker&version=voxblink2_samresnet34.zip and unzip it
# into ./models/wespeaker/wespeaker-SimAMResNet34 directory
# download model WAVLM from https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view?usp=sharing and copy it
# into ./models/UniSpeech directory
import torch
import torch.nn as nn
import wespeaker
import torchaudio
import utmos

from torchaudio.transforms import Resample
from UniSpeech.downstreams.speaker_verification.models.ecapa_tdnn import ECAPA_TDNN_SMALL

torch.hub.set_dir('./models')

WAVLM_PATH = './models/UniSpeech/wavlm_large_finetune.pth' # copy model checkpoint from UniSpeech here
WESPEAKER_PATH = './models/wespeaker/wespeaker-SimAMResNet34'


class WavLM_TDNN(nn.Module):
    def __init__(self, checkpoint_path=WAVLM_PATH):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None).to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict['model'], strict=False)
        self.model.eval()
        self.sr = self.model.sr

    @torch.no_grad()
    def forward(self, wav, sample_rate=16000):
        wav = torch.from_numpy(wav).unsqueeze(0).float()
        resample = Resample(orig_freq=sample_rate, new_freq=16000)
        wav = resample(wav)
        wav = wav.to(self.device)
        emb = self.model(wav)
        return emb


class WespeakerModel(nn.Module):
    def __init__(self, checkpoint_path=WESPEAKER_PATH):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = wespeaker.load_model_local(checkpoint_path)
        self.model.set_device(self.device)
        self.sr = self.model.resample_rate

    @torch.no_grad()
    def forward(self, wav, sample_rate=16000):
        wav = torch.from_numpy(wav).unsqueeze(0).float().to(self.device)
        return self.model.extract_embedding_from_pcm(wav, sample_rate).unsqueeze(0)

# class UtmosModel(nn.Module):
#     def __init__(self, checkpoint_path=WESPEAKER_PATH):
#         super().__init__()
#         self.model = wespeaker.load_model_local(checkpoint_path)
#
#     def forward(self, wav, sample_rate):
#         wav = torch.from_numpy(wav).unsqueeze(0).float().to(self.device)
#         return self.extract_embedding_from_pcm(wav, sample_rate)


if __name__ == '__main__':
    ################## UniSpeech ##################
    import numpy as np
    wavlm_model = WavLM_TDNN()
    print('WavLM TDNN  model loaded successfully')
    emb_1 = wavlm_model(torchaudio.load('./test/voxceleb1/hn8GyCJIfLM_0000012.wav')[0][0].numpy())
    emb_2 = wavlm_model(torchaudio.load('./test/voxceleb1/xTOk1Jz-F_g_0000015.wav')[0][0].numpy())
    cos_tar = np.dot(emb_1, emb_2.T) / (np.linalg.norm(emb_1) * np.linalg.norm(emb_2))
    cos_tar = cos_tar[0][0]
    emb_3 = wavlm_model(torchaudio.load('./test/voxceleb1/HXUqYaOwrxA_0000015.wav')[0][0].numpy())
    cos_imp = np.dot(emb_1, emb_3.T) / (np.linalg.norm(emb_1) * np.linalg.norm(emb_3))
    cos_imp = cos_imp[0][0]
    assert np.allclose([cos_tar,cos_imp], [0.5979, 0.0925], atol=1.e-4)
    ################### Wespeaker ###################
    wespeaker_model = WespeakerModel()
    print('Wespeaker model loaded successfully')
    emb_1 = wespeaker_model(torchaudio.load('./test/voxceleb1/hn8GyCJIfLM_0000012.wav')[0][0].numpy())
    emb_2 = wespeaker_model(torchaudio.load('./test/voxceleb1/xTOk1Jz-F_g_0000015.wav')[0][0].numpy())
    cos_tar = np.dot(emb_1, emb_2.T) / (np.linalg.norm(emb_1) * np.linalg.norm(emb_2))
    cos_tar = cos_tar[0][0]
    emb_3 = wespeaker_model(torchaudio.load('./test/voxceleb1/HXUqYaOwrxA_0000015.wav')[0][0].numpy())
    cos_imp = np.dot(emb_1, emb_3.T) / (np.linalg.norm(emb_1) * np.linalg.norm(emb_3))
    cos_imp = cos_imp[0][0]
    assert np.allclose([cos_tar, cos_imp], [0.5586, 0.2006], atol=1.e-4)
    #################### Utmos ####################

    print('All tests passed!')