import os
from typing import Optional

import librosa
import torch
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils import file_utils
from cosyvoice.utils.common import set_all_random_seed
from mmj.utils import os_utils


def get_pretrained_model_dir(model_name: str) -> str:
    """
    获取预训练模型的文件夹路径
    Args:
        model_name:

    Returns:

    """
    return os.path.join(
        os_utils.get_path_under_work_dir('pretrained_models'),
        model_name
    )


def postprocess(speech,
                cosyvoice_sr: float,
                max_val=0.8,
                top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice_sr * 0.2))], dim=1)
    return speech


def add_voice_to_spk2info(
        model_name: str,
        prompt_wav_path: str,
        prompt_text: str,
        spk_id: str,
        overwrite: bool = True,
        save_filepath: Optional[str] = None
) -> None:
    """
    添加一个音色到模型中

    Args:
        model_name:
        prompt_wav_path:
        prompt_text:
        spk_id:
        overwrite: 是否覆盖原文件
        save_filepath: 保存路径

    Returns:

    """
    set_all_random_seed(1234)
    model_dir = get_pretrained_model_dir(model_name)
    model = CosyVoice2(
        model_dir=model_dir,
    )
    # 参考 webui.py prompt_speech_16k
    prompt_speech_16k = postprocess(file_utils.load_wav(prompt_wav_path, 16000),
                                    model.sample_rate)

    resample_rate = model.sample_rate
    prompt_text = model.frontend.text_normalize(prompt_text, split=False, text_frontend=True)
    prompt_text_token, prompt_text_token_len = model.frontend._extract_text_token(prompt_text)
    prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k)
    speech_feat, speech_feat_len = model.frontend._extract_speech_feat(prompt_speech_resample)
    speech_token, speech_token_len = model.frontend._extract_speech_token(prompt_speech_16k)
    if resample_rate == 24000:
        # cosyvoice2, force speech_feat % speech_token = 2
        token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
        speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len
        speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len

    embedding = model.frontend._extract_spk_embedding(prompt_speech_16k)
    model.frontend.spk2info[spk_id] = {
        'embedding': embedding.to('cpu'),
        'prompt_text_token': prompt_text_token,
        'prompt_text_token_len': prompt_text_token_len,
        'speech_feat': speech_feat,
        'speech_feat_len': speech_feat_len,
        'speech_token': speech_token,
        'speech_token_len': speech_token_len,
    }

    if overwrite:
        save_filepath = os.path.join(model_dir, 'spk2info.pt')

    if save_filepath is None or len(save_filepath) == 0:
        print('保存路径为空 跳过保存')
        return

    torch.save(model.frontend.spk2info, save_filepath)
