import os
import sys

from mmj.utils import model_utils, os_utils

if __name__ == "__main__":
    sys.path.append(os_utils.get_path_under_work_dir('third_party', 'Matcha-TTS'))
    model_utils.add_voice_to_spk2info(
        model_name='CosyVoice2-0.5B',
        prompt_wav_path = os.path.join(
            os_utils.get_path_under_work_dir('mmj_asset'),
            '个人录音.wav'
        ),
        prompt_text='接下来，用户可能的实际需求是什么？他们可能想利用现有的预训练音色来快速生成特定音色的语音，而不想从头训练模型。或者他们可能想在不同工具之间共享音色模型，但遇到格式不一致的问题。此外，用户可能对如何选择或处理这些音色有疑问，比如是否需要特定格式或工具。',
        spk_id='Momojie',
        overwrite=True,
        # save_filepath=os.path.join(
        #     os_utils.get_path_under_work_dir('mmj_asset'),
        #     'momojie.pt'
        # )
    )