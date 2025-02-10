from mmj.utils import model_utils


def download_model():
    # SDK模型下载
    from modelscope import snapshot_download
    snapshot_download('iic/CosyVoice2-0.5B', local_dir=model_utils.get_pretrained_model_dir('CosyVoice2-0.5B'))
    snapshot_download('iic/CosyVoice-300M', local_dir=model_utils.get_pretrained_model_dir('CosyVoice-300M'))
    snapshot_download('iic/CosyVoice-300M-25Hz', local_dir=model_utils.get_pretrained_model_dir('CosyVoice-300M-25Hz'))
    snapshot_download('iic/CosyVoice-300M-SFT', local_dir=model_utils.get_pretrained_model_dir('CosyVoice-300M-SFT'))
    snapshot_download('iic/CosyVoice-300M-Instruct', local_dir=model_utils.get_pretrained_model_dir('CosyVoice-300M-Instruct'))
    snapshot_download('iic/CosyVoice-ttsfrd', local_dir=model_utils.get_pretrained_model_dir('CosyVoice-ttsfrd'))


if __name__ == '__main__':
    download_model()
    # print(os_utils.get_path_under_work_dir('pretrained_models'))