import torch 
from torch import nn
from torchaudio.transforms import MelSpectrogram
import timm

class Backbone(nn.Module):
    """
    A feature extractor that produces D-dimensional embeddings from audio samples. 
    
    Args:
        encoder_name (str): Name of the encoder to initialize.
        sample_rate (int): Sampline rate of the input audio.
    """

    def __init__(self, encoder_name, sample_rate: int = 16000):
        super().__init__()
        self.encoder_name = encoder_name
        self.sample_rate = sample_rate
        self.n_mels = 64 
        self.melspec = MelSpectrogram(sample_rate=sample_rate, 
            f_min = 60.0, f_max = 7800.0, n_mels=self.n_mels, 
            win_length=400, hop_length=160, n_fft=1024)
        self.log_offset = 1e-6

        if self.encoder_name == "small":
            self.encoder = timm.create_model("tinynet_e", 
                in_chans=1, num_classes=0, global_pool="")  
        elif self.encoder_name == "base": 
            self.encoder = timm.create_model("tf_efficientnetv2_m",
                in_chans=1, num_classes=0, global_pool="")   
        else:
            raise ValueError("Unknown encoder...")

    def forward(self, x: torch.Tensor):
        x = torch.log(self.melspec(x) + self.log_offset)
        x = self.encoder(x)
        x = x.mean(dim=(2,3))
        x = x.squeeze(-1) 
        return x