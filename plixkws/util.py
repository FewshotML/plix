import numpy as np
import librosa
import torch
import torch.nn.functional as F

def pad_or_trim(array, length, *, axis = -1):
    """
    Pad or trim the audio array to length.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array

def load_clip(audio_path: str, sample_rate: int = 16000, length: int = None):
    """
    Load an audio file.
    Returns a audio waveform: a torch.Tensor of shape (1, length)
    """
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    if audio.ndim == 1:
        audio = audio[None, :]

    length = 16000 if length is None else length 
    audio = pad_or_trim(audio, length=length)

    return torch.tensor(audio)
    
def batch_device(batch: dict, device: str = "cuda"):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch