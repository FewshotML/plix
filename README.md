# [Plug-and-Play Multilingual Few-shot Spoken Words Recognition](https://arxiv.org/pdf/2305.03058.pdf)

[![Downloads](https://static.pepy.tech/personalized-badge/plixkws?period=total&units=international_system&left_color=black&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/plixkws)
[<img src="https://img.shields.io/badge/license-Apache_2.0-blue">](https://github.com/FewshotML/plix)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yrcZ_M5hDcjpTMmjivJqvQX9n8g7rw-6)


## Abstract
As technology advances and digital devices become prevalent, seamless human-machine communication is increasingly gaining significance. The growing adoption of mobile, wearable, and other Internet of Things (IoT) devices has changed how we interact with these smart devices, making accurate spoken words recognition a crucial component for effective interaction. However, building robust spoken words detection system that can handle novel keywords remains challenging, especially for low-resource languages with limited training data. Here, we propose PLiX, a multilingual and plug-and-play keyword spotting system that leverages few-shot learning to harness massive real-world data and enable the recognition of unseen spoken words at test-time. Our few-shot deep models are learned with millions of one-second audio clips across 20 languages, achieving state-of-the-art performance while being highly efficient. Extensive evaluations show that PLiX can generalize to novel spoken words given as few as just one support example and performs well on unseen languages out of the box. We release models and inference code to serve as a foundation for future research and voice-enabled user interface development for emerging devices.

<p align="center">
  <img src="https://github.com/FewshotML/plix/blob/main/img/plix_kws.png" alt="Illustration of Plug-and-Play Multilingual Few-shot Spoken Words Recognition" width="80%"/>
</p>

## Key Contributions 
* We develop PLiX, a general-purpose, multilingual, and plug-and-play, few-shot keyword spotting system trained and evaluated with more than 12 million one-second audio clips sampled at 16kHz.
* Leverage state-of-the-art neural architectures to learn few-shot models that are high performant while being efficient with fewer learnable parameters. 
* A wide-ranging set of evaluations to systematically quantify the efficacy of our system across 20 languages and thousands of classes (i.e., words or terms); showcasing generalization to unseen words at test-time given as few as one support example per class. 
* We demonstrate that our model generalizes exceptionally well in a one-shot setting on 5 unseen languages. Further, in a cross-task transfer evaluation on a challenging FLEURS benchmark, our model performs well for language identification without any retraining. 
* To serve as a building block for future research on spoken word detection with meta-learning and enable product development, we release model weights and inference code as a Python package.

## Quick Start
We provide the library for our PLiX model:
```bash
pip install plixkws
```

Then you can follow the below usage or refer to [test_model.py](https://github.com/FewshotML/plix/blob/main/test_model.py).

```python
import torch
from plixkws import model, util

support_examples = ["./test_clips/aandachtig.wav", "./test_clips/stroom.wav",
    "./test_clips/persbericht.wav", "./test_clips/klinkers.wav",
    "./test_clips/zinsbouw.wav"]
classes = ["aandachtig", "stroom", "persbericht", "klinkers", "zinsbouw"]
int_indices = [0,1,2,3,4]

fws_model = model.load(encoder_name="base", language="en", device="cpu")

support = {
    "paths": support_examples,
    "classes": classes,
    "labels": torch.tensor(int_indices),
}
support["audio"] = torch.stack([util.load_clip(path) for path in support["paths"]])
support = util.batch_device(support, device="cpu")

query = {
    "paths": ["./test_clips/query_klinkers.wav", "./test_clips/query_stroom.wav"]
}
query["audio"] = torch.stack([util.load_clip(path) for path in query["paths"]])
query = util.batch_device(query, device="cpu")

with torch.no_grad():
    predictions = fws_model(support, query)
```

## Real-time Inference

```python

# !pip install pyaudio
import numpy as np
import pyaudio
import torch
from plixkws import model, util

sample_rate = 16000
frames_per_buffer = 512
support_examples = ["./test_clips/aandachtig.wav", "./test_clips/stroom.wav",
    "./test_clips/persbericht.wav", "./test_clips/klinkers.wav",
    "./test_clips/zinsbouw.wav"]
classes = ["aandachtig", "stroom", "persbericht", "klinkers", "zinsbouw"]
int_indices = [0,1,2,3,4]

support = {
    "paths": support_examples,
    "classes": classes,
    "labels": torch.tensor(int_indices)
}
support["audio"] = torch.stack([util.load_clip(path) for path in support["paths"]])
support = util.batch_device(support, device="cpu")

fws_model = model.load(encoder_name="small", language="nl", device="cpu")

p = pyaudio.PyAudio()
stream = p.open(format = pyaudio.paInt16, channels=1, 
    rate=sample_rate, input=True, frames_per_buffer=frames_per_buffer)

frames = []
while True:  
    data = stream.read(frames_per_buffer)
    buffer = np.frombuffer(data, dtype=np.int16)
    frames.append(buffer)
    if len(frames) * frames_per_buffer / sample_rate >= 1:
        audio = np.concatenate(frames)
        audio = audio.astype(float) / np.iinfo(np.int16).max 
        query = {"audio":torch.tensor(audio[np.newaxis, np.newaxis,:], dtype=torch.float32)}
        query = util.batch_device(query, device="cpu")
        with torch.no_grad():
            predictions = fws_model(support, query)
            print(classes[predictions.item()])
        frames = []
```

## Pretrained Model Weights
| Language | Encoder Name and Checkpoint |
| --- | --- |
| Multilingual | [base_multi](https://www.dropbox.com/s/7kqpue5g1f45f0x/base_multi_model.pt?dl=1) |
| Multilingual | [small_multi](https://www.dropbox.com/s/l9ti33nfdkls1n4/small_multi_model.pt?dl=1) |
| English | [base_en](https://www.dropbox.com/s/d6qb4o96b0o0bcl/base_en_model.pt?dl=1) |
| Arabic | [small_ar](https://www.dropbox.com/s/svwjrd129mel601/small_ar_model.pt?dl=1) |
| Czech | [small_cs](https://www.dropbox.com/s/0a1gxyi1kt740me/small_cs_model.pt?dl=1) |
| German | [small_de](https://www.dropbox.com/s/rmcuirj9tz527s3/small_de_model.pt?dl=1) |
| Greek | [small_el](https://www.dropbox.com/s/7pghc17qqn7b433/small_el_model.pt?dl=1) |
| English | [small_en](https://www.dropbox.com/s/yj3hcolw054h26a/small_en_model.pt?dl=1) |
| Estonian | [small_et](https://www.dropbox.com/s/p7bi91fcupj0ufo/small_et_model.pt?dl=1) |
| Spanish | [small_es](https://www.dropbox.com/s/xf5awxt420072y4/small_es_model.pt?dl=1) |
| Persian | [small_fa](https://www.dropbox.com/s/sqptds0gobof7l8/small_fa_model.pt?dl=1) |
| French | [small_fr](https://www.dropbox.com/s/c5kg87y363c6fwz/small_fr_model.pt?dl=1) |
| Indonesian | [small_id](https://www.dropbox.com/s/ll16kp6ir3ncdbs/small_id_model.pt?dl=1) |
| Italian | [small_it](https://www.dropbox.com/s/tj3zjor6mwpid1b/small_it_model.pt?dl=1) |
| Kyrgyz | [small_ky](https://www.dropbox.com/s/860igenj8q40k6a/small_ky_model.pt?dl=1) |
| Dutch | [small_nl](https://www.dropbox.com/s/fbavn10ut72fjjx/small_nl_model.pt?dl=1) |
| Polish | [small_pl](https://www.dropbox.com/s/hn963bdgp4sqv5b/small_pl_model.pt?dl=1) |
| Portuguese | [small_pt](https://www.dropbox.com/s/koolgrs35nbzmt4/small_pt_model.pt?dl=1) |
| Russian | [small_ru](https://www.dropbox.com/s/kqtsur47syzqvk0/small_rw_model.pt?dl=1) |
| Kinyarwanda | [small_rw](https://www.dropbox.com/s/kqtsur47syzqvk0/small_rw_model.pt?dl=1) |
| Swedish | [small_sv-SE](https://www.dropbox.com/s/2qv6u5ns8rb9vhi/small_sv-SE_model.pt?dl=1) |
| Turkish | [small_tr](https://www.dropbox.com/s/jir945jqkdq61w8/small_tr_model.pt?dl=1) |
| Tatar | [small_tt](https://www.dropbox.com/s/bzymm5s7rujximz/small_tt_model.pt?dl=1) |

## Citation
If you find this work useful, please cite our paper:
```
@article{saeed2023plix,
  title={Plug-and-Play Multilingual Few-shot Spoken Words Recognition},
  author={Saeed, Aaqib and Tsouvalas, Vasileios},
  journal={arXiv preprint arXiv:2305.03058},
  year={2023}
}
```
