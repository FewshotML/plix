import torch
from plixkws import model, util

fws_model = model.load(encoder_name="base", language="en", device="cpu")

support = {
    "paths": ["./test_clips/aandachtig.wav", "./test_clips/stroom.wav",
        "./test_clips/persbericht.wav", "./test_clips/klinkers.wav",
        "./test_clips/zinsbouw.wav"],
    "labels": torch.tensor([0,1,2,3,4]),
    "classes": ["aandachtig", "stroom", "persbericht", "klinkers", "zinsbouw"],
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