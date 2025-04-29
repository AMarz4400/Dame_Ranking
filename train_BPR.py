import math
import sys
import os
import copy
import pickle
from datetime import datetime

import reset_seed

SEED = 242
CUDA = "0"
reset_seed.frozen(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA

import numpy as np
import pandas as pd
import torch as tc
import tqdm

from metrics.evaluate import TopKEvaluator, CTREvaluator
from models.DIRECT import DIRECT
from datas.dataset import Dataset, MetaIndex, DocumentDataset
from datas.logger import print
from datas.preprocess import initialize_dataset
from models.Losses import *

plm = "prajjwal1/bert-small"
amazon = ("reviewerID", "asin", "reviewText", "overall")
yelp = ("user_id", "business_id", "text", "stars")
setup = "BPR"

continue_ckpt = False

datafiles = [
    "./datasets/reviews_Toys_and_Games_5.json",
    #"./datasets/reviews_Video_Games_5.json",
    #"./datasets/reviews_Clothing_Shoes_and_Jewelry_5.json",
    #"./datasets/yelp2019_5core.json",
    #"./datasets/reviews_CDs_and_Vinyl_5.json",
]

DATA_CONFIG = {
    "init": {"valid": 0.1,
             "test": 0.2,
             "seed": SEED,
             "min_freq": 1,
             "pretrain": plm,
             "num_worker": 8,
             "force_init": False,
             },
    "meta": {"tokenizer": plm,
             "num_sent": None,
             "len_sent": None,
             "num_hist": 30,
             "len_doc": 510,
             "cache_freq": 1,
             "keep_head": 1.0,
             "drop_stopword": False
             },
    "data": {"sampling": None,
             "cache_freq": 1}}

MODEL_CONFIG = {
    "plm": plm,
    "dropout": 0.3,
    "aspc_num": 5,
    "aspc_dim": 64,
    "gamma1": 5e-3,  # \Loss_c: Contrastive Training 
    "gamma2": 1e-6,  # \Omega_d: Diversity Assumption
    "gamma3": 2.5,  # \Omega_o: Orthogonal Assumption
    "beta": 0.1,
    "sampling": 0.1,
    "threshold": 0.2,
    "device": "cuda:%s" % CUDA,
}

TRAIN_CONFIG = {
    "use_amp": False,
    "learn_rate": 1e-3,
    "batch_size": 32,
    "workers": 2,
    "num_epochs": 2,
    "decay_rate": 0.1,
    "decay_tol": 2,
    "early_stop": 2,
    "weight_decay": 1e-6,
    "optimizer": "AdamW",
    "max_norm": 1.0,
    "frozen_train_size": 30,
    "log_frequency": 200000
}


def get_subsets(root, format_, configs, splits=("train", "valid", "test")):
    assert isinstance(configs, dict) and len(configs) == 3
    assert "init" in configs and "data" in configs and "meta" in configs
    configs = copy.deepcopy(configs)
    root_info = initialize_dataset(datafile, format_, dotokenize=True, **configs["init"])
    configs["init"]["valid"] = configs["init"]["test"] = 0.0
    meta = MetaIndex(root_info["root"], **configs["meta"])
    train_info = initialize_dataset(root_info["root"] + "/train.json", format_, users=meta.users, items=meta.items,
                                    **configs["init"])
    train_meta = MetaIndex(train_info["root"], users=meta.users, items=meta.items, **configs["meta"])
    subsets = [Dataset(train_info["root"], "train", format_, train_meta, **configs["data"], setup=setup)]
    paths = {"train": train_info["root"]}
    for split in splits[1:]:
        splitfile = root_info["root"] + "/" + split + ".json"
        info = initialize_dataset(splitfile, format_, users=train_meta.users, items=train_meta.items, **configs["init"])
        tmp_meta = MetaIndex(train_info["root"], users=meta.users, items=meta.items, **configs["meta"])

        subsets.append(Dataset(info["root"], "train", format_, tmp_meta, **configs["data"],
                               paths=paths, split=split, setup=setup))
        if split == "valid":
            paths['valid'] = info["root"]

    documents = DocumentDataset(train_info["root"] + "/item_doc.txt",
                                configs["meta"]["tokenizer"], configs["meta"]["len_doc"],
                                configs["meta"]["keep_head"], 1)
    return train_meta, subsets, documents


def fit(datas, model, optimizer, learn_rate, batch_size, num_epochs, max_norm, log_frequency, frozen_train_size,
        decay_rate, decay_tol, early_stop, weight_decay, use_amp, workers):
    optimizer = getattr(tc.optim, optimizer)(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    scheduler = tc.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay_rate)
    progress = tqdm.tqdm(total=math.ceil(len(datas[0]) / batch_size) * num_epochs)

    train = tc.utils.data.DataLoader(datas[0], batch_size=batch_size, shuffle=True, num_workers=workers)
    valid = tc.utils.data.DataLoader(datas[1], batch_size=batch_size, shuffle=False, num_workers=2)
    test = tc.utils.data.DataLoader(datas[2], batch_size=batch_size, shuffle=False, num_workers=2)

    bpr_loss_func = BPRLoss()
    total_loss = 0.0
    history = {'loss': []}

    for epoch in range(num_epochs):
        model.train()
        print(f"{datetime.now()}  Epoch {epoch}...")

        for positives, negatives in train:
            pos_scores = model(**positives)  # Dati per l'item positivo
            neg_scores = model(**negatives)  # Dati per l'item negativo
            del positives, negatives

            # Calcolo della BPR loss
            loss = bpr_loss_func(pos_scores, neg_scores)
            history['loss'].append(loss.item())
            total_loss += loss.item() * len(pos_scores)
            del pos_scores, neg_scores

            # Aggiorna i pesi del modello
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress.update(1)

            # Scheduler per la learning rate

        scheduler.step()

    progress.close()
    # Report sulla loss media
    avg_loss = total_loss / len(train)
    print(f"\tTraining data: total loss: {total_loss:.4f}, avg loss: {avg_loss:.4f}")

    lossFile = 'loss_'+str(SEED)+'.csv'
    print(f"\tSaving losses into {lossFile}")

    history['loss'] = np.array(history['loss'])
    df = pd.DataFrame(history['loss'], columns=['loss'])
    df.to_csv(lossFile, index=False)


if __name__ == "__main__":
    for datafile in datafiles:
        format_ = yelp if "yelp" in datafile else amazon
        meta, datas, item_doc = get_subsets(datafile, format_, DATA_CONFIG)
        model = DIRECT(user_num=len(meta.users),
                       item_num=len(meta.items),
                       **MODEL_CONFIG)
        model.prepare_item_embedding(item_doc)
        fit(datas, model, **TRAIN_CONFIG)
