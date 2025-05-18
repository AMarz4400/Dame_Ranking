import os
from datetime import datetime
import torch as tc
import tqdm
import numpy as np
import pandas as pd

import reset_seed
from models.DIRECT import DIRECT
from datas.logger import print
from metrics.evaluate import TopKEvaluator, CTREvaluator
import train
import trainBPR
import fire

# Random seed e GPU setup
SEED = 242
CUDA = "0"
reset_seed.frozen(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA

# Configurazioni PLM e dataset
plm = "prajjwal1/bert-small"
amazon = ("reviewerID", "asin", "reviewText", "overall")
yelp = ("user_id", "business_id", "text", "stars")

# Configurazioni dati
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
             "cache_freq": 1}
}

# Configurazione modello
MODEL_CONFIG = {
    "plm": plm,
    "dropout": 0.3,
    "aspc_num": 5,
    "aspc_dim": 64,
    "gamma1": 5e-3,
    "gamma2": 1e-6,
    "gamma3": 2.5,
    "beta": 0.1,
    "sampling": 0.1,
    "threshold": 0.2,
    "device": f"cuda:{CUDA}",
}

# Configurazione training
TRAIN_CONFIG = {
    "use_amp": False,
    "learn_rate": 1e-3,
    "batch_size": 32,
    "workers": 2,
    "num_epochs": 50,
    "decay_rate": 0.1,
    "decay_tol": 2,
    "early_stop": 2,
    "weight_decay": 1e-6,
    "optimizer": "AdamW",
    "max_norm": 1.0,
    "frozen_train_size": 30,
    "log_frequency": 200000
}

# Definizione della grid di iperparametri
from itertools import product

HYPER_GRID = {
    "learn_rate": [1e-4, 5e-4, 1e-3],
    "batch_size": [16, 32],
    "weight_decay": [0, 1e-6],
    # aggiungi altri iperparametri se serve
}

# Genera tutte le configurazioni a partire dalla grid
configs = [dict(zip(HYPER_GRID.keys(), values))
           for values in product(*HYPER_GRID.values())]

def grid(datafile, setup="default"):
    """
    Esegue la grid search per il dataset specificato e setup ('default' o 'bpr').
    Scrive su file di log ogni configurazione esplorata con relativo score,
    e alla fine salva le metriche di test del modello migliore.
    """
    # Definizione del file di log
    base_name = os.path.basename(datafile).replace('.json', '')
    log_filename = f"grid_{setup}_{base_name}.txt"
    # Header file di log
    with open(log_filename, 'w') as log_file:
        log_file.write(f"Grid Search Log - setup={setup}, datafile={datafile}")

        log_file.write(f"Timestamp: {datetime.now()}")

    # Stampo in console e log inizio grid
    print(f"[GRID] Inizio grid search: setup={setup}, datafile={datafile}")
    with open(log_filename, 'a') as log_file:
        log_file.write(f"[GRID] Inizio grid search: {datetime.now()}")

    # Inizializzazione best
    GRID_best_score = float('inf')
    GRID_best_model = f"outputs/GRID_seed{SEED}_{setup}_{base_name}.pth"

    # Preparo dati e modello
    format_ = amazon if "yelp" not in datafile else yelp
    meta, datas, item_doc = train.get_subsets(datafile, format_, DATA_CONFIG)
    model = DIRECT(user_num=len(meta.users), item_num=len(meta.items), **MODEL_CONFIG)
    model.prepare_item_embedding(item_doc)

    ctr_grader = CTREvaluator(threshold=4.0)
    topk_grader = TopKEvaluator(meta=datas[0].meta, train=datas[0], valid=datas[1], k=[10])

        # Loop su configurazioni
    for config in configs:
        # Print e log avvio training per configurazione
        print(f"[GRID] Avvio training per config: {config}")
        with open(log_filename, 'a') as log_file:
            log_file.write(f"[TRAIN] Start training: {datetime.now()}, config={config}")

        cfg = TRAIN_CONFIG.copy()
    for config in configs:
        cfg = TRAIN_CONFIG.copy()
        cfg.update(config)
        # Esecuzione fit
        if setup == "default":
            score, _ = train.fit(datas, model, **cfg)
        else:  # setup == 'bpr'
            score, _, _ = trainBPR.fit(datas, model, **cfg)

        # Log della configurazione esplorata
        with open(log_filename, 'a') as log_file:
            log_file.write(f"Config: {config} -> Score: {score:.6f}\n")

        # Salvataggio se migliore
        if score < GRID_best_score:
            GRID_best_score = score
            tc.save(model.state_dict(), GRID_best_model)

    # Carico il miglior modello
    model.load_state_dict(tc.load(GRID_best_model))

    # Print e log inizio valutazione test
    print(f"[GRID] Inizio valutazione test del modello migliore: {GRID_best_model}")
    with open(log_filename, 'a') as log_file:
        log_file.write(f"[TEST] Inizio test metrics: {datetime.now()}")

    # Metriche su test set
    # CTR
    auc, acc, f1, mse, mae, _ = ctr_grader.evaluate(model, datas[2])
    # TopK
    topk_metrics = topk_grader.evaluate(model)
    precision_at_10 = topk_metrics['precision'][0]
    recall_at_10 = topk_metrics['recall'][0]

    # Log finali delle metriche di test
    with open(log_filename, 'a') as log_file:
        log_file.write("\n=== Final Test Metrics (modello migliore) ===\n")
        log_file.write(f"CTR → Accuracy={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}, MSE={mse:.4f}, MAE={mae:.4f}\n")
        log_file.write(f"TopK → Precision@10={precision_at_10:.4f}, Recall@10={recall_at_10:.4f}\n")

    print(f"Grid search completata. Log salvato in {log_filename}")
    return GRID_best_score, model


# Espongo le funzioni via Google Fire
if __name__ == "__main__":
    fire.Fire()
