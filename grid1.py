import trainBPR
import train
import fire
import itertools
import os

import reset_seed

SEED = 242
CUDA = "0"
reset_seed.frozen(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA

import torch as tc
from models.DIRECT import DIRECT
from datas.logger import print
from metrics.evaluate import TopKEvaluator, CTREvaluator

plm = "prajjwal1/bert-small"
amazon = ("reviewerID", "asin", "reviewText", "overall")
yelp = ("user_id", "business_id", "text", "stars")
setup = "DEFAULT"

continue_ckpt = False

datafiles = [
    "./datasets/reviews_Toys_and_Games_5.json",
    # "./datasets/reviews_Video_Games_5.json",
    # "./datasets/reviews_Clothing_Shoes_and_Jewelry_5.json",
    # "./datasets/yelp2019_5core.json",
    # "./datasets/reviews_CDs_and_Vinyl_5.json",
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
    "num_epochs": 1,
    "decay_rate": 0.1,
    "decay_tol": 2,
    "early_stop": 2,
    "weight_decay": 1e-6,
    "optimizer": "AdamW",
    "max_norm": 1.0,
    "frozen_train_size": 30,
    "log_frequency": 200000
}

HYPER_GRID = {
    "learn_rate": [1e-4, 1e-3],
#    "batch_size": [16, 32],
#    "weight_decay": [0, 1e-6],
    # aggiungi qui altre varianti di TRAIN_CONFIG...
}


def generate_grid(grid_dict):
    keys, values = zip(*grid_dict.items())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def grid(datafile, setup="default"):
    # Definizione del file di log
    base_name = os.path.basename(datafile).replace('.json', '')
    log_filename = f"grid_{setup}_{base_name}.txt"
    # Header file di log
    with open(log_filename, 'w') as log_file:
        log_file.write(f"Grid Search Log - setup={setup}, datafile={datafile}\n")
        log_file.write(f"Timestamp: {datetime.now()}\n\n")

    print(f"[GRID] Inizio grid search: setup={setup}, datafile={datafile}")
    with open(log_filename, 'a') as log_file:
        log_file.write(f"[GRID] Inizio grid search: {datetime.now()}")

    format_ = amazon
    GRID_best_score = float("inf")
    GRID_best_model = "outputs/GRID_seed%d_%s_%s.pth" % (SEED, setup, os.path.split(datafile)[-1])
    meta, datas, item_doc = train.get_subsets(datafile, format_, DATA_CONFIG)
    model = DIRECT(user_num=len(meta.users),
                   item_num=len(meta.items),
                   **MODEL_CONFIG)
    model.prepare_item_embedding(item_doc)
    ctr_grader = CTREvaluator(threshold=4.0)
    topk_grader = TopKEvaluator(meta=datas[0].meta, train=datas[0], valid=datas[1], k=[10])

    for grid_cfg in generate_grid(HYPER_GRID):
        # aggiorno TRAIN_CONFIG con la singola combinazione
        cfg = TRAIN_CONFIG.copy()
        cfg.update(grid_cfg)
        print(f"[GRID] Avvio training per config: {cfg}")
        with open(log_filename, 'a') as log_file:
            log_file.write(f"[TRAIN] Start training: {datetime.now()}, config={cfg}")

        # fit a seconda del setup
        if setup == "default":
            score, _ = train.fit(datas, model, **cfg)
            with open(log_filename, 'a') as log_file:
                log_file.write(f"Config: {grid_cfg} -> MSE: {score:.6f}\n")
                print(f"Config: {grid_cfg} -> MSE: {score:.6f}\n")
        elif setup == "bpr":
            score, _, _ = trainBPR.fit(datas, model, **cfg)
            with open(log_filename, 'a') as log_file:
                log_file.write(f"Config: {grid_cfg} -> Recall: {score:.6f}\n")
                print(f"Config: {grid_cfg} -> BPR: {score:.6f}\n")

        # confronto e salvataggio
        if score < GRID_best_score:
            GRID_best_score = score
            tc.save(model.state_dict(), GRID_best_model)
            best_cfg = cfg

    model.load_state_dict(tc.load(GRID_best_model))

    print(f"BEST CONFIG: {best_cfg}")

    # Metriche TopK sullo stesso modello/test set
    topk_metrics = topk_grader.evaluate(model)
    precision_at_10 = topk_metrics["precision"][0]
    recall_at_10 = topk_metrics["recall"][0]
    print(f"Test Metrics → Precision@10={precision_at_10:.4f} • Recall@10={recall_at_10:.4f}")

    # Metriche CTR sul test set
    auc, acc, f1, mse, mae = ctr_grader.evaluate(model, datas[2])
    print(f"Test Metrics → Accuracy={acc:.4f} • AUC={auc:.4f} • "
          f"F1={f1:.4f} • MSE={mse:.4f} • MAE={mae:.4f}")

    # Log finali delle metriche di test
    with open(log_filename, 'a') as log_file:
        log_file.write("\n=== Final Test Metrics (modello migliore) ===\n")
        log_file.write(f"CTR → Accuracy={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}, MSE={mse:.4f}, MAE={mae:.4f}\n")
        log_file.write(f"TopK → Precision@10={precision_at_10:.4f}, Recall@10={recall_at_10:.4f}\n")

    print(f"Grid search completata. Log salvato in {log_filename}")

    return GRID_best_score, model


if __name__ == "__main__":
    fire.Fire()

