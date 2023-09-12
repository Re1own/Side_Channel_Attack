import argparse
import copy
import collections
import functools as ft

import h5py
import numpy as np
from scalib.metrics import SNR
import scalib.modeling
import scalib.attacks
import scalib.postprocessing
from tqdm import tqdm

from six.moves import cPickle as pickle #for performance




class Settings:
    """Command-line settings (hashable object)."""
    pass
def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return

def target_variables(byte):
    """variables that will be profiled"""
    return [f"{base}_{byte}" for base in ("k", "output")]

def parse_args():
    parser = argparse.ArgumentParser(
            description="Attack against plaintext xor key"
            )
    parser.add_argument(
        "--attacks",
        type=int,
        default=100,
        help="Number of attack runs (default: %(default)s).",
    )
    parser.add_argument(
        "--profile",
        type=int,
        default=5000,
        help="Number of traces used for profiling (default: %(default)s).",
    )
    parser.add_argument(
        "--poi",
        type=int,
        default=120,
        help="Number of POIs for each variable (default: %(default)s).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=8,
        help="Dimensionality of projected space for LDA (default: %(default)s).",
    )
    # parser.add_argument(
    #     "--database",
    #     type=str,
    #     default="./atmega8515-raw-traces.h5",
    #     help="Location of the 'raw traces' ASCAD file (default: %(default)s).",
    # )
    return parser.parse_args(namespace=Settings())

def compute_snr(settings):
    """Returns the SNR of the traces samples for each target variable."""
    snrs = {v: dict() for i in range(16) for v in target_variables(i)}
    traces, labels = get_traces(settings, start=0, l=settings.profile)
    for v, m in tqdm(snrs.items(), total=len(snrs), desc="SNR Variables"):
        snr = SNR(np=1, nc=256, ns=traces.shape[1])
        x = labels[v].reshape((settings.profile, 1))
        # Note: if the traces do not fit in RAM, you can call multiple times fit_u
        # on the same SNR object to do incremental SNR computation.
        snr.fit_u(traces, x)
        m["SNR"] = snr.get_snr()[0, :]
        # Avoid NaN in case of scope over-range
        np.nan_to_num(m["SNR"], nan=0.0)
    return snrs


def map_to_0_255(data_array):
    max_value = np.max(data_array)
    min_value = np.min(data_array)

    mapped_array = np.int16((data_array - min_value) / (max_value - min_value) * 510 - 255)
    return mapped_array
def get_traces(settings, start, l):
    """Load traces and labels"""
    traces = np.load("E:\\Side_Channel_Attack\\Traces\\Xor_20000_variable_key\\traces.npy")
    traces = map_to_0_255(traces)
    traces = traces[start:l]
    plaintext = np.load("E:\\Side_Channel_Attack\\Traces\\Xor_20000_variable_key\\p.npy")
    plaintext = plaintext.astype(np.uint16)
    plaintext = plaintext[start:l]
    key =np.load("E:\\Side_Channel_Attack\\Traces\\Xor_20000_variable_key\\k.npy")
    key = key.astype(np.uint16)
    key = key[start:l]
    labels = var_labels(key, plaintext)
    return traces, labels

def get_traces2(settings, start, l):
    """Load traces and labels"""
    traces = np.load("E:\\Side_Channel_Attack\\Traces\\Xor_50000_fixed_key\\traces.npy")
    traces = map_to_0_255(traces)
    traces = traces[start:l]
    plaintext = np.load("E:\\Side_Channel_Attack\\Traces\\Xor_50000_fixed_key\\p.npy")
    plaintext = plaintext.astype(np.uint16)
    plaintext = plaintext[start:l]
    key =np.load("E:\\Side_Channel_Attack\\Traces\\Xor_50000_fixed_key\\k.npy")
    key = key.astype(np.uint16)
    key = key[start:l]
    labels = var_labels(key, plaintext)
    return traces, labels
def var_labels(key, plaintext):
    "Compute value of variables of interest based on ASCAD metadata."
    output = key ^ plaintext

    labels = {}
    for i in range(16):
        labels[f"k_{i}"] = key[:, i]
        labels[f"p_{i}"] = plaintext[:, i]
        labels[f"output_{i}"] = output[:, i]

    return labels

def compute_templates(settings, snrs):
    """Compute the POIs, LDA and gaussian template for all variables."""
    models = dict()
    # Select POIs
    for k, m in snrs.items():
        poi = np.argsort(m["SNR"])[-settings.poi:].astype(np.uint32)
        poi.sort()
        models[k] = {"poi": poi}
    traces, labels = get_traces(settings, start=0, l=settings.profile)
    vs = list(models.keys())
    mlda = scalib.modeling.MultiLDA(
        ncs=len(models) * [256],
        ps=len(models) * [settings.dim],
        pois=[models[v]["poi"] for v in vs],
    )
    x = np.array([labels[v] for v in vs]).transpose()
    mlda.fit_u(traces, x)
    mlda.solve()
    for lda, v in zip(mlda.ldas, vs):
        models[v]["lda"] = lda
    return models

@ft.lru_cache(maxsize=None)
def sasca_graph():
    sasca = scalib.attacks.SASCAGraph(SASCA_GRAPH, n=1)
    sasca.set_table("sbox", SBOX.astype(np.uint32))
    return sasca

SASCA_GRAPH = """
NC 256
TABLE sbox

VAR SINGLE k
VAR MULTI p
VAR MULTI output
PROPERTY output = p ^ k
"""
SBOX = np.array(
    [
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB,
        0x76, 0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4,
        0x72, 0xC0, 0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71,
        0xD8, 0x31, 0x15, 0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2,
        0xEB, 0x27, 0xB2, 0x75, 0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6,
        0xB3, 0x29, 0xE3, 0x2F, 0x84, 0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB,
        0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF, 0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45,
        0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8, 0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5,
        0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2, 0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44,
        0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73, 0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A,
        0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB, 0xE0, 0x32, 0x3A, 0x0A, 0x49,
        0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79, 0xE7, 0xC8, 0x37, 0x6D,
        0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08, 0xBA, 0x78, 0x25,
        0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A, 0x70, 0x3E,
        0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E, 0xE1,
        0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB,
        0x16,
    ],
    dtype=np.uint16,
)



def attack(traces, labels, models):
    """Run a SASCA attack on the given traces and evaluate its performance.
    Returns the true key and the byte-wise key distribution estimated by the attack.
    """
    # correct secret key
    secret_key = [labels[f"k_{i}"][0] for i in range(16)]
    # distribution for each of the key bytes
    key_distribution = []
    # Run a SASCA for each S-Box
    for i in range(16):
        sasca = copy.deepcopy(sasca_graph())
        # Set the labels for the plaintext byte
        sasca.set_public(f"p", labels[f"p_{i}"].astype(np.uint32))
        for var in target_variables(i):
            model = models[var]
            prs = model["lda"].predict_proba(traces[:, model["poi"]])
            sasca.set_init_distribution(var.split('_')[0], prs)
        sasca.run_bp(it=3)
        distribution = sasca.get_distribution(f"k")[0, :]
        key_distribution.append(distribution)
    key_distribution = np.array(key_distribution)
    return secret_key, key_distribution

def run_attack_eval(traces, labels, models):
    """Run a SASCA attack on the given traces and evaluate its performance.
    Returns the log2 of the rank of the true key.
    """




    secret_key, key_distribution = attack(traces, labels, models)
    rmin, r, rmax = scalib.postprocessing.rank_accuracy(
        -np.log2(key_distribution), secret_key, max_nb_bin=2**20
    )
    lrmin, lr, lrmax = (np.log2(rmin), np.log2(r), np.log2(rmax))
    return lr

def run_attacks_eval(settings, models):
    """Return the list of the rank of the true key for each attack."""
    # Offset in traces to no attack the training traces
    #traces, labels = get_traces(settings, start=settings.profile, l=settings.attacks)
    traces, labels = get_traces(settings, 0, settings.attacks)
    # print(trace1)
    # print(key1)
    return 2**np.array(list(tqdm(map(
        lambda a: run_attack_eval(
            traces[a:a+1,:],
            {k: val[a:a+1] for k, val in labels.items()},
            models
            ),
        range(settings.attacks),
        ),
        total=settings.attacks,
        desc="attacks",
        )))

def success_rate(ranks, min_rank=1):
    return np.sum(ranks <= min_rank) / ranks.size

if __name__ == "__main__":
    settings = parse_args()

    print("Start SNR estimation")
    snr = compute_snr(settings)
    print("done")
    print("Start modeling")
    models = compute_templates(settings, snr)
    print("done")

    print("Start attack")

    # trace1, label1 = get_traces2(settings, 0, 1)
    # # print(trace1)
    # # print(key1)
    # secret_key = [label1[f"k_{i}"][0] for i in range(16)]
    # key_distribution = []
    # secret_key, key_distribution = attack(trace1, label1, models)

    ranks = run_attacks_eval(settings, models)
    print('Attack ranks', collections.Counter(ranks))
    print(f'Success rate (rank 1): {success_rate(ranks, min_rank=1) * 100:.0f}%')
    print(f'Success rate (rank 2**32): {success_rate(ranks, min_rank=2 ** 32) * 100:.0f}%')
    print("done")
