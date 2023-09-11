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
    return [f"{base}_{byte}" for base in ("k", "p", "output")]

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
        default=15000,
        help="Number of traces used for profiling (default: %(default)s).",
    )
    parser.add_argument(
        "--poi",
        type=int,
        default=100,
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
    traces = np.load("D:\\Side_Channel_Attack\\Traces\\Xor_20000_variable_key\\traces.npy")
    traces = map_to_0_255(traces)
    traces = traces[start:l]
    plaintext = np.load("D:\\Side_Channel_Attack\\Traces\\Xor_20000_variable_key\\p.npy")
    plaintext = plaintext.astype(np.uint16)
    plaintext = plaintext[start:l]
    key =np.load("D:\\Side_Channel_Attack\\Traces\\Xor_20000_variable_key\\k.npy")
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

if __name__ == "__main__":
    settings = parse_args()

    print("Start SNR estimation")
    snr = compute_snr(settings)
    print("Start modeling")
    models = compute_templates(settings, snr)
    print("done")