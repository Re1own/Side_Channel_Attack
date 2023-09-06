import copy

from scalib.metrics import SNR, Ttest
from scalib.modeling import LDAClassifier
from scalib.attacks import FactorGraph, BPState
from scalib.postprocessing import rank_accuracy
from scalib.metrics import SNR
import scalib.modeling
import scalib.attacks
import scalib.postprocessing

from utils import sbox, gen_traces
import numpy as np
def map_to_0_255(data_array):
    max_value = np.max(data_array)
    min_value = np.min(data_array)

    mapped_array = np.int16((data_array - min_value) / (max_value - min_value) * 510 - 255)
    return mapped_array
def get_traces(start, l):
    """Load traces and labels"""
    traces = np.load("E:\\side_channel_attack\\Traces\\Xor_50000_fixed_key\\traces.npy")
    traces = map_to_0_255(traces)
    traces = traces[start:l]
    plaintext = np.load("E:\\side_channel_attack\\Traces\\Xor_50000_fixed_key\\p.npy")
    plaintext = plaintext.astype(np.uint8)
    plaintext = plaintext[start:l]
    key =np.load("E:\\side_channel_attack\\Traces\\Xor_50000_fixed_key\\k.npy")
    key = key.astype(np.uint8)
    key = key[start:l]
    labels = var_labels(key, plaintext)
    return traces, labels
def var_labels(key, plaintext):
    "Compute value of variables of interest based on ASCAD metadata."
    output = key ^ plaintext

    labels = {}
    for i in range(16):
        labels[f"k{i}"] = key[:, i]
        labels[f"p{i}"] = plaintext[:, i]
        labels[f"x{i}"] = output[:, i]

    return labels

def get_traces1(start, l):
    """Load traces and labels"""
    traces = np.load("E:\\side_channel_attack\\Traces\\Xor_20000_variable_key\\traces.npy")
    traces = map_to_0_255(traces)
    traces = traces[start:l]
    plaintext = np.load("E:\\side_channel_attack\\Traces\\Xor_20000_variable_key\\p.npy")
    plaintext = plaintext.astype(np.uint8)
    plaintext = plaintext[start:l]
    key =np.load("E:\\side_channel_attack\\Traces\\Xor_20000_variable_key\\k.npy")
    key = key.astype(np.uint8)
    key = key[start:l]
    labels = var_labels(key, plaintext)
    return traces, labels
def sasca_graph(graph_desc):
    sasca = scalib.attacks.SASCAGraph(graph_desc, n=1)
    # sasca.set_table("sbox", SBOX.astype(np.uint32))
    return sasca

def main():
    nc = 256
    npoi = 100
    ntraces_p = 20000

    ntraces_a = 40

    traces_p, labels_p = get_traces1(0, ntraces_p)

    traces_a, labels_a = get_traces(0, ntraces_a)

    _, ns = traces_p.shape

    x = np.zeros((ntraces_p, 16), dtype=np.uint16)
    for i in range(16):
        x[:, i] = labels_p[f"x{i}"]

    snr = SNR(nc=nc, ns=ns, np=16)
    snr.fit_u(traces_p, x)
    snr_val = snr.get_snr()

    print("    2.2 Select POIs with highest SNR.")
    pois = [np.argsort(snr_val[i])[-npoi:] for i in range(16)]

    print("3. Profiling")
    # We build a LDA model (pooled Gaussian templates) for each of the 16
    # Sboxes (xi).

    print("    3.1 Build LDAClassifier for each xi")
    models = []
    for i in range(16):
        lda = LDAClassifier(nc=nc, ns=npoi, p=1)
        lda.fit_u(l=traces_p[:, pois[i]], x=labels_p[f"x{i}"].astype(np.uint16))
        lda.solve()
        models.append(lda)


    print("    3.2 Get xi distributions from attack traces")
    probas = [models[i].predict_proba(traces_a[:, pois[i]]) for i in range(16)]

    print("4. Attack")
    print("    4.1 Create the SASCA Graph")
    graph_desc = f"""
                    NC {nc}
                    TABLE sbox   # The Sbox
                    """

    for i in range(16):
        graph_desc += f"""
                    VAR SINGLE k{i} # The key
                    PUB MULTI p{i}  # The plaintext
                    VAR MULTI x{i}  # Sbox output
                    VAR MULTI y{i}  # Sbox input
                    PROPERTY y{i} = k{i} ^ p{i} # Key addition
                    PROPERTY x{i} = sbox[y{i}]  # Sbox lookup
                    """

    # Initialize FactorGraph with the graph description and the required tables
    factor_graph = FactorGraph(graph_desc, {"sbox": sbox})

    print("    4.2 Create belief propagation state.")
    # We have to give the number of attack traces and the values for the public variables.
    bp = BPState(
        factor_graph,
        ntraces_a,
        {f"p{i}": labels_a[f"p{i}"].astype(np.uint32) for i in range(16)},
    )

    for i in range(16):
        bp.set_evidence(f"x{i}", probas[i])

    print("    4.3 Run belief propagation")
    for i in range(16):
        bp.bp_acyclic(f"k{i}")

    print("5. Attack evaluation")
    print("    5.1 Byte-wise attack")

    # correct secret key
    secret_key = []
    # distribution for each of the key bytes
    key_distribution = []
    # the best key guess of the adversary
    guess_key = []
    # rank for all the key bytes
    ranks = []

    for i in range(16):
        sk = labels_a[f"k{i}"][0]  # secret key byte
        distribution = bp.get_distribution(f"k{i}")

        guess_key.append(np.argmax(distribution))
        ranks.append(256 - np.where(np.argsort(distribution) == sk)[0])

        secret_key.append(sk)
        key_distribution.append(distribution)

    print("")
    print("        secret key (hex):", " ".join(["%3x" % (x) for x in secret_key]))
    print("        best key   (hex):", " ".join(["%3x" % (x) for x in guess_key]))
    print("        key byte ranks  :", " ".join(["%3d" % (x) for x in ranks]))
    print("")



if __name__ == "__main__":
    main()
