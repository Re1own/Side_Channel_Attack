from tqdm import tnrange
import numpy as np

# Store your key_guess here, compare to known_key
key_guess = []
known_key = [0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c]

# Which bit to target
bitnum = 0

full_diffs_list = []

for subkey in tnrange(0, 1, desc="Attacking Subkey"):

    max_diffs = [0] * 256
    full_diffs = [0] * 256

    for guess in range(0, 256):
        full_diff_trace = calculate_diffs(guess, subkey, bitnum)
        max_diffs[guess] = np.max(full_diff_trace)
        full_diffs[guess] = full_diff_trace

    # Make copy of the list
    full_diffs_list.append(full_diffs[:])

    # Get argument sort, as each index is the actual key guess.
    sorted_args = np.argsort(max_diffs)[::-1]

    # Keep most likely
    key_guess.append(sorted_args[0])

    # Print results
    print("Subkey %2d - most likely %02X (actual %02X)" % (subkey, key_guess[subkey], known_key[subkey]))

    # Print other top guesses
    print(" Top 5 guesses: ")
    for i in range(0, 5):
        g = sorted_args[i]
        print("   %02X - Diff = %f" % (g, max_diffs[g]))

    print("\n")