import numpy as np

HW = [bin(n).count("1") for n in range(0,256)]


def intermediate(pt, keyguess):
    return pt ^ keyguess

traces = np.load("D:\\Side_Channel_Attack\\Traces\\Xor_50000_fixed_key\\traces.npy")
pt = np.load("D:\\Side_Channel_Attack\\Traces\\Xor_50000_fixed_key\\p.npy")

numtraces = np.shape(traces)[0]-1
numpoint = np.shape(traces)[1]

#Use less than the maximum traces by setting numtraces to something
#numtraces = 15

bestguess = [0]*16
#Set 16 to something lower (like 1) to only go through a single subkey & save time!
for bnum in range(0, 16):
    cpaoutput = [0]*256
    maxcpa = [0]*256
    for kguess in range(0, 256):
        # print("Subkey %2d, hyp = %02x: "%(bnum, kguess),)


        #Initialize arrays & variables to zero
        sumnum = np.zeros(numpoint)
        sumden1 = np.zeros(numpoint)
        sumden2 = np.zeros(numpoint)

        hyp = np.zeros(numtraces)
        for tnum in range(0, numtraces):
            hyp[tnum] = HW[intermediate(pt[tnum][bnum], kguess)]


        #Mean of hypothesis
        meanh = np.mean(hyp, dtype=np.float64)

        #Mean of all points in trace
        meant = np.mean(traces, axis=0, dtype=np.float64)

        #For each trace, do the following
        for tnum in range(0, numtraces):
            hdiff = (hyp[tnum] - meanh)
            tdiff = traces[tnum,:] - meant

            sumnum = sumnum + (hdiff*tdiff)
            sumden1 = sumden1 + hdiff*hdiff 
            sumden2 = sumden2 + tdiff*tdiff

        cpaoutput[kguess] = sumnum / np.sqrt( sumden1 * sumden2 )
        maxcpa[kguess] = max(abs(cpaoutput[kguess]))
        #print(kguess)
        #print(maxcpa[kguess])

    #Find maximum value of key
    bestguess[bnum] = np.argmax(maxcpa)
    print(bestguess[bnum])

print("Best Key Guess: ")
# for b in bestguess: print("%02x "%b,)