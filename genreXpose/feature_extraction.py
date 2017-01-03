import os
import numpy as np
import glob
from utils import GENRE_DIR, GENRE_LIST, AUDIOANALYSIS_DIR

# Load pyAudioAnalysis
import sys
sys.path.append(AUDIOANALYSIS_DIR)
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction as aF

def read_features(genre_list, base_dir=GENRE_DIR):
    """
        Reads the full features list from disk and
        returns them in a numpy array.
    """
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        for fn in glob.glob(os.path.join(base_dir, genre, "*.wav.npy")):
            ceps = np.load(fn)
            num_ceps = len(ceps)
            X.append(np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))
            y.append(label)
    return np.array(X), np.array(y)

def read_features_test(test_file):
    """
        Reads the full features from disk and
        returns them in a numpy array.
    """
    X = []
    y = []
    ceps = np.load(test_file)
    num_ceps = len(ceps)
    X.append(np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))
    return np.array(X), np.array(y)


def create_feature_test(fn):
    """
        Creates the MFCC features from the test files,
        saves them to disk, and returns the saved file name.
    """
    aF.mtFeatureExtractionToFile(fn, 1.0, 1.0, 0.050, 0.050, fn, True, True, True)
    return fn + ".npy"

if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    for subdir, dirs, files in os.walk(GENRE_DIR):
        traverse = list(set(dirs).intersection( set(GENRE_LIST) ))
        break
    print "Working with these genres --> ", traverse

    for genre in traverse:
        dir = "{}/{}".format(GENRE_DIR, genre)
        aF.mtFeatureExtractionToFileDir(dir, 1.0, 1.0, 0.050, 0.050, True, True, True);

    stop = timeit.default_timer()
    print "Total feature extraction and feature writing time (s) = ", (stop - start)
