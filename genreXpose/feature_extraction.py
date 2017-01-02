import os
import numpy as np
import glob
from utils import GENRE_DIR, GENRE_LIST, TEST_DIR, FEATURE_EXTRACTION_SCRIPT

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
    cmd = "python {} featureExtractionFile -i {} -mw 1.0 -ms 1.0 -sw 0.050 -ss 0.050 -o {}".format(FEATURE_EXTRACTION_SCRIPT, fn, fn)
    print cmd
    os.system(cmd)
    return fn + ".npy"

if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    for subdir, dirs, files in os.walk(GENRE_DIR):
        traverse = list(set(dirs).intersection( set(GENRE_LIST) ))
        break
    print "Working with these genres --> ", traverse

    for genre in traverse:
        # TODO : Use a Python call rather than a system call
        cmd = "python {} featureExtractionDir -i {}/{}/ -mw 1.0 -ms 1.0 -sw 0.050 -ss 0.050".format(FEATURE_EXTRACTION_SCRIPT, GENRE_DIR, genre)
        os.system(cmd)

    stop = timeit.default_timer()
    print "Total feature extraction and feature writing time (s) = ", (stop - start)
