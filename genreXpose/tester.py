import os

from sklearn.externals import joblib

from feature_extraction import read_features_test, create_feature_test
from utils import GENRE_DIR, GENRE_LIST, TEST_DIR, convert_test_to_wav

genre_list = GENRE_LIST

clf = None

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#          Please run the classifier script first
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def test_model_on_single_file(file_path):
    global max_prob_index
    clf = joblib.load('saved_model/model_ceps.pkl')
    X, y = read_features_test(create_feature_test(file_path))
    probs = clf.predict_proba(X)
    print "\t".join(str(x) for x in traverse)
    print "\t".join(str("%.3f" % x) for x in probs[0])
    probs=probs[0]
    max_prob = max(probs)
    for i,j in enumerate(probs):
        if probs[i] == max_prob:
            max_prob_index=i
    
    print max_prob_index
    predicted_genre = traverse[max_prob_index]
    print "\n\npredicted genre = ",predicted_genre
    return predicted_genre

if __name__ == "__main__":
    convert_test_to_wav()
    global traverse
    for subdir, dirs, files in os.walk(GENRE_DIR):
        traverse = list(set(dirs).intersection(set(GENRE_LIST)))
        break

    for subdir, dirs, files in os.walk(TEST_DIR):
        for f in files:
            if str.lower(f[-3:]) == 'wav':
                print f
                predicted_genre = test_model_on_single_file(TEST_DIR + f)
        break

    # should predict genre as "ROCK"
    #predicted_genre = test_model_on_single_file(test_file)
    
