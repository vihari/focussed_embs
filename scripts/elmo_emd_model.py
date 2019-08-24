from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import argparse
import bisect
import os
import codecs
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from gensim.corpora.dictionary import Dictionary
from pyemd import emd
from sklearn import metrics
import sys
from emd_model import tokenize

import h5py

sws = stopwords.words('english')
DEBUG = False
concat = False

# copied from keyedvectors.py (gensim)
def wmdistance(document1, document2, cembs1, cembs2, model, vocab):
    # Remove out-of-vocabulary words.
    len_pre_oov1 = len(document1)
    len_pre_oov2 = len(document2)
    vocab_words = set(model.keys()) #set(map(lambda _: _[0], vocab))
    if vocab is not None:
        vocab_map = {_[0]: int(_[1]) for _ in vocab}

    # if a token appears more than once, the context embedding of the one that appears last is only considered
    # TODO: I took this decision because considering all repetitions could put other methods at major disadvantage.
    cembs1 = dict([(document1[di], cembs1[di]) for di in range(len(document1))])
    cembs2 = dict([(document2[di], cembs2[di]) for di in range(len(document2))])
    
    document1 = [token for token in document1 if token in vocab_words]
    document2 = [token for token in document2 if token in vocab_words]
    diff1 = len_pre_oov1 - len(document1)
    diff2 = len_pre_oov2 - len(document2)
    # if diff1 > 0 or diff2 > 0:
        # logger.info('Removed %d and %d OOV words from document 1 and 2 (respectively).', diff1, diff2)

    if len(document1) == 0 or len(document2) == 0:
        #logger.info(
        #    "At least one of the documents had no words that werein the vocabulary. "
        #    "Aborting (returning inf)."
        #)
        return float('inf')

    dictionary = Dictionary(documents=[document1, document2])
    vocab_len = len(dictionary)

    if vocab_len == 1:
        # Both documents are composed by a single unique token
        return 0.0

    # Sets for faster look-up.
    docset1 = set(document1)
    docset2 = set(document2)

    # Compute distance matrix.
    distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.float64)
    for i, t1 in dictionary.items():
        for j, t2 in dictionary.items():
            if t1 not in docset1 or t2 not in docset2:
                continue
            # Compute Euclidean distance between word vectors.
            #v1, v2 = model[t1], model[t2]
            v1, v2 = cembs1[t1], cembs2[t2]
            if concat:
                v1, v2 = np.concatenate([v1, model[t1]]), np.concatenate([v2, model[t2]])
            # v1, v2= v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)
            # distance_matrix[i, j] = sqrt(np_sum((self[t1] - self[t2])**2))
            distance_matrix[i, j] = np.sqrt(np.sum((v1 - v2)**2))

    if np.sum(distance_matrix) == 0.0:
        # `emd` gets stuck if the distance matrix contains only zeros.
        # logger.info('The distance matrix is all zeros. Aborting (returning inf).')
        return float('inf')

    def nbow(document):
        d = np.zeros(vocab_len, dtype=np.float64)
        nbow = dictionary.doc2bow(document)  # Word frequencies.
        doc_len = len(document)
        for idx, freq in nbow:
            d[idx] = freq / float(doc_len)  # Normalized word frequencies.
        return d

    def nbow2(document):
        d = np.zeros(vocab_len, dtype=np.float64)
        nbow = dictionary.doc2bow(document)  # Word frequencies.
        doc_len = len(document)
        denom = 0
        for idx, freq in nbow:
            token = dictionary.id2token[idx]
            sc = float(freq)/np.log(vocab_map[token])
            d[idx] = sc # Normalized word frequencies.
            denom += sc
        d /= denom
        return d
    
    # Compute nBOW representation of documents.
    d1 = nbow(document1)
    d2 = nbow(document2)

    # Compute WMD.
    return emd(d1, d2, distance_matrix)    


import multiprocessing as mp


def worker(line_and_emb):
    """
    Works on the tuple containing the pair of questions and their contextual embeddings and returns the emd distance. 
    """
    l, all_elmo_embs = line_and_emb
    l = l.strip()
    fields = l.split('\t')
    words1 = tokenize(fields[1])
    words2 = tokenize(fields[2])
    elmo_embs1 = all_elmo_embs[:len(words1)]
    elmo_embs2 = all_elmo_embs[len(words1):]

    dist = wmdistance(words1, words2, elmo_embs1, elmo_embs2, wv_model, vocab)
    return dist


def auc_roc(train_files, elmo_files, wv_model, vocab):
    dev_nondups, dev_dups = [], []
    test_nondups, test_dups = [], []
    for fi in range(len(train_files)):
        fname = train_files[fi]
        with codecs.open(fname, 'r', 'utf-8') as f, h5py.File(elmo_files[fi], 'r') as ef:
            pool = mp.Pool(10)
            lines_and_embs = []
            all_lines = f.readlines()
            for li, line in enumerate(all_lines):
                lines_and_embs.append((line, ef[str(li)][...]))
            dists = pool.map(worker, lines_and_embs)
            for li, l in enumerate(all_lines):
                if l.split('\t')[0]=='0':
                    if fi==2:
                        dev_nondups.append(-dists[li])
                    else: test_nondups.append(-dists[li])
                else:
                    if fi==2:
                        dev_dups.append(-dists[li])
                    else: test_dups.append(-dists[li])

    #dups = map(lambda _: BIG if np.isinf(_) else _, dups)
    #nondups = map(lambda _: BIG if np.isinf(_) else _, nondups)
    # print (len(dev_dups), len(dev_nondups))
    # print (len(test_dups), len(test_nondups))
    dev_dups, dev_nondups = filter(np.isfinite, dev_dups), filter(np.isfinite, dev_nondups)
    test_dups, test_nondups = filter(np.isfinite, test_dups), filter(np.isfinite, test_nondups)
    # print (len(dev_dups), len(dev_nondups))
    # print (len(test_dups), len(test_nondups))
    
    return (metrics.roc_auc_score([-1]*len(dev_nondups) + [1]*len(dev_dups), dev_nondups+dev_dups, average='macro'),
            metrics.roc_auc_score([-1]*len(test_nondups) + [1]*len(test_dups), test_nondups+test_dups, average='macro'))


def read_vocab(fname):
    vocab = []
    with codecs.open(fname, "r", "utf-8") as f:
        for l in f:
            l = l.strip()
            fs = l.split()
            if len(fs)>1:
                vocab.append((fs[0], fs[1]))
    return vocab


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, help='Folder containing train, test and dev files', required=True)
    parser.add_argument('--wvs', type=str, help='Word vectors model', default=None)
    parser.add_argument('--concat', action='store_true')
    # suppied vocab intersection with wvs vocab is considered.
    parser.add_argument('--vocab', type=str, help='Vocab file name', required=False)
    args, unparsed = parser.parse_known_args()
    
    concat = args.concat
    if DEBUG:
        print ("Concat: %d" % concat)
    
    dev_stats, test_stats = [], []
    vocab = None
    wv_model = KeyedVectors.load_word2vec_format(args.wvs, binary=True, limit=400000)
    if args.vocab is not None:
        vocab = read_vocab(args.vocab)
        vocab_words = map(lambda _: _[0], vocab)
        vocab_words = set(filter(lambda _: _ in wv_model, vocab_words))
    else:
        vocab_words = set(wv_model.vocab.keys())
    vecs = {}
    for wrd in vocab_words:
        if wrd in wv_model:
            vecs[wrd] = wv_model.get_vector(wrd)
    wv_model = vecs
        

    fldr = os.path.expanduser("~/repos/CQADupStack/%s_se/" % args.domain)
    for attempt in range(3):
        train, test, dev = "%s/train_%d.tsv" % (fldr, attempt), "%s/test_%d.tsv" % (fldr, attempt), "%s/dev_%d.tsv" % (fldr, attempt) 
        elmo_files = ["vectors_%s_small/elmo_emb_train_%d.hdf5" % (args.domain, attempt), "vectors_%s_small/elmo_emb_test_%d.hdf5" % (args.domain, attempt), "vectors_%s_small/elmo_emb_dev_%d.hdf5" % (args.domain, attempt)]
        ds, ts = auc_roc([train, test, dev], elmo_files, wv_model, vocab)
        dev_stats.append(ds)
        test_stats.append(ts)

    if args.concat:
        wstr = "|elmo+" + args.wvs + "|"
    else: wstr = "|elmo|"
    wstr += "|".join(map(lambda _: "%0.5f" % _, dev_stats)) + "|"
    wstr += "|".join(map(lambda _: "%0.5f" % _, test_stats)) + "|"
    print (wstr)
