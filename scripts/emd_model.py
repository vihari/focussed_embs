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


sws = stopwords.words('english')
DEBUG = False

def fltr(token):
    blacklist = ['how', 'when', 'why', 'where', 'what', 'for', 'my', 'i', 'android', 'device', 'phone']
    if token not in sws and token not in blacklist:
        return True
    return False


def tokenize(text):
    return [_ for _ in filter(fltr, [w for w in WordPunctTokenizer().tokenize(text.lower()) if w.isalpha()])]


# copied from keyedvectors.py (gensim)
def wmdistance(document1, document2, model, vocab):
    # Remove out-of-vocabulary words.
    len_pre_oov1 = len(document1)
    len_pre_oov2 = len(document2)
    vocab_words = set(model.keys()) #set(map(lambda _: _[0], vocab))
    if vocab is not None:
        vocab_map = {_[0]: int(_[1]) for _ in vocab}

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
            v1, v2 = model[t1], model[t2]
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


def worker(l):
    # print ("Vocab len: %d wv model len %d" % (len(vocab), len(wv_model)))
    l = l.strip()
    fields = l.split('\t')
    words1 = tokenize(fields[1])
    words2 = tokenize(fields[2])
    
    dist = wmdistance(words1, words2, wv_model, vocab)
    return dist


def auc_roc(train, test, dev, wv_model, vocab):
    dev_nondups, dev_dups = [], []
    test_nondups, test_dups = [], []
    for fname in [train, test, dev]:
        with codecs.open(fname, 'r', 'utf-8') as f:
            pool = mp.Pool(10)
            all_lines = f.readlines()
            dists = pool.map(worker, all_lines)
            for li, l in enumerate(all_lines):
                if l.split('\t')[0]=='0':
                    if fname==dev:
                        dev_nondups.append(-dists[li])
                    else: test_nondups.append(-dists[li])
                else:
                    if fname==dev:
                        dev_dups.append(-dists[li])
                    else: test_dups.append(-dists[li])

    # print (len(dev_dups), len(dev_nondups), len(dev_dups)+len(dev_nondups))
    # print (len(test_dups), len(test_nondups), len(test_dups)+len(test_nondups))
    # An inf is returned by emd when either of the documents has zero overlap with the vocabulary.
    # since, the vocab remains the same for all the models, they are all tested on the same number of points.
    # TODO: However, I noticed sometime later that the dev and test numbers change slightly (may be 1 or 2) between runs which baffles me.
    # Confirm that, see test.log in home.
    dev_dups, dev_nondups = filter(np.isfinite, dev_dups), filter(np.isfinite, dev_nondups)
    test_dups, test_nondups = filter(np.isfinite, test_dups), filter(np.isfinite, test_nondups)
    #print (len(dev_dups), len(dev_nondups), len(dev_dups)+len(dev_nondups))
    #print (len(test_dups), len(test_nondups), len(test_dups)+len(test_nondups))
    
    return (metrics.roc_auc_score([-1]*len(dev_nondups) + [1]*len(dev_dups), dev_nondups+dev_dups, average='macro'),
            metrics.roc_auc_score([-1]*len(test_nondups) + [1]*len(test_dups), test_nondups+test_dups, average='macro'))


def pick_threshold(fname, wv_model, vocab):
    nondups, dups = [], []
    with codecs.open(fname, 'r', 'utf-8') as f:
        pool = mp.Pool(10)
        all_lines = f.readlines()
        dists = pool.map(worker, all_lines)
        for li, l in enumerate(all_lines):
            if l.split('\t')[0]=='0':
                nondups.append(dists[li])
            else:
                dups.append(dists[li])

    dups.sort()
    nondups.sort()
    _mx = max([_ for _ in nondups+dups if not np.isinf(_)])
    _mn = min(min(nondups), min(dups))
    rng = np.arange(_mn, _mx, (_mx-_mn)/10000.)
    # print ("Debug: ", rng[0], rng[-1], rng[1]-rng[0])
    # print ("Debug (2): ", min(nondups), max([_ for _ in nondups if not np.isinf(_)]))
    max_f1 = -1
    stats = None
    for cand in rng:
        di = bisect.bisect_left(dups, cand)
        ndi = bisect.bisect_left(nondups, cand)
        recall =  di/float(len(dups))
        precision = (di/float(di + ndi)) if (di+ndi)>0 else 1
        f1 = 2*recall*precision/(recall+precision)
        max_f1 = max(max_f1, f1)
        if max_f1==f1:
            thresh = cand
            stats = [recall, precision, f1]
    # print (thresh, stats)
    return thresh, stats


def evaluate(fname, thresh, wv_model, vocab, debug_fldr=None):
    tp, tn = 0., 0.
    fp, fn = 0., 0.
    if debug_fldr is not None:
        if not os.path.exists(debug_fldr):
            os.mkdir(debug_fldr)
        tp_file = codecs.open("%s/tp.txt" % debug_fldr, "w", "utf-8")
        tn_file = codecs.open("%s/tn.txt" % debug_fldr, "w", "utf-8")
        fp_file = codecs.open("%s/fp.txt" % debug_fldr, "w", "utf-8")
        fn_file = codecs.open("%s/fn.txt" % debug_fldr, "w", "utf-8")
        
    with codecs.open(fname, 'r', 'utf-8') as f:
        for l in f:
            l = l.strip()
            fields = l.split('\t')
            words1 = [_ for _ in tokenize(fields[1])]
            words2 = [_ for _ in tokenize(fields[2])]
            words1 = filter(fltr, words1)
            words2 = filter(fltr, words2)
            
            dist = wmdistance(words1, words2, wv_model, vocab)
            qp = "%s:::%s" % (" ".join(words1), " ".join(words2))
            if fields[0]=='1':
                if dist < thresh:
                    tp += 1
                    if debug_fldr is not None:
                        tp_file.write("%s:::%s:::%s:::%f:::%f\n" % (qp, '1', '1', dist, thresh))
                else:
                    fn += 1
                    if debug_fldr is not None:
                        fn_file.write("%s:::%s:::%s:::%f:::%f\n" % (qp, '1', '0', dist, thresh))
            else:
                if dist >= thresh:
                    tn += 1
                    if debug_fldr is not None:
                        tn_file.write("%s:::%s:::%s:::%f:::%f\n" % (qp, '0', '0', dist, thresh))
                else:
                    fp += 1
                    if debug_fldr is not None:
                        fp_file.write("%s:::%s:::%s:::%f:::%f\n" % (qp, '0', '1', dist, thresh))

    if debug_fldr is not None:
        tp_file.close()
        fn_file.close()
        tn_file.close()
        fp_file.close()
    r, p = tp/(tp+fn), tp/(tp+fp)
    return r, p, 2*r*p/(r+p)


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
    parser.add_argument('--fldr', type=str, help='Folder containing train, test and dev files', required=True)
    parser.add_argument('--wvs', type=str, help='Word vectors model', required=True)
    # suppied vocab intersection with wvs vocab is considered.
    parser.add_argument('--vocab', type=str, help='Vocab file name', required=False)

    dev_stats, test_stats = [], []
    args, unparsed = parser.parse_known_args()
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
        

    if DEBUG: 
        print ("Done loading of model. Picking Threshold...")

    for attempt in range(3):
        train, test, dev = "%s/train_%d.tsv" % (args.fldr, attempt), "%s/test_%d.tsv" % (args.fldr, attempt), "%s/dev_%d.tsv" % (args.fldr, attempt) 
        # [thresh, train_stats] = pick_threshold(train, wv_model, vocab)
        # if DEBUG:
        #     print("Starting to evaluate...")
        # dev_stats = evaluate(dev, thresh, wv_model, vocab)
        # if DEBUG: print (dev_stats)
        
        # TGT = args.wvs[:args.wvs.rindex('/')]
        # test_stats = evaluate(test, thresh, wv_model, vocab)
        # if DEBUG:
        #     print (test_stats)
        # all_train_stats.append(train_stats)
        # all_dev_stats.append(dev_stats)
        # all_test_stats.append(test_stats)
        ds, ts = auc_roc(train, test, dev, wv_model, vocab)
        dev_stats.append(ds)
        test_stats.append(ts)

    """
    wstr = "|" + args.wvs
    mean, sdev = np.mean(np.array(all_train_stats), axis=0), np.sqrt(np.var(np.array(all_train_stats), axis=0))
    wstr += "|" + "|".join(["%0.3f (%0.3f)" % (mean[i], sdev[i]) for i in range(len(mean))])

    mean, sdev = np.mean(np.array(all_dev_stats), axis=0), np.sqrt(np.var(np.array(all_dev_stats), axis=0))
    wstr += "|" + "|".join(["%0.3f (%0.3f)" % (mean[i], sdev[i]) for i in range(len(mean))])

    mean, sdev = np.mean(np.array(all_test_stats), axis=0), np.sqrt(np.var(np.array(all_test_stats), axis=0))
    wstr += "|" + "|".join(["%0.3f (%0.3f)" % (mean[i], sdev[i]) for i in range(len(mean))])
 
    wstr = ""
    for i in range(len(all_train_stats)):
        wstr += "|%s (%d)|" % (args.wvs, i)
        wstr += "|".join(map(lambda _: "%0.3f" % _, all_train_stats[i])) + "|"
        wstr += "|".join(map(lambda _: "%0.3f" % _, all_dev_stats[i])) + "|"
        wstr += "|".join(map(lambda _: "%0.3f" % _, all_test_stats[i])) + "|"
        wstr += "\n"
        
    sys.stdout.write (wstr)
    """

    wstr = "|" + args.wvs + "|"
    wstr += "|".join(map(lambda _: "%0.5f" % _, dev_stats)) + "|"
    wstr += "|".join(map(lambda _: "%0.5f" % _, test_stats)) + "|"
    print (wstr)
    # print (np.mean(test_stats)*100, np.std(test_stats)*100)
