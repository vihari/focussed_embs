#!/usr/bin/python
"""
FastText like supervised classification with word vectors fixed, so just a logistic regression
"""
import tensorflow as tf
from gensim.models import KeyedVectors
import argparse
import numpy as np
import tqdm
import sys
import codecs
import cc_expts
import sys
import sklearn.metrics as sklearn_metrics

EMB_SIZE = 300
NUM_CLASSES = 12
prefix = "__label__"


def macro(labels, preds, num_labels):
    labels = np.array(labels)
    preds = np.array(preds)
    accs = []
    for label in range(num_labels):
        lidxs = np.where(labels==label)
        this_labels, this_preds = labels[lidxs], preds[lidxs]
        accs.append(np.mean(this_labels==this_preds))
    #print accs
    # ignore nans meaning the label did not occur in this chunk
    return np.nanmean(np.array(accs)), np.nanmean(np.array(accs)[:num_labels//2])


def one_hot(ys, num_classes):
    oh = np.zeros(shape=[len(ys), num_classes])
    oh[:,ys] = 1
    return oh


def stats(preds, gold, smax_probs):
    """
    Use it only for binary labels case
    """
    assert len(preds)==len(gold)
    cm = sklearn_metrics.confusion_matrix(gold, preds)
    cm = cm.astype(np.float32)
    # print cm
    avg_p = [cm[i, i]/sum(cm[:, i]) for i in range(len(cm))]
    avg_r = [cm[i, i]/sum(cm[i, :]) for i in range(len(cm))]
    avg_p = map(lambda _: 0 if np.isnan(_) else _, avg_p)
    avg_r = map(lambda _: 0 if np.isnan(_) else _, avg_r)
    # print avg_p, avg_r
    
    avg_r, avg_p = sum(avg_r)/len(avg_r), sum(avg_p)/len(avg_p)
    probs = map(lambda li: smax_probs[li, 1], range(len(preds)))
    auc = sklearn_metrics.roc_auc_score(map(lambda _: -1 if _==0 else 1, gold), probs)
    return auc, avg_p, avg_r


def run(args):
    label_freqs = {}
    word_vecs = KeyedVectors.load_word2vec_format(args.embs, binary=True if args.embs.endswith('.bin') else False, unicode_errors='ignore', limit=400000)
    
    EMB_SIZE = args.embsize
    assert EMB_SIZE==word_vecs.vector_size
    with codecs.open(args.valid, "r", "utf-8") as f:
        valid_labels = [w[len(prefix):] for l in f for w in l.split() if w.startswith(prefix)]
    with codecs.open(args.test, "r", "utf-8") as f:
        test_labels = [w[len(prefix):] for l in f for w in l.split() if w.startswith(prefix)]
    with codecs.open(args.train, "r", "utf-8") as f:
        for l in f:
            for w in l.split():
                if w.startswith(prefix):
                    label = w[len(prefix):]
                    if label not in valid_labels or label not in test_labels:
                        continue
                    label_freqs[label] = label_freqs.get(label, 0) + 1
    label_freqs = label_freqs.items()
    
    label_freqs.sort(key=lambda _: _[1])
    # print ("Label Frequencies")
    # print (label_freqs)
    label_to_id = {}
    for li, k in enumerate(label_freqs):
        label_to_id[k[0]] = li
        
    NUM_CLASSES = len(label_to_id)
    #print ("Found: %d classes" % NUM_CLASSES)
                        
    ph_x = tf.placeholder(tf.float32, shape=[None, EMB_SIZE], name="input")
    ph_y = tf.placeholder(tf.int32, shape=[None])

    net = tf.layers.dense(ph_x, NUM_CLASSES, activation=tf.identity)
    smax_probs = tf.nn.softmax(logits=net)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=ph_y)
    )
    # hinge loss
    # loss = tf.reduce_mean(tf.maximum(1-tf.reduce_sum(net*
    #                                                  tf.one_hot(ph_y, NUM_CLASSES, on_value=1., off_value=-1.), axis=1), 0))
    # MSE
    # loss = tf.reduce_mean((net - tf.one_hot(ph_y, NUM_CLASSES))**2)
    
    preds = tf.argmax(net, axis=1)
    # tf_acc, update_op = tf.metrics.accuracy(ph_y, predictions = preds)
    tf_acc = tf.reduce_mean(tf.to_float(ph_y == preds))

    #loss = tf.Print(loss, data=[ph_x, ph_y, net], message="debug")
    #print loss
    opt = tf.train.AdamOptimizer()
    train_op = opt.minimize(loss)
    
    train_x, train_y = [], []
    valid_x, valid_y = [], []
    test_x, test_y = [], []
    test_content = []
    num_total, num_skipped = 0, 0
    if args.vocab is not None:
        sys.stderr.write("Loading vocab from: %s\n" % args.vocab)
        vocab = set(map(lambda _: _[0], cc_expts.load_vocab(args.vocab)))
        vocab = set(filter(lambda _: _ in word_vecs.vocab, vocab))
    else:
        vocab = word_vecs.vocab
    vecs = {}
    for w in vocab:
        vec = word_vecs.word_vec(w)
        vecs[w] = vec

    for fi, fname in enumerate([args.train, args.valid if args.valid is not None else args.test, args.test]):
        with codecs.open(fname, "r", "utf-8") as f:
            linenum = 0
            for l in tqdm.tqdm(f.readlines()):
                _dummy = [w[len(prefix):] for w in l.split() if w.startswith(prefix)]
                assert len(_dummy)==1, "Either more than one label or no label in line num: %d of train file -- %d labels found" % (linenum, len(_dummy))
                if _dummy[0] not in label_to_id:
                    continue
                y = label_to_id[_dummy[0]]

                wrds = [w for w in l.split() if w in vocab]
                num_total += 1
                if len(wrds)<2:
                    num_skipped += 1
                    continue
                wrds = wrds[:min(100, len(wrds))]
                x = np.array([vecs[w] for w in wrds])
                x = np.mean(x, axis=0)
                if fi==0:
                    train_x.append(x)
                    train_y.append(y)
                elif fi==1:
                    valid_x.append(x)
                    valid_y.append(y)
                else:
                    test_content.append(" ".join(wrds))
                    test_x.append(x)
                    test_y.append(y)
                linenum += 1

    # print ("%d of %d skipped" % (num_skipped, num_total))
    init_op = tf.global_variables_initializer()
    localvars_op = tf.local_variables_initializer()

    # print ("Starting training on %d train instances" % len(train_x))
    BATCH_SIZE = 100
    np_loss = 0
    best_valid_acc = -1E10
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        sess.run(localvars_op)
        num_steps_per_epoch = len(train_x)/BATCH_SIZE
        for epoch in range(args.nepochs):
            for step in range(num_steps_per_epoch):
                idxs = [np.random.randint(0, len(train_x)) for _ in range(BATCH_SIZE)]
                np_x = np.array([train_x[idx] for idx in idxs])
                np_y = np.array([train_y[idx] for idx in idxs])
                _, np_loss, np_preds, np_logits = sess.run([train_op, loss, preds, smax_probs], feed_dict={ph_x: np_x, ph_y: np_y})
                #if step%100==0:
                #    print ("Epoch: %d step: %d Loss: %f" % (epoch, step, np_loss))

            _, valid_preds, np_logits = sess.run([tf_acc, preds, smax_probs], feed_dict={ph_x: np.array(valid_x), ph_y: np.array(valid_y)})
            _m = macro(valid_y, valid_preds, NUM_CLASSES)
            valid_acc = np.mean(valid_y==valid_preds)
            valid_accs = (valid_acc, _m[0], _m[1])
            #valid_probs = np.mean(np.power(one_hot(valid_y, NUM_CLASSES) - np_logits, 2))
            valid_probs = np.mean(np.sum(-one_hot(valid_y, NUM_CLASSES)*np.log(np_logits), 1))
            # valid_accs = (valid_probs, _m[0], _m[1])
            
            _, test_preds, np_logits = sess.run([tf_acc, preds, smax_probs], feed_dict={ph_x: np.array(test_x), ph_y: np.array(test_y)})
            _m = macro(test_y, test_preds, NUM_CLASSES)
            test_acc = np.mean(test_y==test_preds)
            test_accs = (test_acc, _m[0], _m[1])
            # test_auc, test_precision, test_recall = stats(test_preds, test_y, np_logits)
            #test_accs = (test_auc, test_precision, test_recall)
            
            test_debug = []
            for ti in range(len(test_y)):
                test_debug.append("%d_%d %s" % (test_y[ti], test_preds[ti], test_content[ti]))
            #test_probs = np.mean(np.power(one_hot(test_y, NUM_CLASSES) - np_logits, 2))
            test_probs = np.mean(np.sum(-one_hot(test_y, NUM_CLASSES)*np.log(np_logits), 1))
            # test_accs = (test_probs, _m[0], _m[1])
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_valid_accs = valid_accs
                best_test_accs = test_accs
                best_test_debug = "\n".join(test_debug)

            sys.stderr.write ("\rEpoch: %d Loss: %f" % (epoch, np_loss))
            sys.stderr.flush()

        # print ("EMB file: %s"% args.embs)
        # print ("Validation micro accuracy: %f macro accuracy: %f infrequent classes: %f" % (best_valid_accs[0], best_valid_accs[1], best_valid_accs[2]))
        # print ("Evaluating on %d test instances..." % len(test_x))
        # print ("|" + args.embs + "|" + str(best_valid_acc) + "|" + "|".join(map(str, best_test_accs)) + "|")
        print ("|" + args.embs + "|" + str(best_valid_acc) + "|" + "|".join(map(str, test_accs)) + "|")
        #with codecs.open("%s_debug.txt" % args.embs, "w") as f:
        #    f.write(best_test_debug)
        #print ("Test: micro accuracy %f macro accuracy %f infrequent classes: %f" % (best_test_accs[0], best_test_accs[1], best_test_accs[2]))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-embsize', type=int, required=True, dest='embsize')
    parser.add_argument('-embs', type=str, required=True, dest='embs')
    parser.add_argument('-vocab', type=str, default=None, dest='vocab')
    parser.add_argument('-train', type=str, required=True, dest='train')
    parser.add_argument('-valid', type=str, required=False, dest='valid')
    parser.add_argument('-test', type=str, required=True, dest='test')
    parser.add_argument('-nepochs', type=int, default=20, dest='nepochs')
    parser.add_argument('-seed', type=int, default=0, dest='seed')

    args = parser.parse_args()
    sd = args.seed
    np.random.seed(sd)
    tf.set_random_seed(sd)
    run(args)
