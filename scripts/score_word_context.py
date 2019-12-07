#!/usr/bin/python
"""
A hacky script to prepare for pre-processing of the selected docs with their score and emits a score for each word and the context pair.
"""
import argparse
from gensim.models import KeyedVectors
import os
import re
import codecs
import argparse
import numpy as np
import tqdm
import math
from nltk.tokenize import sent_tokenize


def cosine_sim(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1==0 or n2==0: return 0
    
    u1 = v1/np.linalg.norm(v1)
    u2 = v2/np.linalg.norm(v2)
    return np.dot(u1, u2)


def write_score_files(flags):
    sf = open(flags.rscores)
    pcf = codecs.open("%s" % flags.content, "r", "utf-8")

    # word and context ir score
    s2 = codecs.open("vectors_%s/wc_emb_%s.txt" % (flags.domain, flags.suffix), "w", "utf-8")
    
    vecs = KeyedVectors.load_word2vec_format(flags.tgt, binary=True)
    ws = 5
    emb_size = vecs.vector_size
    mgr = tqdm.tqdm(total=80000)
    ignored = set()

    for cl in pcf:
        mgr.update()
        sl = sf.readline()
        try:
            ir_score = max(0, float(sl.strip()))
        except ValueError:
            print (sl.strip())
            raise ValueError("::%s::" % sl.strip())
        emb_scores = []
        ir_scores = []
        #emb_scores3 = []
        tokens = cl.strip().split()

        if (len(tokens)==0): continue
        for wi in range(len(tokens)):
            ir_scores.append(ir_score)
            if tokens[wi] not in vecs:
                emb_scores.append(0.)
                # emb_scores3.append(0.)
                continue
            cemb = np.zeros([emb_size])
            numc = 0
            for wj in range(max(0, wi-ws), min(len(tokens), wi+ws)):
                # omit same token in the context so as not to inflate the score
                if wi!=wj and (tokens[wj] in vecs) and (tokens[wi]!=tokens[wj]):
                    numc += 1
                    cemb += vecs.word_vec(tokens[wj])
            if numc==0:
                emb_scores.append(0.)
                #emb_scores3.append(0.)
            else:
                _cs = cosine_sim(cemb/numc, vecs.word_vec(tokens[wi]))
                emb_scores.append(_cs)
                #emb_scores3.append(math.exp(_cs))

        #emb_scores4 = [sum(emb_scores3)/len(emb_scores3)]*len(emb_scores3)
        emb_scores2 = [sum(emb_scores)/len(emb_scores)]*len(emb_scores)
        # assert(len(emb_scores)==len(tokens) and len(ir_scores)==len(tokens))
        s2.write(" ".join(["%s:::%0.2f" % (tokens[_i], emb_scores[_i]) for _i in range(len(tokens))]) + "\n")

    print ("Ignored %d because they are missing from vocab" % len(ignored))
    s2.close()
    pcf.close()
    sf.close()
    

def run():
    # python scripts/score_word_context.py --domain physics_small --content vectors_physics_small/selected.txt --rscores vectors_physics_small/ir-doc-scores.txt --tgt physics_wvs.txt
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, help='Domain', required=True)
    parser.add_argument('--content', type=str, help='File containing the selected mini-documents', required=True)
    parser.add_argument('--rscores', type=str, help='Retrieval scores', required=True)
    parser.add_argument('--suffix', type=str, default='', help='Suffix used to uniquely identify the output file', required=False)
    parser.add_argument('--tgt', type=str, help="File name of the word vectors trained on target content", required=True)
    args, unparsed = parser.parse_known_args()
    TGT = args.domain
    print ("Domain : %s" % TGT)
    if not os.path.exists("vectors_%s" % TGT):
        os.mkdir("vectors_%s" % TGT)

    write_score_files(args)


if __name__=='__main__':
    run()
