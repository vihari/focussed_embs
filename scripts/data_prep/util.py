def _get_vocab(fname):
    text = ""
    vocab = {}
    with open(fname) as f:
        for line in f.readlines():
            _l = line.strip().split("\t")
            for w in (_l[1]+ " " +_l[2]).split():
                vocab[w] = vocab.get(w, 0) + 1
    return vocab
            
def run():
    _dir = "gaming"
    content_vocab = {}
    with open("%s/content.txt" % _dir) as f:
        for line in f.readlines():
            for w in line.strip().split():
                content_vocab[w] = content_vocab.get(w, 0) + 1
    content_vocab = dict([i for i in content_vocab.items() if i[1]>=5])
                
    dev_vocab, test_vocab, train_vocab = _get_vocab("%s/dev.tsv" % _dir), _get_vocab("%s/test.tsv" % _dir), _get_vocab("%s/train.tsv" % _dir)

    print (dev_vocab.keys()[:10], train_vocab.keys()[:10], test_vocab.keys()[:10], content_vocab.keys()[:10])
    print ("Size of vocabulary of content %d" % len(content_vocab))
    print ("Intersection between train and content: %d/%d" % (len(set.intersection(set(train_vocab.keys()), set(content_vocab.keys()))), len(train_vocab)))
    print ("Intersection between test and content: %d/%d" % (len(set.intersection(set(test_vocab.keys()), set(content_vocab.keys()))), len(test_vocab)))
    print ("Intersection between dev and content: %d/%d" % (len(set.intersection(set(dev_vocab.keys()), set(content_vocab.keys()))), len(dev_vocab)))

if __name__=='__main__':
    run()
