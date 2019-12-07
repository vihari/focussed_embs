"""
Writes the text of the forum to a file
prepares and writes to file the question duplication pairs to a file

1. Get all duplicate pairs
2. Split them such train, test and dev do not share any post id
3. Add random non-duplicate pairs to train. Then to test and dev such that no post id is repeated.
"""
import query_cqadupstack as qcqa
import os
import tqdm
import random
import argparse
import codecs

sfs = ['gaming.zip',  'mathematica.zip',  'physics.zip',       'stats.zip',  'webmasters.zip',
 'android.zip', 'gis.zip' , 'programmers.zip',  'tex.zip',	 'wordpress.zip',
 'english.zip', 'unix.zip']

parser = argparse.ArgumentParser()
parser.add_argument('--domain', type=str, help='Domain', required=True)
args, unparsed = parser.parse_known_args()

sf = args.domain
out_dir = "%s_10mb" % sf
o = qcqa.load_subforum(os.path.expanduser("/mnt/blossom/more/viharip/data/question_duplicates/%s.zip" % sf))
labels = {'dup': 1, 'nondup': 0}
dup_pairs = o.get_all_duplicate_pairs()
print len(dup_pairs)

train_f, dev_f = .5, .1

for attempt in range(5):
    train_pairs, test_pairs, dev_pairs = [], [], []
    random.seed(attempt)
    # sort so that same question id is present the least in other splits
    dup_pairs.sort()
    for i, dup_pair in enumerate(dup_pairs):
        dp1, dp2 = dup_pair
        #if len(train_pairs)<len(dup_pairs)*train_f:
        _r = random.random()
        if _r<train_f:
            train_pairs.append((dp1, dp2, 1))
        #elif len(dev_pairs)<len(dup_pairs)*dev_f:
        elif _r<(train_f+dev_f):
            dev_pairs.append((dp1, dp2, 1))
        else:
            test_pairs.append((dp1, dp2, 1))


    target_len = 5*len(train_pairs)
    print ("Collecting -ve examples for train...")
    while(len(train_pairs) < target_len):
        p1, p2, label = o.get_random_pair_of_posts()

        if label!='nondup':
            continue
        train_pairs.append([p1, p2, 0])
    print ("Done")

    target_len = 5*len(dev_pairs)
    print ("Collecting -ve examples for dev...")
    while(len(dev_pairs) < target_len):
        p1, p2, label = o.get_random_pair_of_posts()
        if label!='nondup' or (p1, p2, 0) in train_pairs:
            continue
        dev_pairs.append([p1, p2, 0])
    print ("Done")

    target_len = 5*len(test_pairs)
    print ("Collecting -ve examples for test...")
    while(len(test_pairs) < target_len):
        p1, p2, label = o.get_random_pair_of_posts()
        if label!='nondup' or (p1, p2, 0) in train_pairs or (p1, p2, 0) in dev_pairs:
            continue
        test_pairs.append([p1, p2, 0])
    print ("Done")

    random.shuffle(train_pairs)
    random.shuffle(test_pairs)
    random.shuffle(dev_pairs)

    num_ones = lambda x : len([1 for _ in x if _[2]==1])
    num_zeros = lambda x : len([0 for _ in x if _[2]==0])

    print ("Collected %d (%d, %d) %d (%d, %d) %d (%d, %d) train, test and dev pairs" %
           (len(train_pairs), num_ones(train_pairs), num_zeros(train_pairs), len(test_pairs), num_ones(test_pairs), num_zeros(test_pairs), len(dev_pairs), num_ones(dev_pairs), num_zeros(dev_pairs)))
    fnames = ["%s/train_%d.tsv" % (out_dir, attempt), "%s/test_%d.tsv" % (out_dir, attempt), "%s/dev_%d.tsv" % (out_dir, attempt)]
    pairs = [train_pairs, test_pairs, dev_pairs]
    all_selected = [p for pair in train_pairs for p in pair]
    all_selected += [p for pair in test_pairs for p in pair]
    all_selected += [p for pair in dev_pairs for p in pair]

    for i in range(len(fnames)):
        with codecs.open(fnames[i], "w", "utf-8") as f:
            for p in pairs[i]:
                p1, p2, label = p
                t1, t2 = o.perform_cleaning(o.get_posttitle(p1)), o.perform_cleaning(o.get_posttitle(p2))
                f.write('\t'.join([str(label), t1, t2]))
                f.write('\n')

    content_file = codecs.open("%s/content_%d.txt" % (out_dir, attempt), "w", "utf-8")
    no_content_file = codecs.open("%s/non_overlapping_content_%d.txt" % (out_dir, attempt), "w", "utf-8")
    num_content = 0
    for pid in tqdm.tqdm(o.get_all_postids()):
        if pid not in all_selected:
            r = random.random()
            if False:
                ans_id = o.get_acceptedanswer(pid)
                if not ans_id: continue
                # text = o.get_answerbody(ans_id).strip() + "\n"
                text = o.perform_cleaning(o.get_answerbody(ans_id).strip()) + "\n"
            else:
                ans_ids = o.get_answers(pid)
                text = ""
                for ans_id in ans_ids:
                    text += o.perform_cleaning(o.get_answerbody(ans_id).strip()) + "\n"
            if r<0.5 or num_content>10000000: #num_content>2000:
                no_content_file.write(text)
            else:
                # num_content += 1
                num_content += len(text)
                content_file.write(text)
                
            
    # with codecs.open("%s/content_%d.txt" % (out_dir, attempt), "w", "utf-8") as f:
    #     for p in tqdm.tqdm(train_ids):
    #         ans_ids = o.get_answers(p)
    #         comment_ids = o.get_post_comments(p) + [_ for _a in ans_ids for _ in o.get_answer_comments(_a)]
    #         text = ""
    #         #text = o.perform_cleaning(o.get_post_title_and_body()).strip() + u"\n"
    #         for _id in ans_ids:
    #             text += o.perform_cleaning(o.get_answerbody(_id)).strip() + u"\n"
    #         for _id in comment_ids:
    #             text += o.perform_cleaning(o.get_commentbody(_id).strip()) + u"\n\n"
    #         f.write("%s" % text)
        
# dup_lines = set()

# for i in tqdm.tqdm(range(2*N)):
#     p1, p2, label = o.get_random_pair_of_posts()
#     if label in labels:
#         t1, t2 = o.get_posttitle(p1), o.get_posttitle(p2)
#         dup_lines.append("%d\t%s\t%s\t%d%d" % (label, t1, t2, p1, p2))

#     if len(dup_lines)>=N:
#         break

# with open("physics_text/dups.tsv", "w") as f:
#     for l in dup_lines:
#         f.write(l+"\n")
