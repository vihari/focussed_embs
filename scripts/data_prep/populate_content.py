import query_cqadupstack as qcqa
import os
import tqdm

sf = "unix"
out_dir = "%s_se" % sf
o = qcqa.load_subforum(os.path.expanduser("/mnt/blossom/more/viharip/data/question_duplicates/%s.zip" % sf))

for name in ["train", "dev", "test"]:
    with open("%s/%s_content.txt" % (out_dir, name), "w") as f:
        rf = open("%s/%s.tsv" % (out_dir, name))
        all_titles = set()
        for l in rf:
            _, a, b = l.strip().split('\t')
            all_titles.add(a)
            all_titles.add(b)

        pids = o.get_all_postids()
        for pid in tqdm.tqdm(pids, "Processing: %s.tsv" % name):
            t = o.perform_cleaning(o.get_posttitle(pid))
            if t in all_titles:
                # ans_ids = o.get_answers(pid)
                # comment_ids = o.get_post_comments(pid) + [_ for _a in ans_ids for _ in o.get_answer_comments(_a)]
                # text = t + u"\n"
                # for _id in ans_ids:
                #     text += o.perform_cleaning(o.get_answerbody(_id)).strip() + u"\n"
                # for _id in comment_ids:
                #     text += o.perform_cleaning(o.get_commentbody(_id).strip()) + u"\n\n"
                ans_id = o.get_acceptedanswer(pid)
                if not ans_id: continue
                text = o.get_answerbody(ans_id).strip() + u"\n"
                text = o.perform_cleaning(o.get_answerbody(ans_id).strip()) + u"\n"
                f.write(("%s" % text).encode('utf-8'))
        rf.close()

with open("%s/non_overlapping_content.txt" % out_dir, "w") as f:
    all_titles = set()
    for name in ["train", "dev", "test"]:
        with open("%s/%s.tsv" % (out_dir, name)) as rf:
            for l in rf:
                _, a, b = l.strip().split('\t')
                all_titles.add(a)
                all_titles.add(b)

    pids = o.get_all_postids()
    for pid in tqdm.tqdm(pids, "Fetching content non-overlapping with any of splits"):
        t = o.perform_cleaning(o.get_posttitle(pid))
        if t not in all_titles:
            ans_id = o.get_acceptedanswer(pid)
            if not ans_id: continue
            text = o.get_answerbody(ans_id).strip() + u"\n"
            text = o.perform_cleaning(o.get_answerbody(ans_id).strip()) + u"\n"
            f.write(("%s" % text).encode('utf-8'))
