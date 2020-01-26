This repository contains data and code used for ACL 2019 publication "Topic Sensitive Attention on Generic Corpora Corrects Sense Bias inPretrained Embeddings" https://arxiv.org/pdf/1906.02688.pdf

# Data
## Stack Exchange data
The folders with pattern: `data/{physics,gaming,unix,android}_{se,10mb}` contain the labeled duplicates and free text of size either 1MB (for folder name ending in 'se') or 10MB (for folder name ending in '10mb').  
Each of labeled/unlabeled train/test data contain 5 splits.  
* `content_{0,1,2,3,4}.txt, non_overlapping_content_{0,1,2,3,4}.txt` are the train and test data used in language model experiments.  
* `train_{0,1,2,3,4}.tsv`, `dev_{0,1,2,3,4}.tsv`, `test_{0,1,2,3,4}.tsv` are the train/dev/test splits for duplicate detection. 

Scripts used to prepare the data can be found in `scripts/data_prep`.

## Medical abstracts (Ohsumed data)
The folder: `data/ohsumed` contains supervised data in `{train,test}.txt` and the samll target text in `content.txt`.

# Code
## Embedding training 
The useflow includes the following steps:
1. Collect and augment your target data with topically relevant subset from a multi-topic and a large corpus such as Wikipedia. 
2. Train embeddings on your target data for ample number of epochs (200 epochs is optimal for 1MB of text).
3. Use the text collected in step 1, your smallish target data and embeddings trained in step 2 to train your final embedding. 

The following sections give more details for each section. 

### Step 1 -- Topic sensitive text retriever
Collect all relevant text snippets from a large text corpus such as Wikipedia.

The maven (Java) project under the folder `lens` can be used to collect target relevant content from Source. 

Follow the steps below.  
	1. Place your source (multi-topic generic corpora) text in a folder named `lens/vector_wiki` and name it `content.txt`.  
	2. Use the small target relevant text in a folder called `lens/vector_X` with the file name `content.txt` again and `X` is any name you would want to call it.   
	3. Once you have built the maven project, run using the command: `cd lens/; java -jar target/lens-0.01.one-jar.jar X 0.9 50  -Djava.util.concurrent.ForkJoinPool.common.parallelism=50` where 'X' is the target name assigned in step (2). See 'lens/src/main/java/Lens.java' for the meaning of the command line arguments.   
	
You should now find in the folder `vectors_X/ir_select/selected.txt`, `vectors_X/ir_select/ir-doc-scores.txt`. The former contains the selected snippets one per line and the latter file contains the corresponding doc score assigned to the text span. We also include topic irrelevant and random text spans (which is hard-coded to 5% of the total unselected text spans) towards the end of this file. The folder should also include `ir_select/selected-debug.txt` which gives more detail on why a snippet from wiki is picked by showing the vwords that triggered the selection and closest text span from the target corpus. 

### Step 2 -- Train your target embeddings
Use the standard word2vec code included in `src/word2vec.c` trained for sufficient number of epochs for this step.

### Step 3 -- Train embeddings on your augmented data
* Context score each of the tokens in the data with target embeddings using `scripts/score_word_context.py`.
* Finally train embeddings using `src/weighted_word2vec.c`

## Downstream tasks
As explained in the paper, we use simple archs for evaluation of embeddings.

**Document classification**
We used https://github.com/roomylee/rnn-text-classification-tf with `--cell_type vanilla`.

**Deduplication**
We use EMD distance from the embeddings and evaluate the goodness of the embeddings based on correlation of these scores with the actual label trough AUC score. One can use `scripts/emd_model.py` for this purpose.

Usage example 
```
python scripts/emd_model.py --fldr data/physics_se --wvs <Word vectors> --vocab data/physics_se/vocab.txt
```

**LM Experiments**
As mentioned in the paper, we used `https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py` for training language model.
