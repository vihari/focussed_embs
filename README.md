This repository contains data and code used for ACL 2019 publication "Topic Sensitive Attention on Generic Corpora Corrects Sense Bias inPretrained Embeddings" https://arxiv.org/pdf/1906.02688.pdf

# Data
## Stack Exchange data
The folders with pattern: `data/{physics,gaming,unix,android}_{se,10mb}` contain the labeled duplicates and free text spanning either 1MB (for folder name ending in 'se') or 10MB (for folder name ending in '10mb').  
Each of labeled/unlabeled train/test data contain 5 splits.  
* `content_{0,1,2,3,4}.txt, non_overlapping_content_{0,1,2,3,4}.txt` are the train and test data used in language model experiments.  
* `train_{0,1,2,3,4}.tsv`, `dev_{0,1,2,3,4}.tsv`, `test_{0,1,2,3,4}.tsv` are the train/dev/test splits for duplicate detection. 

Scripts used to prepare the data can be found in `scripts/data_prep`.

## Medical abstracts (Ohsumed data)
The folder: `data/ohsumed` contains supervised data in `{train,test}.txt` and the samll target text in `content.txt`.

# Code

## Topic sensitive text retriever
The crucial part of the algorithm is collection of relevant snippets of text from a large generic corpus such as Wikipedia.

The maven (Java) project under teh folder `lens` can be used to collect target relevant content from Source. 

Follow the steps below.  
	1. Place your source (multi-topic generic corpora) text in a folder named `lens/vector_wiki` and name it `content.txt`.  
	2. Use the small target relevant text in a folder called `lens/vector_X` with the file name `content.txt` again.   
	3. Once you have built the maven project, run using the command: `cd lens/; java -jar target/lens-0.01.one-jar.jar X 0.9 50  -Djava.util.concurrent.ForkJoinPool.common.parallelism=50` where 'X' is the target name assigned in step (2). See 'lens/src/main/java/Lens.java' for the meaning of the command line arguments.   
	

	
