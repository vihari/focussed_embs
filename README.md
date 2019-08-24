This repository contains data and code used for ACL 2019 publication "Topic Sensitive Attention on Generic Corpora Corrects Sense Bias inPretrained Embeddings" https://arxiv.org/pdf/1906.02688.pdf

# Data
The folders: *physics_se, gaming_se, android_se, unix_se* contains the deduplication dataset from stackexchange domain bearing the same name.
Each of these folders contain the following files and utility as shown below:  
- train_{0,1,2,3,4}.tsv, dev_{0,1,2,3,4}.tsv, test_{0,1,2,3,4}.tsv contain questions along with their label split in to train, dev and test splits respectively. The numbers reported in the paper is performance averaged across splits corresponding to the first three numbeals.
- content_{0,1,2,3,4}.txt contains small sampled content from questions, answers or comments. The size of the text samples in these folders are restricted to around 1MB. The numbers reported in the paper are averaged over using either of content_{0,1,2}.txt as the available target related text.
- non_overlapping_content_{0,1,2,3,4}.txt is used for reporting PPL numbers.

The folders *{physics,gaming,android,unix}*_10mb are similar but with the only different that the sampled text from the target domain is size constrained to 10MB unlike the earlier 1MB. 

# Code

## Topic sensitive text retriever
The crucial part of the algorithm is collection of relevant snippets of text from a large generic corpus such as Wikipedia.

The maven (Java) project can be used to collect target relevant content from Source. 

Follow the steps below.  
	1. Place your source (multi-topic generic corpora) text in a folder named `lens/vector_wiki` and name it `content.txt`.
	2. Use the small target relevant text in a folder called `lens/vector_X` with the file name `content.txt` again. 
	3. Once you have built the maven project, run using the command: `cd lens/; java -jar target/lens-0.01.one-jar.jar X 0.9 200` where 'X' is the target name assigned in step (2). See 'lens/src/main/java/Lens.java' for the meaning of the command line arguments. 
	
	
