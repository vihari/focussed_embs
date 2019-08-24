#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
const int EMB_SIZE = 300;

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  char *word;
};

struct stats_word {
  float pos_mean, pos_var;
  float neg_mean, neg_var;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
// stats:: file containing mean and variance for positive and negative contexts for each word
char read_vocab_file[MAX_STRING], read_stats_file[MAX_STRING], read_vecs_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 2, window = 5;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *tgt_vecs;
struct stats_word *stats;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;


// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin, char *eof) {
  int a = 0, ch;
  while (1) {
    ch = fgetc_unlocked(fin);
    if (ch == EOF) {
      *eof = 1;
      break;
    }
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin, char *eof) {
  char word[MAX_STRING], eof_l = 0;
  ReadWord(word, fin, &eof_l);
  if (eof_l) {
    *eof = 1;
    return -1;
  }
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

void ReadVocab() {
  long long a, i = 0;
  char c, eof = 0;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin, &eof);
    if (eof) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);

  fin = fopen(train_file, "rb");
  while(1) {
    ReadWord(wrd, fin, &eof);
    train_words += 1;
    if (eof) break;
  }
  fclose(fin);
}

void ReadVecs() {
  long long a, i = 0;
  char c, eof = 0;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vecs_file, "rb");
  a = posix_memalign((void **)&tgt_vecs, 128, (long long)vocab_size * EMB_SIZE * sizeof(real));
  if (fin == NULL) {
    printf("Vecs file not found\n");
    exit(1);
  }
  while (1) {
    ReadWord(word, fin, &eof);
    if (eof) break;
    a = SearchVocab(word);
    int j;
    if (a==-1) {
      real temp;
      for (j=0; j<EMB_SIZE; j++) fscanf(fin, "%f", &temp);
      fscanf(fin, "%c", &c);
      continue;
    }

    for (j=0; j<EMB_SIZE; j++)
      fscanf(fin, "%f", &tgt_vecs[a*EMB_SIZE + j], &c);
    fscanf(fin, "%c", &c);
    i++;
  }
  fclose(fin);
}

void ReadStats() {
  long long a, i = 0;
  char c, eof = 0;
  char word[MAX_STRING];
  FILE *fin = fopen(read_stats_file, "rb");
  a = posix_memalign((void **)&stats, 128, (long long)vocab_size * EMB_SIZE * sizeof(struct stats_word));
  if (fin == NULL) {
    printf("Stats file not found\n");
    exit(1);
  }
  while (1) {
    ReadWord(word, fin, &eof);
    if (eof) break;
    a = SearchVocab(word);
    int j;
    fscanf(fin, "%f%f%f%f%c", &stats[a].pos_mean, &stats[a].pos_var, &stats[a].pos_mean, &stats[a].pos_var , &c);
    i++;
  }
  fclose(fin);
}

real CosSim(real* v1, real* v2){
  int i = 0;
  real d1 = 0, d2 = 0, d3 = 0;
  for (i=0; i<EMB_SIZE; i++){
    d1 += v1[i]*v2[i];
    d2 += v1[i]*v1[i];
    d3 += v2[i]*v2[i];
  }
  return d1/(sqrt(d2)*sqrt(d3));
}

void ScoreContexts(){
  ReadVocab();
  ReadVecs();
  ReadStats();
  
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  short trainable[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label;
  unsigned long long next_random = 1;
  char eof = 0;
  real f, g;
  clock_t now;

  real *neu1 = (real *)calloc(EMB_SIZE, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  FILE *ofi = fopen(output_file, "wb");

  while(1) {
    if (word_count - last_word_count > 10000) {
      last_word_count = word_count;
      now=clock();
      printf("%cProgress: %.2f%%  Words/thread/sec: %.2fk  ", 13,
	     word_count / (real)(train_words + 1) * 100,
	     word_count / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
      fflush(stdout);
    }
    
    if (sentence_length == 0) {
      while (1) {
	fprintf(ofi, "\n");
	char lit_word[MAX_STRING];
	ReadWord(lit_word, fi, &eof);
	word = SearchVocab(lit_word);
	if (eof) break;
	word_count++;
	if (word == -1) continue;
        if (word == 0) break;
        sen[sentence_length] = word;
	trainable[sentence_length] = -1;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (eof) break;
    word = sen[sentence_position];
    if (word == -1) continue;
    next_random = next_random * (unsigned long long)25214903917 + 11;

    cw = 0;
    // window + 1 because we have added sampling inside this loop
    for (a = 0; a < (window + 1) * 2 + 1; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
	
	// The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[last_word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[last_word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
	
        for (c = 0; c < EMB_SIZE; c++) neu1[c] += tgt_vecs[c + last_word * EMB_SIZE];
        cw++;
      }
    real score = 0;
    if (cw) {
      for (c = 0; c < EMB_SIZE; c++) neu1[c] /= cw;
      real sim = CosSim(&tgt_vecs[word*EMB_SIZE], neu1);
      struct stats_word st = stats[word];
      real d1 = pow(sim - st.pos_mean, 2)/st.pos_var;
      real d2 = pow(sim - st.neg_mean, 2)/st.neg_var;

      if (d1<d2) score = tanh(d2-d1);
    }

    if (score > 0) fprintf(ofi, "%s__%0.3f ", vocab[word].word, score);
    else fprintf(ofi, "%s ", vocab[word].word);
      
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(ofi);
  fclose(fi);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char** argv) {
  if (argc == 1) {
    printf("Context Scorer\n\n");
    printf("Options:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to assign scores to contexts\n");
    printf("\t-read-stats <file>\n");
    printf("\t\tFile containing context scoring related stats\n");
    printf("\t-read-vecs <file>\n");
    printf("\t\tPointer to word vector model that is used to score contexts\n");
    printf("\t-output <file>\n");
    printf("\t\tContext scores appended to words written to this file\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./context_scorer -train data.txt -output trainable_data.txt -window 5 -read-vecs vectors_physics_se/cbow_vectors.bin -cbow 1 -read-stats vectors_physics_se/stats.txt\n\n");
    return 0;
  }

  int i;
  output_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vecs", argc, argv)) > 0) strcpy(read_vecs_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-stats", argc, argv)) > 0) strcpy(read_stats_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int*) calloc(vocab_hash_size, sizeof(int));
  ScoreContexts();
}
