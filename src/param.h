#pragma once

#include <string>
#include <ctime>

using namespace std;

typedef struct PARAM{

		string train_file;
    string unigram_probs_file;
    string validation_file;
    int ngram_size;
    int n_vocab;
    double normalization_init;
    int n_hidden;
    int n_threads;
    int num_noise_samples;
    int embedding_dimension;
    int minibatch_size;
    int validation_minibatch_size;
    bool persistent;
    int num_epochs;
    double learning_rate;
    bool use_momentum;
    double initial_momentum;
    double final_momentum;
    double L2_reg;
    bool init_normal;
    string embeddings_prefix;
    string words_file;

} param;


