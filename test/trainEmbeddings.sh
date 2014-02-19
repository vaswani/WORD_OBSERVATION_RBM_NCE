#! /bin/bash

#first extract training data

python ../prep_scripts/generateTrainingData.py --ngram_size 3 --vocab_size 5000 --output_training_file train --output_probs_file probs --output_validation_file validation --train_words_file words.train --input_validation_file words.validation --output_vocab_file words

export OMP_NUM_THREADS=2 ;../src/RBMDahlNCE --embedding_dimension 50 --n_vocab 5000 --train_file train --ngram_size 3 --unigram_probs_file probs --minibatch_size 128 --n_hidden 100 --learning_rate 1. --num_epochs 10 --words_file words --embeddings_prefix embeddings --use_momentum 0 --num_noise_samples 128 --validation_file validation --n_threads 2 --normalization_init 50

