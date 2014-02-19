WORD_OBSERVATION_RBM_NCE
========================

Train word embeddings with an RBM and Noise Contrastive Estimation training

INTRODUCTION
============
1. What are word embeddings ? 
  The embedding of a word is a vector of real numbers in 'n' dimensional space where 'n' is specified by the user. Good word embeddings will cluster similar words together. There has been a lot of work in NLP on improving classification tasks using word embeddings. For example
  a. Word representations: a simple and general method for semi-supervised learning. http://dl.acm.org/citation.cfm?id=1858721
  b. Natural Language Processing (almost) from Scratch. http://arxiv.org/abs/1103.0398
  ...


2. What is a restricted Boltzmann machine (RBM) ?
  A RBM is a stochastic model that can learn a joint probabilitity distribution over its inputs (http://en.wikipedia.org/wiki/Restricted_Boltzmann_machine). In this case, the inputs are n-grams of words. (The user can specify the order). The architecture of the RBM implemented in this tool is the same as http://www.cs.toronto.edu/~gdahl/papers/wrrbm_icml2012.pdf. 

3. What is noise contrastive estimation (NCE) ? 
 In http://www.cs.toronto.edu/~gdahl/papers/wrrbm_icml2012.pdf, the RBM was trained with contrastive divergence. This implementation trains the word RBM with Noise Contrastive Estimation (NCE). http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GutmannH10.pdf. NCE allows for a principled and easy way to train unnormalized statistical models. The goal is to estimate the parameters to differentiate between observed data and artifically generated noise. Unlike contrastive divergence, where the objective function is not clear, in NCE, we are optimizing a principled objective function whose progress we can track.
NCE has been employed to train neural language models with success. 
 a. A fast and simple algorithm for training neural probabilistic language models. In Proceedings of ICML, 2012
 b. Decoding with large-scale neural language models improves translation. Ashish Vaswani, Yinggong Zhao, Victoria Fossum, and David Chiang, 2013. In Proceedings of EMNLP (This paper is accompanied with very efficient code to train a large scale neural language model. You can find it here http://nlg.isi.edu/software/nplm/) 


DEPENDENCIES
============  

1. C++ compiler and GNU make
2. Boost 1.47.0 or later http://www.boost.org
3. Eigen 3.1.x http://eigen.tuxfamily.org
4. Python 2.7.x, not 3.x http://python.org (To run the data prep scripts)


COMPILING
=========
1. Go to the src directory
2. Modify the variables according to your environment
3. type Make
4. The binary RBMDahlNCE should be produced

USAGE
=====

1. Preparing your data:
  The program RBMDahlNCE needs 4 inputs 
  --train_file -> A training file of integerized space separated n-grams. The integers must be between 0 and V where V is the
                  Size of the vocabulary
  --words_file -> The vocabulary, one word per line where line 'i' is the word with integer representation 'i-1' (Assuming that lines start 
                  at index 1)
  --unigram_probs_file -> The unigram probabilities of the words. Each line has two tab separate entries: <integer_id_of_word>\t<unigram_probability_of_word>
  In prep_scripts/generateTrainingData.py can generate the data for you. 

2. Running the RBM:
  Please look at test/trainEmbeddings.sh for usage


  





