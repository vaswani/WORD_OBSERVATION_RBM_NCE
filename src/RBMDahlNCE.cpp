#include <iostream>
#include <list>
#include <ctime>
#include <cstdio>

#include <tclap/CmdLine.h>
#include <boost/algorithm/string/join.hpp>

#include "param.h"
#include "util.h"
#include "RBM.h"
#include "RBMDahlFunctions.h"
//#include<tr1/random>
#include <time.h>
//#include <chrono>
//#include <random>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <stdio.h>
#include <iomanip>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <omp.h>
//#define EIGEN_DONT_PARALLELIZE
using namespace std;
using namespace TCLAP;
using namespace Eigen;
using namespace boost::random;

typedef boost::unordered_map<vector<int>, double> vector_map;


int main(int argc, char** argv)
{
    //omp_set_num_threads(4);
    //cerr<<"the number of threads is "<<omp_get_num_threads()<<endl;

    //Eigen::initParallel();
    //srand ( time(NULL) );
    //srand(time(0));
    //tr1::mt19937 eng;
    //tr1::mt19937 eng2;
    //unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	  unsigned seed = std::time(0);
    //unsigned test_seed = 1234; //for testing only
    mt19937 eng_int (seed);  // mt19937 is a standard mersenne_twister_engine, I'll pass this to sample h given v
    mt19937 eng_real (seed);  // mt19937 is a standard mersenne_twister_engine

    //std::cerr << "Random value: " << generator() << std::endl;

    param rbmParam;
    try{
      // program options //
      CmdLine cmd("Command description message ", ' ' , "1.0");

      ValueArg<string> train_file("", "train_file", "training file. Default:train" , true, "string", "train", cmd);
      ValueArg<string> words_file("", "words_file", "words file. Default:words" , true, "string", "words", cmd);
      ValueArg<string> validation_file("", "validation_file", "validation file. Default:validation" , true, "validation", "string", cmd);
      ValueArg<string> unigram_probs_file("", "unigram_probs_file", "unigram probs file. Default:unigram_probs" , true, "unigram_probs", "string", cmd);
      ValueArg<string> embeddings_prefix("", "embeddings_prefix", "embedding prefix for the embeddings. Default:embeddings.cpp.epoch" , false, "embeddings.cpp.epoch", "string", cmd);

      ValueArg<int> ngram_size("", "ngram_size", "The size of ngram that you want to consider. Default:3", false, 3, "int", cmd);

      ValueArg<int> n_vocab("", "n_vocab", "The vocabulary size. This has to be supplied by the user.Default:1000", true, 1000, "int", cmd);

      ValueArg<int> n_hidden("", "n_hidden", "The number of hidden nodes. Default 100", false, 100, "int", cmd);
      ValueArg<int> n_threads("", "n_threads", "The number of threads. Default:1", false, 1, "int", cmd);
      ValueArg<int> num_noise_samples("", "num_noise_samples", "The number of noise samples. Default:100", false, 100, "int", cmd);


      ValueArg<int> embedding_dimension("", "embedding_dimension", "The size of the embedding dimension. Default 50", false, 10, "int", cmd);

      ValueArg<int> minibatch_size("", "minibatch_size", "The minibatch size. Default: 64", false, 64, "int", cmd);
      ValueArg<int> validation_minibatch_size("", "validation_minibatch_size", "The validation set minibatch size. Default: 64", false, 64, "int", cmd);
      
      ValueArg<int> num_epochs("", "num_epochs", "The number of epochs. Default:10 ", false, 10, "int", cmd);

      ValueArg<double> learning_rate("", "learning_rate", "Learning rate for training. Default:0.0.25", false, 0.25, "double", cmd);
      ValueArg<double> normalization_init("", "normalization_init", "The initial normalization parameter. Default:8.43385", false,20, "double", cmd);

      ValueArg<bool> use_momentum("", "use_momentum", "Use momentum during training or not. Default:0", false, 0, "bool", cmd);
      ValueArg<double> initial_momentum("", "initial_momentum", ".Initial value of momentum. Default:0.9", false, 0.9, "double", cmd);
      ValueArg<double> final_momentum("", "final_momentum", "Final value of momentum. Default:0.9", false, 0.9, "double", cmd);

      //ValueArg<bool> persistent("", "persistent", "Use persistent CD or not. Default:0", false, 0, "bool", cmd);

      ValueArg<double> L2_reg("", "L2_reg", "The L2 regularization weight. Weight decay is only applied to the U parameter. Default:0.00001", false, 0.00001, "double", cmd);

      ValueArg<bool> init_normal("", "init_normal", "Initialize parameters form a normal distribution Default:1", false, 1, "bool", cmd);
      cmd.parse(argc, argv);

      // define program parameters //
      rbmParam.train_file = train_file.getValue();
      rbmParam.validation_file= validation_file.getValue();
      rbmParam.unigram_probs_file= unigram_probs_file.getValue();
      rbmParam.ngram_size = ngram_size.getValue();
      rbmParam.n_vocab= n_vocab.getValue();
      rbmParam.n_hidden= n_hidden.getValue();
      rbmParam.n_threads  = n_threads.getValue();
      rbmParam.normalization_init = normalization_init.getValue();
      rbmParam.num_noise_samples = num_noise_samples.getValue();
      rbmParam.embedding_dimension = embedding_dimension.getValue();
      rbmParam.minibatch_size = minibatch_size.getValue();
      rbmParam.validation_minibatch_size = validation_minibatch_size.getValue();
      rbmParam.num_epochs= num_epochs.getValue();
      //rbmParam.persistent = persistent.getValue();
      rbmParam.learning_rate = learning_rate.getValue();
      rbmParam.use_momentum = use_momentum.getValue();
      rbmParam.embeddings_prefix = embeddings_prefix.getValue();
      rbmParam.words_file = words_file.getValue();
      rbmParam.initial_momentum = initial_momentum.getValue();
      rbmParam.final_momentum = final_momentum.getValue();
      rbmParam.L2_reg = L2_reg.getValue();
      rbmParam.init_normal= init_normal.getValue();

      // print program command to stdout//

      cerr << "Command line: " << endl;

      for (int i = 0; i < argc; i++)
      {
        cerr << argv[i] << endl;
      }
      cerr << endl;

      cerr << train_file.getDescription() << " : " << train_file.getValue() << endl;
      cerr << unigram_probs_file.getDescription() << " : " << unigram_probs_file.getValue() << endl;
      cerr << ngram_size.getDescription() << " : " << ngram_size.getValue() << endl;
      cerr << embedding_dimension.getDescription() << " : " << embedding_dimension.getValue() << endl;
      cerr << n_hidden.getDescription() << " : " << n_hidden.getValue() << endl;
      cerr << n_threads.getDescription() << " : " << n_threads.getValue() << endl;
      cerr << num_noise_samples.getDescription() << " : " << num_noise_samples.getValue() << endl;
      cerr << n_vocab.getDescription() << " : " << n_vocab.getValue() << endl;
      cerr << normalization_init.getDescription() << " : " << normalization_init.getValue() << endl;
      cerr << minibatch_size.getDescription() << " : " << minibatch_size.getValue() << endl;
      cerr << validation_minibatch_size.getDescription() << " : " << minibatch_size.getValue() << endl;
      cerr << num_epochs.getDescription() << " : " << num_epochs.getValue() << endl;
      //cerr << persistent.getDescription() << " : " << persistent.getValue() << endl;
      cerr << learning_rate.getDescription() << " : " << learning_rate.getValue() << endl;
      cerr << use_momentum.getDescription() << " : " << use_momentum.getValue() << endl;
      cerr << words_file.getDescription() << " : " << words_file.getValue() << endl;
      cerr << initial_momentum.getDescription() << " : " << initial_momentum.getValue() << endl;
      cerr << final_momentum.getDescription() << " : " << final_momentum.getValue() << endl;
      cerr << L2_reg.getDescription() << " : " << L2_reg.getValue() << endl;
      cerr << embeddings_prefix.getDescription() << " : " << embeddings_prefix.getValue() << endl;
      cerr << init_normal.getDescription() << " : " << init_normal.getValue() << endl;

    }
    catch (TCLAP::ArgException &e)
    {
      cerr << "error: " << e.error() <<  " for arg " << e.argId() << endl;
      exit(1);
    }
    cerr<<"train file is "<<rbmParam.train_file<<endl;
    //read the training file
    //
    vector<vector<int> > unshuffled_training_data;
    //cerr<<"the size of unshuffled training data is"<<unshuffled_training_data.size()<<endl;
    readTrainFile(rbmParam,unshuffled_training_data);
	  int training_data_size = unshuffled_training_data.size();
    cerr<<"Training data size was "<<training_data_size<<endl;
    vector<vector<int> > validation_set_vector;
    readDataFile(rbmParam.validation_file,rbmParam,validation_set_vector);
    cerr<<"read the validation file"<<endl;

	  //now shuffling training data
	  random_shuffle ( unshuffled_training_data.begin(), unshuffled_training_data.end() );

    //now dump the training data to a temp file
    writeTempData(unshuffled_training_data,rbmParam);

    //clearing the data vector so that we dont' run out of memory
    unshuffled_training_data.clear();
    int validation_set_size = validation_set_vector.size();
    Matrix<int,Dynamic,Dynamic>  shuffled_training_data,validation_set(validation_set_vector.size(),rbmParam.ngram_size);
    shuffled_training_data.setZero(training_data_size,rbmParam.ngram_size);

    //storing the training data in the training matrix and the validation matrix
    cerr<<"storing the training data into the eigen matrix"<<endl;
    
    readTrainFileMatrix("temp.dat",rbmParam,shuffled_training_data);
    cerr<<"storing the validation data into the eigen matrix"<<endl;
    for (int i = 0;i<validation_set_vector.size();i++)
    {
        //string joined = boost::algorithm::join(unshuffled_training_data[i], " ");
        for (int j=0;j<rbmParam.ngram_size;j++)
        {
            validation_set(i,j) = validation_set_vector[i][j];
        }
    }

    validation_set_vector.clear();
    //cerr<<"the size of the training data is "<<shuffled_training_data.rows()<<endl;
    vector<string> word_list;
    cerr<<"Reading words file"<<endl;
    readWordsFile(rbmParam.words_file, word_list);
    cerr<<" Word list size is "<<word_list.size()<<endl;

    //reading the unigram probs
    vector<double> unigram_probs = vector<double>(rbmParam.n_vocab);
    readUnigramProbs(rbmParam,unigram_probs);
    cerr<<"Setting up the alias sampler"<<endl;
    //now i have the unigram probs, I need to setup alias method
    vector<int> J(rbmParam.n_vocab,-1);
    vector<double> q(rbmParam.n_vocab,0.);
    setupAliasMethod(unigram_probs,J ,q,rbmParam.n_vocab);
    //cerr<<"q is "<<q.size()<<endl;
    //cerr<<"J is "<<J.size()<<endl;
    //for multithreading, I will make copies of q and J, 
    //unigram probs, and the random number generators 
    //one for each thread
    vector<vector<int> > J_vector;
    vector<vector<double> > q_vector; 
    vector<vector<double> > unigram_probs_vector;
    vector<mt19937> eng_int_vector; 
    vector<mt19937> eng_real_vector; 
    vector<uniform_int_distribution<> >unif_int_vector;
    vector<uniform_real_distribution<> >unif_real_vector;

    for (int i=0;i<rbmParam.n_threads;i++)
    {
        vector<int> temp_J = J;
        J_vector.push_back(temp_J);
        vector<double> temp_q = q;
        q_vector.push_back(temp_q);
        vector<double> temp_unigram_probs = unigram_probs;
        unigram_probs_vector.push_back(temp_unigram_probs);
        clock_t t = clock()+rand();
        mt19937 eng_int_temp (t);  
        eng_int_vector.push_back(eng_int_temp);
        uniform_int_distribution<> unif_int_temp(0, rbmParam.n_vocab-1);
        unif_int_vector.push_back(unif_int_temp);
        mt19937 eng_real_temp (t);  // mt19937 is a standard mersenne_twister_engine
        eng_real_vector.push_back(eng_real_temp);
        uniform_real_distribution<> unif_real_temp(0.0, 1.0);
        unif_real_vector.push_back(unif_real_temp);
        //cerr<<"the clock was "<<t<<endl;

    }
    //initalizing the threads again
    for (int i=0;i<rbmParam.n_threads;i++)
    {
        clock_t t = clock()+rand();
        eng_int_vector[i].seed(t);
        eng_real_vector[i].seed(t);
 
    }

    uniform_int_distribution<> unif_int(0, rbmParam.n_vocab-1);
    uniform_real_distribution<> unif_real(0.0, 1.0);
    //initializing the RBM
    RBM rbm(rbmParam);
    //testing sample h given v
    //int training_data_size = unshuffled_training_data.size();
    cerr<<"The training data size is "<<training_data_size<<endl;
    //int carry_over = (training_data_size%rbmParam.minibatch_size == 0)? 0:1;
    //int num_batches = training_data_size/rbmParam.minibatch_size + carry_over;
    int num_batches = (training_data_size-1)/rbmParam.minibatch_size + 1;
    int final_batch_size = training_data_size - rbmParam.minibatch_size*(num_batches-1);
    //cerr<<"Final batch size was "<<final_batch_size<<endl;
    /*
    if (training_data_size % rbmParam.minibatch_size) {
      final_batch_size  = training_data_size
    }
    */
    //get the generated samples
    vector<int>random_nos_int;
    vector <double> random_nos_real; 

    int random_nos_int_counter = 0;
    int random_nos_real_counter = 0;

    cerr<<"num training batches is "<<num_batches<<endl;
    //performing training
    clock_t t;
    double current_momentum = rbmParam.initial_momentum;
    double momentum_delta = (rbmParam.final_momentum - rbmParam.initial_momentum)/(rbmParam.num_epochs-1);

    //creating the minibatches that we will use
    Matrix<bool,Dynamic,Dynamic> positive_h_minibatch,
      negative_h_minibatch,
      positive_h_validation_set,
      negative_h_validation_set;

    Matrix<double,Dynamic,Dynamic> positive_h_probs_minibatch,
      noise_h_probs_minibatch,
      positive_h_probs_validation_set,
      noise_h_probs_validation_set;

    //persistent_h_probs_validation_set
    Matrix<int,Dynamic,Dynamic> negative_v_minibatch,
      negative_v_validation_set;



    setprecision(15);
    cerr<<"Validation set size: "<<validation_set.rows()<<endl;
    double epsilon = 10E-7;
    double noise_samples_ratio = rbmParam.num_noise_samples; //rbmParam.minibatch_size;
    cerr<<"The noise samples ratio is "<<noise_samples_ratio<<endl;
    double previous_corpus_log_prob  = -999999999999;
    //double current_learning_rate = rbmParam.learning_rate;
    for (int epoch = 0 ;epoch<rbmParam.num_epochs;epoch++)
    { 

        if (validation_set_size !=0)
        {
            cerr<<"Computing log likelihood on the validation set "<<endl;
            vector_map normalization_cache;
            //then compute log likelihood on the validation set
            double corpus_log_prob = 0.;
            Eigen::setNbThreads(1);
            int n_vocab= rbmParam.n_vocab;
            int ngram_size = rbmParam.ngram_size;
            #pragma omp parallel for firstprivate(validation_set_size,ngram_size,n_vocab) reduction(+:corpus_log_prob)
            for (int valid_id = 0;valid_id<validation_set_size;valid_id++)
            {
                int thread_id = omp_get_thread_num();
                vector<int> context(ngram_size-1) ;
                vector<int> ngram(ngram_size);
                for (int position =0;position<ngram_size-1;position++)
                {
                    context[position]=validation_set(valid_id,position);
                    ngram[position] = validation_set(valid_id,position);
                }

                int output_label = validation_set(valid_id,ngram_size-1);
                vector<double> output_values(2,0.); //this will store the normalization log prob and the ngram prob
                //getting the normalization constant and the output_values
                //then we have to compute the normalization constant
                if (normalization_cache.find(context) == normalization_cache.end())
                {

                    rbm.computeNgramProb(context,output_values,thread_id,n_vocab,output_label,ngram_size);
                    corpus_log_prob += output_values[0] - output_values[1];
                    #pragma omp critical
                    {
                        normalization_cache[context] = output_values[1];
                    }
                }
                else //we found the normalization constant 
                {
                    ngram[ngram_size-1] = output_label;
                    double log_normalization_constant = normalization_cache[context];
                    double output_label_log_unnorm_prob = -rbm.computeFreeEnergy(ngram,thread_id);
                    output_label_log_unnorm_prob = (output_label_log_unnorm_prob > 300) ? 300:output_label_log_unnorm_prob;
                    corpus_log_prob += output_label_log_unnorm_prob - log_normalization_constant;
                }
            }
            #pragma omp barrier
            cerr<<"corpus log prob is "<<corpus_log_prob<<endl;
            if (corpus_log_prob < previous_corpus_log_prob) {
              rbmParam.learning_rate = rbmParam.learning_rate/2.;
            }
            previous_corpus_log_prob = corpus_log_prob;
        }
        //we have to compute the NCE likelihood in every epoch
        double NCE_likelihood = 0.;

        //cerr<<"current momentum is "<<current_momentum<<endl;
        cerr<<"epoch is "<<epoch+1<<endl;
        for(int batch=0;batch<num_batches;batch++)
        {
            int current_minibatch_size = rbmParam.minibatch_size;
            
            int minibatch_start_index = rbmParam.minibatch_size * batch;
            // Set the bath size for batches 0 to num_batches-2 
            if (batch == 0) {
              positive_h_probs_minibatch.resize(rbmParam.minibatch_size,rbmParam.n_hidden);
              noise_h_probs_minibatch.resize(rbmParam.num_noise_samples,rbmParam.n_hidden);

              negative_v_minibatch.resize(rbmParam.minibatch_size,rbmParam.ngram_size);
              negative_v_validation_set.resize(validation_set.rows(),rbmParam.ngram_size);

              positive_h_probs_minibatch.setZero();
              noise_h_probs_minibatch.setZero();

              negative_v_minibatch.setZero();
              negative_v_validation_set.setZero();
            } 

            if (batch == num_batches-1) {
              current_minibatch_size = final_batch_size;
              positive_h_probs_minibatch.resize(final_batch_size,rbmParam.n_hidden);
              noise_h_probs_minibatch.resize(rbmParam.num_noise_samples,rbmParam.n_hidden);

              negative_v_minibatch.resize(final_batch_size,rbmParam.ngram_size);
              negative_v_validation_set.resize(validation_set.rows(),rbmParam.ngram_size);

              positive_h_probs_minibatch.setZero();
              noise_h_probs_minibatch.setZero();

              negative_v_minibatch.setZero();
              negative_v_validation_set.setZero();
            }


            //cerr<<"batch number is "<<batch<<endl;
            if (batch%1000 == 0)
            {
                cerr<<"processed "<<batch<<" batches"<<endl;
            }
            //doing fprop for the positive samples and for the noise samples
            

            //basically, we will have to compute rbmParam.num_noise_samples triples for the entire minibatch. The noise distribution is 
            //a product of the unigram distributions
            Matrix<int,Dynamic,Dynamic> noise_triples;
            noise_triples.setZero(rbmParam.num_noise_samples,rbmParam.ngram_size);
            clock_t t;
            t = clock();
            //parallelizing the creation with multithreading
            //Eigen::initParallel();
            Eigen::setNbThreads(1);

            int num_noise_samples = rbmParam.num_noise_samples;
            int ngram_size = rbmParam.ngram_size;
            //omp_set_num_threads(2);
            #pragma omp parallel for firstprivate(ngram_size, \
                                    num_noise_samples)
            for (int position_id= 0;position_id < ngram_size;position_id++)
            {
                int thread_id = omp_get_thread_num();
                for (int sample_id = 0;sample_id < num_noise_samples ;sample_id++)
                {
                    int mixture_component = unif_int_vector[thread_id](eng_int_vector[thread_id]);
                    //noise_triples(sample_id,position_id)= mixture_component;
                    double p = unif_real_vector[thread_id](eng_real_vector[thread_id]);
                    noise_triples(sample_id,position_id)= mixture_component;
                    if (q_vector[thread_id][mixture_component] >= p)
                    {
                        noise_triples(sample_id,position_id)= mixture_component;
                    }
                    else
                    {
                        noise_triples(sample_id,position_id) = J_vector[thread_id][mixture_component];
                    }

                }
            }
            #pragma omp barrier
            //cerr<<"the noise triples are "<<noise_triples<<endl;
            //getchar();
            //FPROP WITH TRUE DATA
            //omp_set_num_threads(2);

            rbm.fProp_omp(shuffled_training_data,current_minibatch_size,minibatch_start_index,positive_h_probs_minibatch,rbmParam);

            //FPROP WITH NEGATIVE DATA
            rbm.fProp_noise_omp(noise_triples,rbmParam.num_noise_samples,0,noise_h_probs_minibatch,rbmParam);   
           
            //COMPUTING THE NOISE PROBABILITIES AND TRUE PROBABILITIES FOR THE NOISE SAMPLES
            vector<double> noise_weights(num_noise_samples,0.);
            vector<double> positive_weights(current_minibatch_size,0.);

            double current_c_exp = rbm.c_exp; //setting  the current normalization constant to be the current normalization constant
            double current_c = rbm.c; //setting  the current normalization constant to be the current normalization constant
            vector<double> c_gradient_vector(rbmParam.n_threads,0.0);

            NCE_likelihood += rbm.computeModelWeights(num_noise_samples,
                0,
                current_c,
                ngram_size,
                noise_triples,
                rbm.noise_activations,
                noise_h_probs_minibatch,
                noise_weights,
                c_gradient_vector, 
                unigram_probs_vector,
                -1,
                num_noise_samples,
                epsilon,
                noise_samples_ratio);

            //omp_set_num_threads(1);
            //omp_set_num_threads(1);
            //COMPUTING THE NOISE PROBABILITIES FOR THE POSITIVE SAMPLES
            //omp_set_num_threads(1);
            NCE_likelihood += rbm.computeModelWeights(
                current_minibatch_size,
                minibatch_start_index,
                current_c,
                ngram_size,
                shuffled_training_data,
                rbm.activations,
                positive_h_probs_minibatch,
                positive_weights,
                c_gradient_vector, 
                unigram_probs_vector,
                1,
                num_noise_samples,
                epsilon,
                noise_samples_ratio);

            //cerr<<"performed positive and negative fprop"<<endl; 
            //UPDATE THE PARAMETERS

            //omp_set_num_threads(1);
            //omp_set_num_threads(2);
            rbm.updateParameters_omp(shuffled_training_data,
                current_minibatch_size,
                minibatch_start_index,
                num_noise_samples,
                noise_triples,
                positive_h_probs_minibatch,
                noise_h_probs_minibatch,
                positive_weights,
                noise_weights,
                c_gradient_vector,
                rbmParam,
                current_momentum);
            //cerr<<"updated the gradient"<<endl;
            //cerr<<"current c is "<<ryybm.c<<endl;
            //cerr<<"average f is "<<NCE_likelihood<<endl;

        }
        current_momentum += momentum_delta;
        //write the embedings after each epoch
        rbm.writeEmbeddings(rbmParam,epoch,word_list);
        //write the parameters after every epoch as well
        rbm.writeParams(epoch);
        cerr<<"Epoch "<<epoch<<" NCE likelihood is "<<NCE_likelihood<<endl;
        //cerr<<"current c is "<<rbm.c<<endl;
    }
    return 0;

}
