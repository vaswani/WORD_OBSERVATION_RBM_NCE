#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
//#include <random>
#include "param.h"
#include <ctime>
#include <stdio.h>
//#include <chrono>
#include <math.h>
#include "util.h"
#include <iomanip>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/unordered_map.hpp> 
#include <omp.h>
#include <algorithm>
//#include "log_add.h"
//#define EIGEN_DONT_PARALLELIZE
//#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <assert.h>

typedef boost::unordered_map<int,bool> int_map;

using namespace std;
using namespace Eigen;
using namespace boost::random;

typedef double Real;
typedef Matrix<bool,1 , Dynamic> vectorHidden;
const double max_value = std::exp(250);
const double min_value = 0;

class RBM
{
    
    //private:
    public:
        //vectorHidden hidden_layer;
        Matrix<bool,1 , Dynamic> hidden_layer;
        Matrix<double,Dynamic,Dynamic> W;
        Matrix<double,Dynamic,Dynamic> W_gradient;
        Matrix<double,Dynamic,Dynamic> W_running_gradient;
        Matrix<double,Dynamic,Dynamic> U;
        Matrix<double,Dynamic,Dynamic> U_running_gradient;
        Matrix<double,Dynamic,Dynamic> velocity_U;
        Matrix<double,1,Dynamic> v_bias;
        Matrix<double,1,Dynamic> v_bias_gradient;
        Matrix<double,1,Dynamic> v_bias_running_gradient;
        Matrix<double,1,Dynamic> h_bias;
        Matrix<double,1,Dynamic> h_bias_gradient;
        Matrix<double,1,Dynamic> h_bias_running_gradient;
        //in order to make this parallel, I will create a vector of parameters 
        //equal to the number of threads
        vector<Matrix<double,Dynamic,Dynamic> > W_vector;
        vector<Matrix<double,Dynamic,Dynamic> > U_vector;
        vector<Matrix<double,1,Dynamic> > v_bias_vector;
        vector<Matrix<double,1,Dynamic> > h_bias_vector;
        Matrix<double,Dynamic,Dynamic> feature_inputs_positive,
          feature_inputs_noise,
          activations,
          noise_activations,
          activations_validation,
          feature_inputs_validation,
          validation_activation;
        vector<double> c_gradient ;
        double c;
        double c_exp;
        double c_running_gradient ;

    public:
        //initializing directly
        RBM(Matrix<double,Dynamic,Dynamic> input_W,
            Matrix<double,Dynamic,Dynamic> input_U,
            Matrix<double,1,Dynamic> input_v_bias,
            Matrix<double,1,Dynamic> input_h_bias,
            int n_threads,
            param & rbmParam)
        {
            W = input_W;
            U = input_U;
            v_bias = input_v_bias;
            h_bias = input_h_bias;

            //unsigned int num_threads = omp_get_num_threads();
            //cout<<"num threads is "<<num_threads<<endl;
            for (int i = 0;i<n_threads;i++)
            {
                //cout<<"i is "<<i<<endl;
                Matrix<double,Dynamic,Dynamic> temp_W = W;
                W_vector.push_back(temp_W);
                //cout<<"W vec element "<<i<<" is "<<W_vector[i]<<endl;
                Matrix<double,Dynamic,Dynamic> temp_U = U;
                U_vector.push_back(temp_U);
                //cout<<"U vec element "<<i<<" is "<<U_vector[i]<<endl;
                Matrix<double,1,Dynamic> temp_v_bias = v_bias;
                v_bias_vector.push_back(temp_v_bias);
                //cout<<"v bias element "<<i<<" is "<<v_bias_vector[i]<<endl;
                Matrix<double,1,Dynamic> temp_h_bias = h_bias;
                h_bias_vector.push_back(temp_h_bias);
                //cout<<"h bias element "<<i<<" is "<<h_bias_vector[i]<<endl;
            }
            feature_inputs_positive.resize(rbmParam.embedding_dimension*rbmParam.ngram_size,rbmParam.minibatch_size);
            feature_inputs_noise.resize(rbmParam.embedding_dimension*rbmParam.ngram_size,rbmParam.num_noise_samples);
            feature_inputs_validation.resize(rbmParam.embedding_dimension*rbmParam.ngram_size,rbmParam.validation_minibatch_size);
            activations.resize(rbmParam.n_hidden,rbmParam.minibatch_size);
            noise_activations.resize(rbmParam.n_hidden,rbmParam.num_noise_samples);
            W_gradient.setZero(W.rows(),W.cols());
            v_bias_gradient.setZero(v_bias.cols());
            W_running_gradient.setZero(W.rows(),W.cols());
            v_bias_running_gradient.setZero(v_bias.size());
            U_running_gradient.setZero(U.rows(),U.cols());
            h_bias_gradient.setZero(h_bias.size());
            h_bias_running_gradient.setZero(h_bias.size());


        }
        //RBM(param rbmParam);
        RBM(param rbmParam)
        {
            //cout<<"in the RBM constructor"<<endl;
            //initializing the weights et
            W.resize(rbmParam.n_vocab,rbmParam.embedding_dimension);
            U.resize(rbmParam.n_hidden,rbmParam.embedding_dimension*rbmParam.ngram_size);
            velocity_U.setZero(rbmParam.n_hidden,rbmParam.embedding_dimension*rbmParam.ngram_size);
            v_bias.resize(1,rbmParam.n_vocab);
            h_bias.resize(1,rbmParam.n_hidden);
			      unsigned seed = std::time(0);
            clock_t t;
            cout<<t+rand()<<endl;
            cout<<t+rand()<<endl;
            mt19937 eng_W (t+rand());  // mt19937 is a standard mersenne_twister_engine
            mt19937 eng_U (t+rand());  // mt19937 is a standard mersenne_twister_engine
            mt19937 eng_bias_h (t+rand());  // mt19937 is a standard mersenne_twister_engine
            mt19937 eng_bias_v (t+rand());  // mt19937 is a standard mersenne_twister_engine
            cerr<<"W rows is "<<W.rows()<<" and W cols is "<<W.cols()<<endl;

            cerr<<"U rows is "<<U.rows()<<" and U cols is "<<U.cols()<<endl;

            cerr<<"v_bias rows is "<<v_bias.rows()<<" and v_bias cols is "<<v_bias.cols()<<endl;

            cerr<<"h_bias rows is "<<h_bias.rows()<<" and h_bias cols is "<<h_bias.cols()<<endl;

            void * distribution ;
            if (rbmParam.init_normal == 0)
            {
                uniform_real_distribution<> unif_real(-0.01, 0.01); 
                //initializing W
                for (int i =0;i<W.rows();i++)
                {
                    //cout<<"i is "<<i<<endl;
                    for (int j =0;j<W.cols();j++)
                    {
                        //cout<<"j is "<<j<<endl;
                        W(i,j) = unif_real(eng_W);    
                    }
                }
                //initializing U

                for (int i =0;i<U.rows();i++)
                {
                    for (int j =0;j<U.cols();j++)
                    {
                        U(i,j) = unif_real(eng_U);    
                    }
                }
                //initializing v_bias
                for (int i =0;i<v_bias.cols();i++)
                {
                    //cout<<"i is "<<i<<endl;
                    v_bias(i) = unif_real(eng_bias_v);
                }

                //initializing h_bias
                for (int i =0;i<h_bias.cols();i++)
                {
                    h_bias(i) = unif_real(eng_bias_h);
                }

            }
            else //initialize with gaussian distribution with mean 0 and stdev 0.01
            {
                normal_distribution<double> unif_normal(0.,0.01);
                //initializing W
                for (int i =0;i<W.rows();i++)
                {
                    //cout<<"i is "<<i<<endl;
                    for (int j =0;j<W.cols();j++)
                    {
                        //cout<<"j is "<<j<<endl;
                        W(i,j) = unif_normal(eng_W);    
                    }
                }
                //initializing U

                for (int i =0;i<U.rows();i++)
                {
                    for (int j =0;j<U.cols();j++)
                    {
                        U(i,j) = unif_normal(eng_U);    
                    }
                }
                //initializing v_bias
                for (int i =0;i<v_bias.cols();i++)
                {
                    //cout<<"i is "<<i<<endl;
                    v_bias(i) = unif_normal(eng_bias_v);
                }

                //initializing h_bias
                for (int i =0;i<h_bias.cols();i++)
                {
                    h_bias(i) = unif_normal(eng_bias_h);
                }
              
            }
            //unsigned int num_threads = omp_get_num_threads();
            //cout<<"num threads is "<<num_threads<<endl;
            for (int i = 0;i<rbmParam.n_threads;i++)
            {
                //cout<<"i is "<<i<<endl;
                Matrix<double,Dynamic,Dynamic> temp_W = W;
                W_vector.push_back(temp_W);
                //cout<<"W vec element "<<i<<" is "<<W_vector[i]<<endl;
                Matrix<double,Dynamic,Dynamic> temp_U = U;
                U_vector.push_back(temp_U);
                //cout<<"U vec element "<<i<<" is "<<U_vector[i]<<endl;
                Matrix<double,1,Dynamic> temp_v_bias = v_bias;
                v_bias_vector.push_back(temp_v_bias);
                //cout<<"v bias element "<<i<<" is "<<v_bias_vector[i]<<endl;
                Matrix<double,1,Dynamic> temp_h_bias = h_bias;
                h_bias_vector.push_back(temp_h_bias);
                //cout<<"h bias element "<<i<<" is "<<h_bias_vector[i]<<endl;
                c_gradient.push_back(0.0);
            }

            c = -rbmParam.normalization_init;
            c_exp = std::exp(c);
            feature_inputs_positive.resize(rbmParam.embedding_dimension*rbmParam.ngram_size,rbmParam.minibatch_size);
            feature_inputs_noise.resize(rbmParam.embedding_dimension*rbmParam.ngram_size,rbmParam.num_noise_samples);
            activations.resize(rbmParam.n_hidden,rbmParam.minibatch_size);
            noise_activations.resize(rbmParam.n_hidden,rbmParam.minibatch_size);
            W_gradient.setZero(W.rows(),W.cols());
            v_bias_gradient.setZero(v_bias.cols());
            W_running_gradient.setZero(W.rows(),W.cols());
            v_bias_running_gradient.setZero(v_bias.size());
            U_running_gradient.setZero(U.rows(),U.cols());
            h_bias_gradient.setZero(h_bias.size());
            h_bias_running_gradient.setZero(h_bias.size());


        }

        template <typename DerivedInt, typename DerivedDouble>
        void fProp_omp(const MatrixBase<DerivedInt>& v_layer_minibatch,
            int minibatch_size,
            int minibatch_start_index,
            MatrixBase<DerivedDouble> &const_h_layer_probs_minibatch,
            param & rbmParam)
        {
            UNCONST(DerivedDouble,const_h_layer_probs_minibatch,h_layer_probs_minibatch);
            //Eigen::initParallel();
            Eigen::setNbThreads(1);
            int embedding_dimension = rbmParam.embedding_dimension;
            int ngram_size = rbmParam.ngram_size;
            int n_hidden = rbmParam.n_hidden;

            //first setting up the input matrix
            #pragma omp parallel firstprivate(minibatch_start_index,minibatch_size,embedding_dimension,n_hidden,ngram_size)
            {

            #pragma omp for            

            for (int i = 0;i< minibatch_size;i++)
            {
                int thread_id = omp_get_thread_num();
                for (int index = 0;index < ngram_size;index++)
                {
                    feature_inputs_positive.block(index*embedding_dimension,i,embedding_dimension,1) = W_vector[thread_id].row(v_layer_minibatch(minibatch_start_index + i,index)).transpose();
                }
              }
            }
  
            //now getting the logits and allowing eigen to do its own parallization
            Eigen::setNbThreads(rbmParam.n_threads); 
            activations = U*feature_inputs_positive; //dimension is n_hidden x minibatch_size 
            //cout<<"true activations are "<<activations<<endl;
            //getchar();

            Eigen::setNbThreads(1);

            //now getting the sigmoids and the h states
            #pragma omp parallel for firstprivate(minibatch_size,n_hidden)        
            for (int i = 0;i< minibatch_size;i++)
            {
                int thread_id = omp_get_thread_num();

                h_layer_probs_minibatch.row(i) = (1/(1+(-((h_bias_vector[thread_id].transpose() + activations.col(i)).array())).exp()));
            }
            #pragma omp barrier
        }

        template <typename DerivedInt, typename DerivedDouble>
        void fProp_noise_omp(const MatrixBase<DerivedInt>& v_layer_minibatch,
            int minibatch_size,
            int minibatch_start_index,
            const MatrixBase<DerivedDouble> &const_h_layer_probs_minibatch,
            param & rbmParam)
        {
            UNCONST(DerivedDouble,const_h_layer_probs_minibatch,h_layer_probs_minibatch)
            //UNCONST(DerivedInt,const_v_layer_minibatch,v_layer_minibatch)
            //Eigen::initParallel();
            Eigen::setNbThreads(1);
            int embedding_dimension = rbmParam.embedding_dimension;
            int ngram_size = rbmParam.ngram_size;
            int n_hidden = rbmParam.n_hidden;

            //first setting up the input matrix
            #pragma omp parallel firstprivate(minibatch_start_index,minibatch_size,embedding_dimension,n_hidden,ngram_size)
            {
            #pragma omp for            
            for (int i = 0;i< minibatch_size;i++)
            {
                int thread_id = omp_get_thread_num();
                for (int index = 0;index < ngram_size;index++)
                {
                    feature_inputs_noise.block(index*embedding_dimension,i,embedding_dimension,1) = 
                        W_vector[thread_id].row(v_layer_minibatch(minibatch_start_index + i,index)).transpose();
                }
            }
            }

            //cout<<"the negative feature inputs was "<<feature_inputs_noise<<endl;
            //getchar();
            //cout<<"populated the feature table"<<endl;
            //now getting the logits and allowing eigen to do its own parallization
            Eigen::setNbThreads(rbmParam.n_threads); 
            noise_activations = U*feature_inputs_noise; //dimension is n_hidden x minibatch_size 
            //cout<<"noise activations are "<<noise_activations<<endl;
            //getchar();
            //cout<<"computed activations"<<endl;
            //Eigen::initParallel();
            Eigen::setNbThreads(1);

            //now getting the sigmoids and the h states
            #pragma omp parallel for firstprivate(minibatch_size,n_hidden)        

            for (int i = 0;i< minibatch_size;i++)
            {
                int thread_id = omp_get_thread_num();

                h_layer_probs_minibatch.row(i) = (1/(1+(-((h_bias_vector[thread_id].transpose() + noise_activations.col(i)).array())).exp()));

            }
            #pragma omp barrier
        }

        template<typename DerivedInt,typename DerivedDouble>
        void updateParameters_omp(const MatrixBase<DerivedInt> &const_positive_v_minibatch,
                int minibatch_size,
                int minibatch_start_index,
                int num_noise_samples,
                const MatrixBase<DerivedInt> &const_noise_triples,
                const MatrixBase<DerivedDouble> &const_positive_h_probs_minibatch,
                const MatrixBase<DerivedDouble> &const_noise_h_probs_minibatch,
                vector<double> & positive_weights,
                vector<double> & noise_weights,
                vector<double> &c_gradient_vector,
                param & rbmParam,
                double current_momentum)
        {
            //UNCONST THE MATRICES
            UNCONST(DerivedInt,const_positive_v_minibatch,positive_v_minibatch);
            UNCONST(DerivedInt,const_noise_triples,noise_triples);
            UNCONST(DerivedDouble,const_positive_h_probs_minibatch,positive_h_probs_minibatch);
            UNCONST(DerivedDouble,const_noise_h_probs_minibatch,noise_h_probs_minibatch);

            Matrix<double,Dynamic,Dynamic> U_gradient,U_positive_backprop,U_noise_backprop;
            U_gradient.setZero(U.rows(),U.cols());
            bool use_momentum = rbmParam.use_momentum;
            int embedding_dimension = rbmParam.embedding_dimension;
            int ngram_size = rbmParam.ngram_size;
            int n_hidden = rbmParam.n_hidden;
            double L2_reg = rbmParam.L2_reg;
            int n_threads = rbmParam.n_threads;
            //double adjusted_learning_rate = rbmParam.learning_rate/(minibatch_size+num_noise_samples);
            double adjusted_learning_rate = rbmParam.learning_rate/minibatch_size;
            /*
            double true_adjusted_learning_rate = rbmParam.learning_rate/minibatch_size;
            double noise_ratio = float(num_noise_samples)/minibatch_size;
            double noise_adjusted_learning_rate = rbmParam.learning_rate*noise_ratio/num_noise_samples;
            */
            int u_update_threads = (rbmParam.n_threads > 3)? 3: rbmParam.n_threads;

            //omp_set_num_threads(rbmParam.n_threads);
            //first doing backprop
            U_positive_backprop.noalias() = U.transpose()*positive_h_probs_minibatch.transpose();
            U_noise_backprop.noalias() = U.transpose()*noise_h_probs_minibatch.transpose();

            if (use_momentum == 1)
            {
                Eigen::setNbThreads(rbmParam.n_threads);
                if (L2_reg != 0)
                {
                    U_gradient += adjusted_learning_rate*((feature_inputs_positive * positive_h_probs_minibatch).transpose()-
                                                 (feature_inputs_noise * noise_h_probs_minibatch).transpose()-2*L2_reg*U);
                }
                else
                {
                    U_gradient += adjusted_learning_rate*((feature_inputs_positive * positive_h_probs_minibatch).transpose()-
                                                 (feature_inputs_noise * noise_h_probs_minibatch).transpose());
                }
                
            }
            else
            {

                Eigen::setNbThreads(rbmParam.n_threads);
                if (L2_reg != 0)
                {

                    U += adjusted_learning_rate * ((feature_inputs_positive * positive_h_probs_minibatch).transpose()- 
                                                  (feature_inputs_noise * noise_h_probs_minibatch).transpose()-
                                                   2*L2_reg*U);
                }
                else
                {
                    U += adjusted_learning_rate * ((feature_inputs_positive * positive_h_probs_minibatch).transpose()- 
                                                  (feature_inputs_noise * noise_h_probs_minibatch).transpose());

                }
            }

            //first we get the u gradients and then we update the velocity

            if (use_momentum == 1)
            {
                //for (int index = 0;index < ngram_size;index++)
                //{
                //Eigen::initParallel();
                Eigen::setNbThreads(1);

                #pragma omp parallel firstprivate(embedding_dimension,current_momentum,ngram_size)
                {
                #pragma omp for
                for (int col = 0;col<embedding_dimension*ngram_size;col++)
                {

                    velocity_U.col(col) = current_momentum* velocity_U.col(col) +
                                                                      U_gradient.col(col);
                    U.col(col) +=  velocity_U.col(col);
                }

                }
                #pragma omp barrier
                //}        
                        
            }

            #pragma omp parallel for firstprivate(n_threads)
            for (int col = 0;col<U.cols();col++)
            {
                for (int i = 0;i<n_threads;i++)
                {
                  U_vector[i].col(col) = U.col(col);
                }
            }

            //cout<<"computed the u gradient"<<endl; 
            //Eigen::initParallel();
            Eigen::setNbThreads(rbmParam.n_threads);

            //omp_set_num_threads(rbmParam.n_threads);
            

            int_map update_map;
            //updating the w parameters correctly
            //positive gradient
            for (int i = 0;i< minibatch_size;i++)
            {
                for (int index = 0;index < ngram_size;index++)
                {
                    int positive_index = positive_v_minibatch(i+minibatch_start_index,index);
                    //updating the indexes that will be useful since we will be updating only items that have been updated
                    update_map[positive_index] = 1;
                    W.row(positive_index) += adjusted_learning_rate*U_positive_backprop.block(embedding_dimension*index,i,embedding_dimension,1).transpose() ;
                    /*                                                    
                    W.row(positive_index) += adjusted_learning_rate *(positive_h_probs_minibatch.row(i)*
                                                        U.block(0,index*embedding_dimension,n_hidden,embedding_dimension));
                    W.row(negative_index) -= adjusted_learning_rate * (noise_h_probs_minibatch.row(i)*
                                                        U.block(0,index*embedding_dimension,n_hidden,embedding_dimension));
                    */
                    //updating the vbias as well
                    v_bias(positive_index) += adjusted_learning_rate*positive_weights[i];
                    //v_bias(negative_index) -= adjusted_learning_rate;

                }
            }
            //negative gradient
            for (int i = 0;i< num_noise_samples;i++)
            {
                for (int index = 0;index < ngram_size;index++)
                {
                    int noise_index = noise_triples(i,index);
                    update_map[noise_index] = 1;
                    W.row(noise_index) -= adjusted_learning_rate*U_noise_backprop.block(embedding_dimension*index,i,embedding_dimension,1).transpose(); 
                    v_bias(noise_index) -= adjusted_learning_rate*noise_weights[i];

                }
            }
            //cout<<"computed the W gradient"<<endl;
            int num_items =0;
            vector<int> update_items;
            int_map::iterator it;
            for (it = update_map.begin();it != update_map.end();it++)
            {
                update_items.push_back((*it).first);
                num_items++;
            }
            //now updating the thread parameters
            //int n_threads = rbmParam.n_threads;
            Eigen::setNbThreads(1);
            #pragma omp parallel for firstprivate(num_items,n_threads)
            for (int i=0;i<num_items;i++)
            {
                int update_item = update_items[i];
                for (int thread_id=0;thread_id<n_threads;thread_id++)
                {
                    W_vector[thread_id].row(update_item) = W.row(update_item);
                    v_bias_vector[thread_id](update_item) = v_bias(update_item);
                }
            }
            #pragma omp barrier
            
            //updating the h bias in a parallel fashion
            #pragma omp parallel for firstprivate(n_hidden,minibatch_size)
            for (int dim = 0;dim< n_hidden;dim++)
            { 
                for (int i = 0;i< minibatch_size;i++)
                {  
                    h_bias(dim) += adjusted_learning_rate*positive_h_probs_minibatch(i,dim);
                }
            }

            #pragma omp parallel for firstprivate(n_hidden,num_noise_samples)
            for (int dim = 0;dim< n_hidden;dim++)
            { 
                for (int i = 0;i< num_noise_samples;i++)
                {  
                    h_bias(dim) -= adjusted_learning_rate*noise_h_probs_minibatch(i,dim);
                }
            }

            //copying the updated h parameters into the vectors
            for (int thread_id = 0;thread_id<rbmParam.n_threads;thread_id++)
            {
                  h_bias_vector[thread_id] = h_bias;
            }
            //computing the c_h gradient
            double total_c_gradient=0.;
            
            #pragma omp parallel for firstprivate(n_threads) reduction(+:total_c_gradient)
            for (int i = 0;i<n_threads;i++)
            {
                total_c_gradient += c_gradient_vector[i];
            }
            c += adjusted_learning_rate * total_c_gradient;
            c_exp = std::exp(c);
        }
        
        template <typename DerivedInt, typename DerivedDouble>
        void updateParametersAdagrad_omp(const MatrixBase<DerivedInt>& const_positive_v_minibatch,
            int minibatch_size,
            int minibatch_start_index,
            int num_noise_samples,
            const MatrixBase<DerivedInt> &const_noise_triples,
            const MatrixBase<DerivedDouble> &const_positive_h_probs_minibatch,
            const MatrixBase<DerivedDouble> &const_noise_h_probs_minibatch,
            vector<double> & positive_weights,
            vector<double> & noise_weights,
            vector<double> &c_gradient_vector,
            param & rbmParam,
            double current_momentum)
        {
            //UNCONST THE MATRICES
            UNCONST(DerivedInt,const_positive_v_minibatch,positive_v_minibatch);
            UNCONST(DerivedInt,const_noise_triples,noise_triples);
            UNCONST(DerivedDouble,const_positive_h_probs_minibatch,positive_h_probs_minibatch);
            UNCONST(DerivedDouble,const_noise_h_probs_minibatch,noise_h_probs_minibatch);
            //cout<<"minibatch_size is "<<minibatch_size<<endl;
            //cout<<"num noise samples "<<num_noise_samples<<endl;
            Matrix<double,Dynamic,Dynamic> U_gradient,U_positive_backprop,U_noise_backprop;
            U_gradient.setZero(U.rows(),U.cols());
            //h_bias_gradient.setZero();
            int embedding_dimension = rbmParam.embedding_dimension;
            int ngram_size = rbmParam.ngram_size;
            int n_hidden = rbmParam.n_hidden;
            double L2_reg = rbmParam.L2_reg;
            int n_threads = rbmParam.n_threads;
            double learning_rate = rbmParam.learning_rate;
            //int u_update_threads = (rbmParam.n_threads > 3)? 3: rbmParam.n_threads;

            //omp_set_num_threads(rbmParam.n_threads);
            //first doing backprop
            Eigen::setNbThreads(rbmParam.n_threads);
            U_positive_backprop.noalias() = U.transpose()*positive_h_probs_minibatch.transpose();
            U_noise_backprop.noalias() = U.transpose()*noise_h_probs_minibatch.transpose();

            //Eigen::setNbThreads(rbmParam.n_threads);
            if (L2_reg != 0)
            {
                U_gradient = (feature_inputs_positive * positive_h_probs_minibatch).transpose()-
                                             (feature_inputs_noise * noise_h_probs_minibatch).transpose()-2*L2_reg*U;
            }
            else
            {
                U_gradient = (feature_inputs_positive * positive_h_probs_minibatch).transpose()-
                                             (feature_inputs_noise * noise_h_probs_minibatch).transpose();

            }
                
            //first we get the u gradients and then we update the velocity
            //cout<<"Optimized U is "<<U<<endl;
            //getchar();
            Eigen::setNbThreads(1);

            #pragma omp parallel firstprivate(embedding_dimension,ngram_size,learning_rate,n_threads)
            {
            #pragma omp for
            for (int col = 0;col<embedding_dimension*ngram_size;col++)
            {

                U_running_gradient.col(col) += U_gradient.col(col).array().square().matrix();
                U.col(col) +=  learning_rate * (U_gradient.col(col).array()/U_running_gradient.col(col).array().sqrt()).matrix();
                for (int i = 0;i<n_threads;i++)
                {
                    U_vector[i].col(col) = U.col(col);
                }
                //U_gradient.col(col).setZero();

            }

            }
            #pragma omp barrier
                        
            //cout<<"computed the u gradient"<<endl; 
            //cout<<"U norm is "<<U.norm()<<endl;
            //getchar();
            //Eigen::initParallel();
            Eigen::setNbThreads(rbmParam.n_threads);

            //omp_set_num_threads(rbmParam.n_threads);
            

            int_map update_map;
            //updating the w parameters correctly
            //positive gradient
            for (int i = 0;i< minibatch_size;i++)
            {
                for (int index = 0;index < ngram_size;index++)
                {
                    int positive_index = positive_v_minibatch(i+minibatch_start_index,index);
                    //updating the indexes that will be useful since we will be updating only items that have been updated
                    update_map[positive_index] = 1;
                    W_gradient.row(positive_index) += U_positive_backprop.block(embedding_dimension*index,i,embedding_dimension,1).transpose() ;
                    //updating the vbias as well
                    v_bias_gradient(positive_index) += positive_weights[i];

                }
            }
            //negative gradient
            for (int i = 0;i< num_noise_samples;i++)
            {
                for (int index = 0;index < ngram_size;index++)
                {
                    int noise_index = noise_triples(i,index);
                    update_map[noise_index] = 1;
                    W_gradient.row(noise_index) -= U_noise_backprop.block(embedding_dimension*index,i,embedding_dimension,1).transpose(); 
                    v_bias_gradient(noise_index) -= noise_weights[i];

                }
            }
            int num_items =0;
            vector<int> update_items;
            int_map::iterator it;
            for (it = update_map.begin();it != update_map.end();it++)
            {
                update_items.push_back((*it).first);
                num_items++;
            }
            //cout<<"number of update items is "<<update_items.size()<<endl;
            //cout<<"num items is "<<num_items<<endl;

            //now updating the thread parameters
            //int n_threads = rbmParam.n_threads;
            //updating the W and v bias
            Eigen::setNbThreads(1);
            #pragma omp parallel for firstprivate(num_items,n_threads,learning_rate)
            for (int i=0;i<num_items;i++)
            {
                int update_item = update_items[i];
                //cout<<"update item is "<<update_item<<endl;
                //cout<<"W gradient is "<<W_gradient.row(update_item)<<endl;
                //cout<<"W gradient norm "<<W_gradient.row(update_item).norm()<<endl;
                W_running_gradient.row(update_item) += W_gradient.row(update_item).array().square().matrix();
                //cout<<"W running gradient is "<<W_running_gradient.row(update_item)<<endl;
                //cout<<"W running gradient is "<<W_running_gradient.row(update_item).norm()<<endl;
                W.row(update_item) += learning_rate*(W_gradient.row(update_item).array()/W_running_gradient.row(update_item).array().sqrt()).matrix();
                v_bias_running_gradient(update_item) += v_bias_gradient(update_item)*v_bias_gradient(update_item);
                v_bias(update_item) += learning_rate * v_bias_gradient(update_item)/sqrt(v_bias_running_gradient(update_item));
                W_gradient.row(update_item).setZero();
                v_bias_gradient(update_item) = 0.;

                for (int thread_id=0;thread_id<n_threads;thread_id++)
                {
                    W_vector[thread_id].row(update_item) = W.row(update_item);
                    v_bias_vector[thread_id](update_item) = v_bias(update_item);
                }
                //cout<<"optimized W is "<<W.row(update_item)<<endl;
                //cout<<"optimized W norm is "<<W.row(update_item).norm()<<endl;
                //cout<<"optimized v_bias is "<<v_bias(update_item)<<endl;
            }
            //getchar();
            #pragma omp barrier
            Eigen::setNbThreads(1);
 
            //updating the h bias in a parallel fashion
            #pragma omp parallel for firstprivate(n_hidden,minibatch_size,num_noise_samples)
            for (int dim = 0;dim< n_hidden;dim++)
            { 
                for (int i = 0;i< minibatch_size;i++)
                {  
                    h_bias_gradient(dim) += positive_h_probs_minibatch(i,dim);
                }
                for (int i = 0;i< num_noise_samples;i++)
                {  
                    h_bias_gradient(dim) -= noise_h_probs_minibatch(i,dim);
                }
            }
            /*
            #pragma omp parallel for firstprivate(embedding_dimension,num_noise_samples)
            for (int dim = 0;dim< embedding_dimension;dim++)
            { 
                for (int i = 0;i< num_noise_samples;i++)
                {  
                    h_bias_gradient(dim) -= noise_h_probs_minibatch(i,dim);
                }
            }
            */

            //updating the h bias in a parallel fashion
            #pragma omp parallel for firstprivate(n_hidden,learning_rate,n_threads)
            for (int dim = 0;dim< n_hidden;dim++)
            { 
                h_bias_running_gradient(dim) += h_bias_gradient(dim)*h_bias_gradient(dim);
                h_bias(dim) += learning_rate * h_bias_gradient(dim)/sqrt(h_bias_running_gradient(dim));
                for (int thread_id = 0;thread_id<n_threads;thread_id++)
                {
                    h_bias_vector[thread_id](dim) = h_bias(dim);
                }
                h_bias_gradient(dim) =0.;
                //cout<<"optimized h bias dim is "<<h_bias(dim)<<endl;

            }
            //getchar();

            /*
            Eigen::setNbThreads(n_threads);
            //now updating the h bias
            h_bias_running_gradient += h_bias_gradient.array().square();
            h_bias += learning_rate * (h_bias_gradient.array()/h_bias_running_gradient.array().sqrt()).matrix();

            //copying the updated h parameters into the vectors
            for (int thread_id = 0;thread_id<rbmParam.n_threads;thread_id++)
            {
                  h_bias_vector[thread_id] = h_bias;
            }
            */

            //computing the c_h gradient
            double total_c_gradient=0.;
            
            #pragma omp parallel for firstprivate(n_threads) reduction(+:total_c_gradient)
            for (int i = 0;i<n_threads;i++)
            {
                total_c_gradient += c_gradient_vector[i];
            }
            c_running_gradient += total_c_gradient*total_c_gradient;
            c += learning_rate * total_c_gradient/sqrt(c_running_gradient);
            c_exp = std::exp(c);
        }

        inline double computeFreeEnergy(vector<int> &ngram,param &rbmParam)
        {
            //cout<<"in compute free energy"<<endl;
            Real free_energy = 0.;
            Real sum_biases = 0.;
            for (int i = 0;i<rbmParam.ngram_size;i++)
            {
                sum_biases += v_bias(ngram[i]);
            }
            //cout<<"sum biases is "<<sum_biases<<endl;
            Matrix<double,1,Dynamic> temp_w_sum;
            for (int h_j = 0;h_j<rbmParam.n_hidden;h_j++)
            {
                double dot_product = 0.0;
                //weight_type h_bias_sum = 0. 
                for (int index = 0;index<rbmParam.ngram_size;index++)
                {   
                    dot_product += U.row(h_j).block(0,index*rbmParam.embedding_dimension,1,rbmParam.embedding_dimension).dot(W.row(ngram[index])) ;
                }
                free_energy -= std::log(1 + std::exp(dot_product + h_bias(h_j)));

            }

            free_energy-= sum_biases ;
            //cout<<"free energy is "<<free_energy<<endl;
            return free_energy;

        }

        inline double computeFreeEnergy(Matrix<int,Dynamic,Dynamic> &data,int row,param &rbmParam)
        {
            //cout<<"in compute free energy"<<endl;
            Real free_energy = 0.;
            Real sum_biases = 0.;
            for (int i = 0;i<rbmParam.ngram_size;i++)
            {
                sum_biases += v_bias(data(row,i));
            }
            //cout<<"sum biases is "<<sum_biases<<endl;
            Matrix<double,1,Dynamic> temp_w_sum;
            for (int h_j = 0;h_j<rbmParam.n_hidden;h_j++)
            {
                double dot_product = 0.0;
                //weight_type h_bias_sum = 0. 
                for (int index = 0;index<rbmParam.ngram_size;index++)
                {   
                    dot_product += U.row(h_j).block(0,index*rbmParam.embedding_dimension,1,rbmParam.embedding_dimension).dot(W.row(data(row,index))) ;
                }
                free_energy -= std::log(1 + std::exp(dot_product + h_bias(h_j)));

            }

            free_energy-= sum_biases ;
            //cout<<"free energy is "<<free_energy<<endl;
            return free_energy;

        }
        double computeFreeEnergy(vector<int> &ngram,int thread_id)
        { 
            Eigen::setNbThreads(1);
            //cout<<"in compute free energy"<<endl;
            int ngram_size = ngram.size();
            int n_hidden = h_bias_vector[thread_id].size();
            int embedding_dimension = W_vector[thread_id].cols();
            //cout<<"in compute free energy"<<endl;
            Real free_energy = 0.;
            Real sum_biases = 0.;
            for (int i = 0;i<ngram_size;i++)
            {
                sum_biases += v_bias_vector[thread_id](ngram[i]);
            }
            //cout<<"sum biases is "<<sum_biases<<endl;
            Matrix<double,1,Dynamic> temp_w_sum;
            for (int h_j = 0;h_j<n_hidden;h_j++)
            {
                double dot_product = 0.0;
                //double h_bias_sum = 0. 
                for (int index = 0;index<ngram_size;index++)
                {   
                    dot_product += U_vector[thread_id].row(h_j).block(0,index*embedding_dimension,1,embedding_dimension).dot(W_vector[thread_id].row(ngram[index])) ;
                }
                //cout<<"dot product is "<<dot_product<<endl;
                double exp_term = dot_product + h_bias_vector[thread_id](h_j);
                if (exp_term > -20) //if the term in the exp is not very small
                { 
                    free_energy -= (exp_term > 15) ? exp_term : std::log(1 + std::exp(exp_term));
                }
            }

            free_energy -= sum_biases ;
            //cout<<"free energy is "<<free_energy<<endl;
            return free_energy;

        }
        
        /*
        //get the cross entropy on a small validation set.
        double getCrossEntropy(Matrix<int,Dynamic,Dynamic> input,Matrix<int,Dynamic,Dynamic> prediction,param & rbmParam)
        {
             
        }
        */
        double inline computeReconstructionError(Matrix<int,Dynamic,Dynamic> input,
            Matrix<int,Dynamic,Dynamic> prediction)
        {
            //for (int i = 0;i<
            //for i in 
        }

        // write the embeddings to the file
        void writeEmbeddings(param &rbmParam,int epoch,vector<string> word_list)
        {
            setprecision(16);
            stringstream ss;//create a stringstream
            ss << epoch;//add number to the stream
            //return ss.str();//return a string with the contents of the stream
          
            string output_file = rbmParam.embeddings_prefix+"."+ss.str();
            cout << "Writing embeddings to file : " << output_file << endl;

            ofstream EMBEDDINGOUT;
            EMBEDDINGOUT.precision(15);
            EMBEDDINGOUT.open(output_file.c_str());
            if (! EMBEDDINGOUT)
            {
              cerr << "Error : can't write to file " << output_file << endl;
              exit(-1);
            }
            for (int row = 0;row < W.rows();row++)
            {
                EMBEDDINGOUT<<word_list[row]<<"\t";
                for (int col = 0;col < W.cols();col++)
                {
                    EMBEDDINGOUT<<W(row,col)<<"\t";
                }
                EMBEDDINGOUT<<endl;
            }

            EMBEDDINGOUT.close();
        }


        void writeParams(int epoch)
        {
            //write the U parameters
            setprecision(16);
            stringstream ss;//create a stringstream
            ss << epoch;//add number to the stream
            //return ss.str();//return a string with the contents of the stream
          
            string U_output_file = "U."+ss.str();
            cout << "Writing U params to output_file: " << U_output_file << endl;

            ofstream UOUT;
            UOUT.precision(15);
            UOUT.open(U_output_file.c_str());
            if (! UOUT)
            {
              cerr << "Error : can't write to file " << U_output_file << endl;
              exit(-1);
            }
            for (int row = 0;row < U.rows();row++)
            {
                for (int col = 0;col < U.cols()-1;col++)
                {
                    UOUT<<U(row,col)<<"\t";
                }
                //dont want an extra tab at the end
                UOUT<<U(row,U.cols()-1);
                UOUT<<endl;
            }

            UOUT.close();
          
            //write the V bias parameters
            setprecision(16);
            //return ss.str();//return a string with the contents of the stream
          
            string v_bias_output_file = "v_bias."+ss.str();
            cout << "Writing v bias params to output_file: " << v_bias_output_file << endl;

            ofstream VOUT;
            VOUT.precision(15);
            VOUT.open(v_bias_output_file.c_str());
            if (! VOUT)
            {
              cerr << "Error : can't write to file " << v_bias_output_file << endl;
              exit(-1);
            }
            for (int col= 0;col< v_bias.cols()-1;col++)
            {
                VOUT<<v_bias(col)<<"\t";
            }
            VOUT<<v_bias(v_bias.cols()-1);
            VOUT.close();

            //write the V bias parameters
            //setprecision(16);
            //return ss.str();//return a string with the contents of the stream
          
            string h_bias_output_file = "h_bias."+ss.str();
            cout << "Writing vbias params to output_file: " << h_bias_output_file << endl;

            ofstream HOUT;
            HOUT.precision(15);
            HOUT.open(h_bias_output_file.c_str());
            if (! HOUT)
            {
              cerr << "Error : can't write to file " << h_bias_output_file << endl;
              exit(-1);
            }
            for (int col= 0;col< h_bias.cols()-1;col++)
            {
                HOUT<<h_bias(col)<<"\t";
            }
            HOUT<<h_bias(h_bias.cols()-1);
            HOUT.close();

        }
        template<typename DerivedInt, typename DerivedDouble>
        double computeModelWeights(
            int current_minibatch_size,
            int minibatch_start_index,
            double current_c,
            int ngram_size,
            const MatrixBase<DerivedInt> &data,
            const MatrixBase<DerivedDouble> &current_activations,
            const MatrixBase<DerivedDouble> &const_h_probs_minibatch,
            vector<double> &weights,
            vector<double> &c_gradient_vector,
            vector<vector<double> > &unigram_probs_vector,
            int noise_multiple,
            int num_noise_samples,
            double epsilon,
            double noise_samples_ratio) {
            
            //UNCONST(DerivedInt, const_data, data);
            //UNCONST(DerivedDouble, const_current_activations, current_activations);
            UNCONST(DerivedDouble, const_h_probs_minibatch, h_probs_minibatch);

            double log_noise_samples_ratio = std::log(noise_samples_ratio);
            /*
            if (noise_multiple == -1)
            {
                cout<<"noise time "<<endl;
            }
            else
            {
                cout<<"true time"<<endl;
            }
            */
            int correct_predictions =0;
            Eigen::setNbThreads(1);
            int n_hidden = h_bias.size();
            int embedding_dimension = W.cols();
            double minibatch_NCE_likelihood = 0.;
            //first do fprop and compute the feature activations
            
            #pragma omp parallel for firstprivate(epsilon,ngram_size,minibatch_start_index,current_minibatch_size,num_noise_samples,current_c,noise_multiple,embedding_dimension,n_hidden) reduction(+:minibatch_NCE_likelihood)
            for (int train_id = 0;train_id < current_minibatch_size;train_id++)
            {

                int thread_id = omp_get_thread_num();
                double noise_prob  = 1.;
                double log_noise_prob = 0.;
                double sum_biases = 0.;
                //vector<int> ngram (ngram_size);
                //noise_probs[sample_id] = 1.;
                for (int position_id= 0;position_id < ngram_size;position_id++)
                {
                    noise_prob *= unigram_probs_vector[thread_id][data(minibatch_start_index+train_id,position_id)];
                    log_noise_prob += std::log(unigram_probs_vector[thread_id][data(minibatch_start_index+train_id,
                          position_id)]);
                    sum_biases += v_bias_vector[thread_id](data(minibatch_start_index+train_id,position_id));
                    //ngram[position_id] = data(minibatch_start_index+train_id,position_id);
                }

                //cout<<"in compute free energy"<<endl;
                double free_energy = 0.;
                //cout<<"sum biases is "<<sum_biases<<endl;
                for (int h_j = 0;h_j<n_hidden;h_j++)
                {
                    double dot_product = current_activations(h_j,train_id);

                    /*
                    double exp_term = dot_product + h_bias_vector[thread_id](h_j);
                    if (dot_product >-20)
                    { 
                        free_energy -= (dot_product > 15) ? 15 : std::log(1 + std::exp(exp_term));
                    }
                    */
                    double exp_term = dot_product + h_bias_vector[thread_id](h_j);
                    if (exp_term > -20) //if the term in the exp is not very small
                    { 
                        free_energy -= (exp_term > 15) ? exp_term : std::log(1 + std::exp(exp_term));
                    }

                }
                free_energy -= sum_biases ;

                double log_unnorm_prob = -free_energy;
                /*
                if (log_unnorm_prob > 300)
                {
                    cout<<"log unnorm prob was greater than 300"<<endl;
                }
                */
                double model_prob =  1.0;
                double log_model_prob = log_unnorm_prob+current_c;
                if (log_model_prob > 250)
                {
                    //cout<<"warning! log model prob was greater than 300"<<endl
                    model_prob = max_value;

                }
                else if (log_model_prob < -20)
                {
                    //cout<<"warning! log model prob was greater than 300"<<endl
                    model_prob = 0.;
                }
                else
                {
                    //cout<<"log model prob is "<<log_model_prob<<endl;
                    model_prob = std::exp(log_model_prob);
                }
                //cout<<"mode prob is "<<model_prob<<endl;
                //getchar();
                /*
                log_unnorm_prob = (log_unnorm_prob > 300)? 300: log_unnorm_prob;
                log_unnorm_prob = (log_unnorm_prob < -300)? -300: log_unnorm_prob;
                double model_prob = std::exp(log_unnorm_prob+current_c);
                */
                double z = logadd(log_model_prob,log_noise_prob+log_noise_samples_ratio);

                if (noise_multiple == 1)
                {
                    //double ratio = noise_samples_ratio*noise_prob/(model_prob + noise_samples_ratio*noise_prob);
                    double ratio = std::exp(log_noise_samples_ratio + log_noise_prob - z);

                    weights[train_id] = (ratio < epsilon) ? epsilon : ratio;
                }
                else
                {
                    //double ratio = model_prob/(model_prob + noise_samples_ratio*noise_prob);
                    double ratio = std::exp(log_model_prob - z);
                    weights[train_id] =  (ratio < epsilon) ? epsilon : ratio;
                }
                //cout<<"weights is "<<weights[train_id]<<endl;
                //cout<<"positive weight was "<<positive_weights[train_id]<<endl;
                minibatch_NCE_likelihood += log(1-weights[train_id]+epsilon);
                if (1-weights[train_id] >  0.5)
                {
                    correct_predictions++;
                }
                c_gradient_vector[thread_id] +=  noise_multiple * weights[train_id];
                h_probs_minibatch.row(train_id) = h_probs_minibatch.row(train_id)* weights[train_id];
                
            
            }
            #pragma omp barrier
            //getchar();
            //cout<<"the number of correct predictions out of "<<current_minibatch_size<<" are"<<correct_predictions<<endl;
            return(minibatch_NCE_likelihood);
        }

        template<typename DerivedInt>
        void fPropValidation_omp(MatrixBase<DerivedInt>& v_layer_minibatch,
            int minibatch_size,
            int minibatch_start_index,
           param & rbmParam)
        {

            //Eigen::initParallel();
            Eigen::setNbThreads(1);
            int embedding_dimension = rbmParam.embedding_dimension;
            int ngram_size = rbmParam.ngram_size;
            int n_hidden = rbmParam.n_hidden;

            //first setting up the input matrix
            #pragma omp parallel firstprivate(minibatch_start_index,minibatch_size,embedding_dimension,n_hidden,ngram_size)
            {

            #pragma omp for            

            for (int i = 0;i< minibatch_size;i++)
            {
                int thread_id = omp_get_thread_num();
                for (int index = 0;index < ngram_size;index++)
                {
                    feature_inputs_validation.block(index*embedding_dimension,i,embedding_dimension,1) = W_vector[thread_id].row(v_layer_minibatch(minibatch_start_index + i,index)).transpose();
                }
              }
            }
            #pragma omp barrier 
  
            //now getting the logits and allowing eigen to do its own parallization
            Eigen::setNbThreads(rbmParam.n_threads); 
            activations_validation = U*feature_inputs_validation; //dimension is n_hidden x minibatch_size 
            //cout<<"true activations are "<<activations<<endl;
            //getchar();

        }

        //this computes the normalization constant and the conditional probability of the last word given the previous two words
        void computeNgramProb(vector<int> ngram,vector<double> &output_values,
            int thread_id,
            int n_vocab,
            int output_label,
            int ngram_size)
        {
            //cout<<"the output label is "<<output_label<<endl;
            Eigen::setNbThreads(1);
            int n_hidden = h_bias_vector[thread_id].size();
            int embedding_dimension = W_vector[thread_id].cols();
            vector<double> partial_dot_products(n_hidden,0.);
            double partial_sum_biases = 0.;
            double log_normalization_constant = -99999999999;
            //computing partial bias sum as well
            for (int index = 0;index<ngram_size-1;index++)
            {
                partial_sum_biases += v_bias_vector[thread_id](ngram[index]); 
            }


            //first computing the partial dot products that do not need to be recomputed for every vocabulary label
            for (int h_j = 0;h_j<n_hidden;h_j++)
            {
                for (int index = 0;index<ngram_size-1;index++)
                {   
                    partial_dot_products[h_j] += U_vector[thread_id].row(h_j).block(0,index*embedding_dimension,1,embedding_dimension).dot(W_vector[thread_id].row(ngram[index])) ;
                }

            }
            //cout<<"n voacab is "<<n_vocab<<endl;
            for (int vocab_id = 0;vocab_id<n_vocab;vocab_id++)
            {
                double free_energy = 0.;
                double sum_biases = 0.;
                //cout<<"the ngram is "<<vocab_id<<"and the embedding is "<<W_vector[thread_id].row(vocab_id)<<endl;
                //getchar();

                sum_biases = v_bias_vector[thread_id](vocab_id);
                //cout<<"sum biases is "<<sum_biases<<endl;
                for (int h_j = 0;h_j<n_hidden;h_j++)
                {
                    double dot_product = 0.0;
                    dot_product = partial_dot_products[h_j] + U_vector[thread_id].row(h_j).block(0,(ngram_size-1)*embedding_dimension,1,embedding_dimension).dot(W_vector[thread_id].row(vocab_id)) ;

                    //cout<<"dot product for dimension "<<h_j<<" is "<<dot_product<<endl;
                    //getchar();
                    //dot_product = (dot_product < -20)? 0.: dot_product;
                    double exp_term = dot_product + h_bias_vector[thread_id](h_j);
                    if (exp_term > -20) //if the term in the exp is not very small
                    { 
                        free_energy -= (exp_term > 15) ? exp_term : std::log(1 + std::exp(exp_term));
                    }
                    //free_energy -= std::log(1 + std::exp(dot_product + h_bias_vector[thread_id](h_j)));

                }

                free_energy -= sum_biases ;
                //cout<<"free energy is "<<free_energy<<endl;
                //getchar();
                
                if (vocab_id == output_label)
                {
                    //output_values[0] = (-free_energy > 300) ? 300:-free_energy; 
                    output_values[0] = (-free_energy+partial_sum_biases > 300) ? 300:-free_energy+partial_sum_biases;
                    //output_values[0] = -free_energy;
                    //cout<<"label unnorm prob is "<<output_values[0]<<endl;
                }
                log_normalization_constant = logadd(log_normalization_constant,-free_energy);
                log_normalization_constant = (log_normalization_constant > 300) ? 300:log_normalization_constant;
                
            }
            //cout<<"log normalization constant is "<<log_normalization_constant<<endl;
            //output_values[1] = log_normalization_constant;
            output_values[1] = (log_normalization_constant+partial_sum_biases > 300) ? 300:log_normalization_constant+partial_sum_biases;
        }


};

