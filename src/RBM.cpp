#include "RBM.h"
//#include "param.h"

using namespace RBM

/*
//constructor
RBM(param myParam)
{
    cout<<"in the RBM constructor"<<endl;
    //initializing the weights et
    W.resize(rbmParam.n_vocab,rbmParam.embedding_dimension);
    U.resize(rbmParam.n_hidden,rbmParam.embedding_dimension*rbmParam.ngram_size);
    v_bias.resize(1,rbmParam.n_vocab);
    h_bias.resize(1,rbmParam.n_hidden);
    minibatch_h_positive.resize(rbmParam.minibatch_size,rbmParam.n_hidden);
    minibatch_h_negative.resize(rbmParam.minibatch_size,rbmParam.n_hidden);
    minibatch_v_negative.resize(rbmParam.minibatch_size,rbmParam.ngram_size);

    //we have to initialize W and U 
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();

    mt19937 eng_W (seed);  // mt19937 is a standard mersenne_twister_engine
    mt19937 eng_U (seed);  // mt19937 is a standard mersenne_twister_engine
    mt19937 eng_bias_h (seed);  // mt19937 is a standard mersenne_twister_engine
    mt19937 eng_bias_v (seed);  // mt19937 is a standard mersenne_twister_engine

    uniform_real_distribution<double> unif_real(-0.1, 0.1); 
    cout<<"W rows is "<<W.rows()<<" and W cols is "<<W.cols()<<endl;

    cout<<"U rows is "<<U.rows()<<" and U cols is "<<U.cols()<<endl;

    cout<<"v_bias rows is "<<v_bias.rows()<<" and v_bias cols is "<<v_bias.cols()<<endl;

    cout<<"h_bias rows is "<<h_bias.rows()<<" and h_bias cols is "<<h_bias.cols()<<endl;
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
    cout<<"initialized u"<<endl;
    //initializing v_bias
    for (int i =0;i<v_bias.cols();i++)
    {
        //cout<<"i is "<<i<<endl;
        v_bias(i) = unif_real(eng_bias_v);
    }

    cout<<"initialized v"<<endl;
    //initializing h_bias
    for (int i =0;i<h_bias.cols();i++)
    {
        h_bias(i) = unif_real(eng_bias_h);
    }
    cout<<"finished initializing"<<endl;
    cout<<"W is now "<<W<<endl;
    cout<<"U is now "<<U<<endl;
    cout<<"h_bias is now "<<h_bias<<endl;
    cout<<"V bias is now "<<v_bias<<endl;
}
*/
/*
RBM::~RBM(void)
{

}
*/
//samples the hidden state given the visible state. 
//here, we sample the positive h given v
void sample_h_given_v_positive(Matrix<int,Dynamic,Dynamic>& visible_states)
{
        
}
