#include <vector>
#include <set>
#include <assert.h>

void setupAliasMethod(vector<double>& unigram_probs,vector<int> &J ,vector<double> &q,int k)
{
    cout<<"k is "<<k<<endl;
    set<int> S;
    set<int>::iterator s_it;
    set<int> L;
    set<int>::iterator l_it;
    //list<int> S;
    //list<int> L;

    for (int i=0; i<k; i++) 
    {
        q[i] = k*unigram_probs[i];
        //cout<<"q[i] for "<<i<<" is "<<q[i]<<endl;
        if (q[i] < 1.)
        {
            //S.push_back(i);    
            S.insert(i);
        }
        else
        {
            L.insert(i);
        } 
    }   
    /*
    it=myset.find(20);
    myset.erase (it);
    myset.erase (myset.find(40));
    */	
	double s_sum = 0.;
	for (s_it = S.begin();s_it != S.end();s_it++)
	{
		s_sum += q[*s_it];
	}
	double l_sum = 0.;
	for (l_it = L.begin();l_it != L.end();l_it++)
	{
		l_sum += q[*l_it];
	}
	//cout<<"s sum is "<<s_sum<<endl;
	//cout<<"l sum is "<<l_sum<<endl;
    while (S.size() !=0)
    {
        //cout<<"size of S is "<<S.size()<<endl;
        s_it = S.begin();
        l_it = L.begin();
        int s = *s_it;
        int l = *l_it;
        //cout<<"s is "<<s<<" and l is "<<l<<endl;
        //l.erase(l_it);
        J[s] = l;
		assert (J[s] > -1);
        q[l] = q[l] - (1.0000000 -q[s]);
        //cout<<"q[l] is "<<q[l]<<endl;
		if (q[l] > 0.9999999 && q[l] < 1.00000001)
		{
            q[l] = 1.;				
		}
        S.erase(s_it);
        assert (q[l] > 0);
		/*
		if (q[l] > 0.99999)
		{
            q[l]=  1.0;			
		}
		*/
        if (q[l] < 1.0)
        {
            S.insert(l);
            L.erase(l_it);
        }
        //cout<<"size of L is "<<L.size()<<endl;
    }
}


void inline initVNegative(Matrix<int,Dynamic,Dynamic>& visible_states,Matrix<int,Dynamic,Dynamic>& minibatch_v_negative,int minibatch_size,int minibatch_start_index)
{
    //cout<<"minibatch start index is "<<minibatch_start_index<<endl;
    for (int i = 0 ;i<minibatch_size;i++)
    {
        minibatch_v_negative.row(i) = visible_states.row(i+minibatch_start_index);
    }
}


int inline computeReconstructionError(Matrix<int,Dynamic,Dynamic>& original_data,Matrix<int,Dynamic,Dynamic>&reconstruction,param &rbmParam)
{
    int reconstruction_error = 0;
    for (int i = 0;i<original_data.rows();i++)
    {
        for (int j = 0;j<rbmParam.ngram_size;j++)
        {
            if (original_data(i,j) != reconstruction(i,j))
            {
                reconstruction_error++;            
            }
        }
    }
    return (reconstruction_error);

}
