#pragma once
#include <vector>
#include <string>
#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include <ctime>
#include <fstream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

//Eigen matrices are to be passed to functions as const DerivedTypes. They have to be unconsted before use
#define UNCONST(DerivedType,consted,unconsted) Eigen::MatrixBase<DerivedType> &unconsted = const_cast<Eigen::MatrixBase<DerivedType>&>(consted);
 

//code from Sittichai Jiampojamarn's 
inline vector<string> spaceSplit(string line)
{
    vector<string> items;
    //cout<< "line is "<<line<<endl;
    string buff;
    stringstream ss(line);
    //int value;
    //cout<<"ss is "<<ss<<endl;
    string value;
    while (ss >> value)
    {
        //ss >> value;
        //cout<<"value is "<<value<<endl;
        items.push_back(value);
    }

    return items;
}


double logadd(double x, double y)
{
    if (x > y)
        return x + log1p(std::exp(y-x));
    else
        return y + log1p(std::exp(x-y));
}


void readTrainFile(param & rbmParam,vector<vector<int> > & training_data)
{
  cout << "Reading training data from file : " << rbmParam.train_file<< endl;


  ifstream TRAININ;
  TRAININ.open(rbmParam.train_file.c_str());
  if (! TRAININ)
  {
    cerr << "Error : can't read training data from file " << rbmParam.train_file<< endl;
    exit(-1);
  }

  while (! TRAININ.eof())
  {
    string line;
    vector<string> ngram;

    getline(TRAININ, line);
    if (line == "")
    {
      continue;
    }

    ngram = spaceSplit(line);

    if (ngram.size() != rbmParam.ngram_size)
    {
      cerr << "Error the training input was not equal to the ngram size " << endl << line << endl;
      exit(-1);
    }

    vector<int> int_ngram;
    for (int i=0;i<rbmParam.ngram_size;i++)
    {
        int_ngram.push_back((int)atoi(ngram[i].c_str()));
    }
    training_data.push_back(int_ngram);
  }
  TRAININ.close();

}

void readTrainFileMatrix(string data_file,param &rbmParam, Matrix<int,Dynamic,Dynamic> &shuffled_training_data)
{
  cout << "Reading training data from file : " << rbmParam.train_file<< endl;


  ifstream TRAININ;
  TRAININ.open(data_file.c_str());
  if (! TRAININ)
  {
    cerr << "Error : can't read training data from file " << data_file<< endl;
    exit(-1);
  }
  int row = 0;
  while (! TRAININ.eof())
  {
    string line;
    vector<string> ngram;

    getline(TRAININ, line);
    if (line == "")
    {
      continue;
    }

    ngram = spaceSplit(line);

    if (ngram.size() != rbmParam.ngram_size)
    {
      cerr << "Error the training input was not equal to the ngram size " << endl << line << endl;
      exit(-1);
    }

    vector<int> int_ngram;
    for (int i=0;i<rbmParam.ngram_size;i++)
    {
        shuffled_training_data(row,i) = (int)atoi(ngram[i].c_str());
        //cout<<"the integer is "<<(int)atoi(ngram[i].c_str())<<endl;
        //getchar();
        //int_ngram.push_back((int)atoi(ngram[i].c_str()));
    }
    //training_data.push_back(int_ngram);
    row ++;
  }
  TRAININ.close();

}


void readDataFile(string file_name,param & rbmParam,vector<vector<int> > & data_vector)
{
  cout << "Reading data from file : " << file_name<< endl;


  ifstream DATAIN;
  DATAIN.open(file_name.c_str());
  if (! DATAIN)
  {
    cerr << "Error : can't read data from file " << file_name<< endl;
    exit(-1);
  }

  while (! DATAIN.eof())
  {
    string line;
    vector<string> ngram;

    getline(DATAIN, line);
    if (line == "")
    {
      continue;
    }

    ngram = spaceSplit(line);

    if (ngram.size() != rbmParam.ngram_size)
    {
      cerr << "Error the training input was not equal to the ngram size " << endl << line << endl;
      exit(-1);
    }

    vector<int> int_ngram;
    for (int i=0;i<rbmParam.ngram_size;i++)
    {
        int_ngram.push_back((int)atoi(ngram[i].c_str()));
    }
    data_vector.push_back(int_ngram);
  }
  DATAIN.close();

}


//(double)atof(lineList[2].c_str()); I will use this
void readUnigramProbs(param rbmParam,vector<double> & unigram_probs)
{
  //cout << "Reading unigram probs: " << rbmParam.unigram_probs_file<< endl;


  ifstream UNIGRAMIN;
  UNIGRAMIN.open(rbmParam.unigram_probs_file.c_str());
  if (! UNIGRAMIN)
  {
    cerr << "Error : can't read unigram probs from" << rbmParam.unigram_probs_file<< endl;
    exit(-1);
  }

  while (! UNIGRAMIN.eof())
  {
    //cout<<"reding unigrams probs"<<endl;
    string line;
    vector<string> ngram;

    getline(UNIGRAMIN, line);
    if (line == "")
    {
      continue;
    }

    ngram = spaceSplit(line);

    if (ngram.size() != 2)
    {
      cerr << "Error: There should have been two entries on the line " << endl << line << endl;
      exit(-1);
    }
//(double)atof(lineList[2].c_str());
    unigram_probs[(int)atoi(ngram[0].c_str())] = (double)atof(ngram[1].c_str());
    //cout<<"added to unigram probs"<<endl;

  }
  UNIGRAMIN.close();

}

void readParameter(string param_file, Matrix<double,Dynamic,Dynamic> & param)
{
  cout << "Reading training data from file : " << param_file<< endl;


  ifstream TRAININ;
  TRAININ.open(param_file.c_str());
  if (! TRAININ)
  {
    cerr << "Error : can't read training data from file " << param_file<< endl;
    exit(-1);
  }
  int line_counter = 0;
  while (! TRAININ.eof())
  {
    string line;
    vector<string> row;

    getline(TRAININ, line);
    if (line == "")
    {
      continue;
    }

    row = spaceSplit(line);
    /*
    if (ngram.size() != rbmParam.ngram_size)
    {
      cerr << "Error the training input was not equal to the ngram size " << endl << line << endl;
      exit(-1);
    }
    */
    //vector<double> double_row;

    for (int i=0;i<row.size();i++)
    {
        param(line_counter,i) = (double)atof(row[i].c_str());
    }
    //training_data.push_back(int_ngram);
    line_counter++;
  }
  TRAININ.close();

}

void readParameterBias(string param_file, Matrix<double,1,Dynamic> & param)
{
  cout << "Reading training data from file : " << param_file<< endl;


  ifstream TRAININ;
  TRAININ.open(param_file.c_str());
  if (! TRAININ)
  {
    cerr << "Error : can't read training data from file " << param_file<< endl;
    exit(-1);
  }
  int line_counter = 0;
  while (! TRAININ.eof())
  {
    string line;
    vector<string> row;

    getline(TRAININ, line);
    if (line == "")
    {
      continue;
    }

    row = spaceSplit(line);
    /*
    if (ngram.size() != rbmParam.ngram_size)
    {
      cerr << "Error the training input was not equal to the ngram size " << endl << line << endl;
      exit(-1);
    }
    */
    //vector<double> double_row;

    for (int i=0;i<row.size();i++)
    {
        param(i) = (double)atof(row[i].c_str());
    }
    //training_data.push_back(int_ngram);
    line_counter++;
  }
  TRAININ.close();

}


void readRandomNosRealTriple(string file ,vector<vector<double> > & random_nos_real,param &rbmParam)
{
  cout << "Reading random nos from : " << file<< endl;


  ifstream TRAININ;
  TRAININ.open(file.c_str());
  if (! TRAININ)
  {
    cerr << "Error : can't read training data from file " << file<< endl;
    exit(-1);
  }

  while (! TRAININ.eof())
  {
    string line;
    vector<string> ngram;

    getline(TRAININ, line);
    if (line == "")
    {
      continue;
    }

    ngram = spaceSplit(line);

    if (ngram.size() != rbmParam.ngram_size)
    {
      cerr << "Error the training input was not equal to the ngram size " << endl << line << endl;
      exit(-1);
    }

    vector<double> int_ngram;
    for (int i=0;i<rbmParam.ngram_size;i++)
    {
        int_ngram.push_back((double)atof(ngram[i].c_str()));
    }
    random_nos_real.push_back(int_ngram);
  }
  TRAININ.close();

}
void readRandomNosInt(string file ,vector<int> & random_nos_int)
{
  cout << "Reading random nos from : " << file<< endl;


  ifstream TRAININ;
  TRAININ.open(file.c_str());
  if (! TRAININ)
  {
    cerr << "Error : can't read training data from file " << file<< endl;
    exit(-1);
  }

  while (! TRAININ.eof())
  {
    string line;
    vector<string> ngram;

    getline(TRAININ, line);
    if (line == "")
    {
      continue;
    }

    ngram = spaceSplit(line);


    random_nos_int.push_back((int)atoi(ngram[0].c_str()));
  }
  TRAININ.close();

}

void readWordsFile(string file ,vector<string> & word_list)
{
  cout << "Reading word list from : " << file<< endl;


  ifstream TRAININ;
  TRAININ.open(file.c_str());
  if (! TRAININ)
  {
    cerr << "Error : can't read training data from file " << file<< endl;
    exit(-1);
  }

  while (! TRAININ.eof())
  {
    string line;
    vector<string> ngram;

    getline(TRAININ, line);
    if (line == "")
    {
      continue;
    }

    ngram = spaceSplit(line);

    word_list.push_back(ngram[0]);
  }
  TRAININ.close();

}

void readRandomNosReal(string file ,vector<double> & random_nos_real)
{
  cout << "Reading random nos from : " << file<< endl;


  ifstream TRAININ;
  TRAININ.open(file.c_str());
  if (! TRAININ)
  {
    cerr << "Error : can't read training data from file " << file<< endl;
    exit(-1);
  }

  while (! TRAININ.eof())
  {
    string line;
    vector<string> ngram;

    getline(TRAININ, line);
    if (line == "")
    {
      continue;
    }

    ngram = spaceSplit(line);


    random_nos_real.push_back((double)atof(ngram[0].c_str()));
  }
  TRAININ.close();

}

void writeTempData(vector<vector <int> > &shuffled_training_data,param & rbmParam)
{
    //write the U parameters
    //setprecision(16);
    //stringstream ss;//create a stringstream
    //ss << epoch;//add number to the stream
    //return ss.str();//return a string with the contents of the stream
  
    string output_file = "temp.dat";
    cout << "Writing temp training data to output_file: " <<output_file << endl;

    ofstream OUT;
    //OUT.precision(15);
    OUT.open(output_file.c_str());
    if (! OUT)
    {
      cerr << "Error : can't write to file " << output_file << endl;
      exit(-1);
    }
    for (int row = 0;row < shuffled_training_data.size()-1;row++)
    {
        for (int col = 0;col < rbmParam.ngram_size -1;col++)
        {
            OUT<<shuffled_training_data[row][col]<<"\t";
        }
        //dont want an extra tab at the end
        OUT<<shuffled_training_data[row][rbmParam.ngram_size-1];
        OUT<<endl;
    }
    //now writing the last row
    for (int col = 0;col < rbmParam.ngram_size -1;col++)
    {
            OUT<<shuffled_training_data[shuffled_training_data.size()-1][col]<<"\t";
    }
    OUT<<shuffled_training_data[shuffled_training_data.size()-1][rbmParam.ngram_size-1];
    OUT.close();
  

}

void writeTempDataMatrix(Matrix<int,Dynamic,Dynamic> &shuffled_training_data,param & rbmParam)
{
    //write the U parameters
    //setprecision(16);
    //stringstream ss;//create a stringstream
    //ss << epoch;//add number to the stream
    //return ss.str();//return a string with the contents of the stream
  
    string output_file = "temp.dat1";
    cout << "Writing temp training data to output_file: " <<output_file << endl;

    ofstream OUT;
    //OUT.precision(15);
    OUT.open(output_file.c_str());
    if (! OUT)
    {
      cerr << "Error : can't write to file " << output_file << endl;
      exit(-1);
    }
    for (int row = 0;row < shuffled_training_data.rows()-1;row++)
    {
        for (int col = 0;col < rbmParam.ngram_size -1;col++)
        {
            OUT<<shuffled_training_data(row,col)<<"\t";
        }
        //dont want an extra tab at the end
        OUT<<shuffled_training_data(row,rbmParam.ngram_size-1);
        OUT<<endl;
    }
    //now writing the last row
    for (int col = 0;col < rbmParam.ngram_size -1;col++)
    {
            OUT<<shuffled_training_data(shuffled_training_data.rows()-1,col)<<"\t";
    }
    OUT<<shuffled_training_data(shuffled_training_data.rows()-1,rbmParam.ngram_size-1);
    OUT.close();
  

}

