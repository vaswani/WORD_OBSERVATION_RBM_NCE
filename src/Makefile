CXX=/opt/local/bin/g++-mp-4.5
#CXX=/usr/bin/g++
EIGEN=../shared
BOOST=/Users/avaswani/boost/boost_1_52_0/boost
#BOOST=/usr/usc/boost/1.51.0
TCLAP=../shared/tclap1.1.0/include
#TCLAP=../shared/tclap-1.2.1/include
#DEIGEN_NO_DEBUG is imperative for good performance
#CFLAGS=-O3 -DEIGEN_NO_DEBUG -DNDEBUG -m64
CFLAGS=-O3 -DNDEBUG -m64
BOOST_INC=$(BOOST)/include
USEOMP=1
OMP_FLAG=

ifdef USEOMP
	OMP_FLAG=-fopenmp
endif


#Setting the Architecture
all:
	$(CXX) -I$(BOOST_INC) -I$(EIGEN) -I$(TCLAP) $(OMP_FLAG) $(CFLAGS) RBMDahlNCE.cpp -o RBMDahlNCE

clean:
	rm -rf RBMDahlNCE

