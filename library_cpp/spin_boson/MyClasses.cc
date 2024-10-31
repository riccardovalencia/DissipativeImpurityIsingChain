#include "MyClasses.h"
#include <itensor/all.h>
#include <iostream>
#include <math.h>
#include <complex>

using namespace std;
using namespace itensor;

// My classes

MyBondGate::MyBondGate(const SiteSet sites, vector<int> j, const double dt, const ITensor h)
{
	jn_ = j;
	gate_ = expHermitian(h,-1_i * dt);
	sites_ = sites;
}

ITensor MyBondGate::gate()
{
	return gate_;
}

vector<int> MyBondGate::jn()
{
	return jn_;
}

void MyBondGate::modify_gate(ITensor new_gate)
{
	gate_ = new_gate;
}



MyBondGateDiss::MyBondGateDiss(const SiteSet sites, vector<int> jket, vector<int> jbra, double dt, ITensor h)
{
	jnket_ = jket;
	jnbra_ = jbra;
	sites_ = sites;

	// linear approximation - tested and works well for our purposes
	gate_ = h * dt;
	

	// from https://github.com/andyferris/itensor/blob/master/itensor/bondgate.h we can use a higher order expansion 
	// to approximate the exponential
	// Issue is that delta is sparse! 	

	// h *= dt;

	// ITensor unit = delta(inds(h));
	// ITensor term = h;
    // h.mapPrime(1,2);
    // h.mapPrime(0,1);

    // // // exp(x) = 1 + x +  x^2/2! + x^3/3! ..
    // // // = 1 + x * (1 + x/2 *(1 + x/3 * (...
    // // // ~ ((x/3 + 1) * x/2 + 1) * x + 1
    // for(int ord = 100; ord >= 1; --ord)
    //     {
    //     term /= ord;
    //     gate_ = unit + term;
    //     term = gate_ * h;
    //     term.mapPrime(2,1);
    //     }

}

ITensor MyBondGateDiss::gate()
{
	return gate_;
}

vector<int> MyBondGateDiss::jnket()
{
	return jnket_;
}

vector<int> MyBondGateDiss::jnbra()
{
	return jnbra_;
}

MyTrainITensor::MyTrainITensor(vector<ITensor> T, vector<int> j, double gamma)
{
	Ti_   = T[0];
    Tj_   = T[1];
    i_    = j[0];
    j_    = j[1];
	gamma_ = gamma; 
}

ITensor MyTrainITensor::Ti()
{
	return Ti_;
}

ITensor MyTrainITensor::Tj()
{
	return Tj_;
}

int MyTrainITensor::i()
{
	return i_;
}
int MyTrainITensor::j()
{
	return j_;
}
double MyTrainITensor::gamma()
{
	return gamma_;
}

