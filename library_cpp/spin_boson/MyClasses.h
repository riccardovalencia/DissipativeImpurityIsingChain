#ifndef MYCLASSES_H
#define MYCLASSES_H

#include <itensor/all.h>

using namespace std;
using namespace itensor;

class MyBondGate
{
private:
    ITensor gate_;
    vector<int> jn_;
    SiteSet sites_;
public:
	MyBondGate(const SiteSet sites, vector<int> j, const double dt, const ITensor h);
    ITensor gate();
    vector<int> jn();
    void modify_gate(ITensor new_gate);
    // void prime_all(); not working, since the gate_ is private
};

class MyBondGateDiss
{
private:
    ITensor gate_;
    vector<int> jnket_;
    vector<int> jnbra_;
    SiteSet sites_;
public:
	MyBondGateDiss(const SiteSet sites, vector<int> jket, vector<int> jbra, const double dt, const ITensor h);
    ITensor gate();
    vector<int> jnket();
    vector<int> jnbra();
    // void prime_all(); not working, since the gate_ is private
};


// class for keeping ITensors pairs acting one on the ket and one on the bra
// Useful for non-local dissipation 

// class MyTrainITensor
// {
// private:
//     ITensor Tket_;
//     ITensor Tbra_;
//     int jket_;
//     int jbra_;
//     double gamma_;
    
// public:
// 	MyTrainITensor(vector<ITensor> T, vector<int> j, double gamma);
//     ITensor Tket();
//     ITensor Tbra();
//     int jket();
//     int jbra();
//     double gamma();

// };


class MyTrainITensor
{
private:
    ITensor Ti_;
    ITensor Tj_;
    int i_;
    int j_;
    double gamma_;
    
public:
	MyTrainITensor(vector<ITensor> T, vector<int> j, double gamma);
    ITensor Ti();
    ITensor Tj();
    int i();
    int j();
    double gamma();

};




#endif
