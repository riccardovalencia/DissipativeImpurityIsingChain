#ifndef OBSERVABLES_H
#define OBSERVABLES_H

#include <itensor/all.h>

using namespace std;
using namespace itensor;

 
vector<double>
measure_magnetization(MPS* psi, const SiteSet sites , string direction);

double
entanglement_entropy( MPS* , int  );

double 
measure_kink( MPS* psi, const SiteSet sites);

vector<double>
measure_correlations(MPS* psi, const SiteSet sites, const int start, const bool connected);

vector<complex<double> >
measure_magnetization_impurity_first_site(MPS* psi , string direction, bool compute_normalization = true, int q = -1);

vector<complex<double> >
measure_local_obs_impurity_first_site(MPS *psi , const ITensor O, bool compute_normalization, int q = -1);

double
compute_norm_purifed_impurity(MPS* psi);

double
compute_norm_purifed_impurity_QN(MPS* psi);

complex<double>
measure_correlation_impurity_first_site(MPS *psi , const ITensor O, bool compute_normalization, int q1, int q2, bool connected = false);

#endif
