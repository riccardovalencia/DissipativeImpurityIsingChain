#ifndef INITIAL_STATE_H
#define INITIAL_STATE_H

#include <itensor/all.h>

using namespace std;
using namespace itensor;

MPS
initialize_spin_boson_state(const SiteSet sites , const int n_photon , double theta, double phi);

MPS
initialize_spin_boson_state(const SiteSet sites , const int n_photon , const vector<double> theta, const vector<double> phi);

MPS
initial_computational_state(const SiteSet sites , const vector<int> initial_state);


// void
// insert_state(MPS* psi_t0, MPS state_to_insert, const SiteSet sites, const SiteSet sites_state_to_insert, const int start, const int L, const int N);

void
insert_state(MPS* psi, MPS psi_seed, const int start, bool inverted,bool dagger);

void 
insert_QN_state(MPS* psi, MPS psi_seed, const int start, bool inverted,bool dagger);

#endif
