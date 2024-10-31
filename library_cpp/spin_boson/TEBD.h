#ifndef TEBD_H
#define TEBD_H

#include <itensor/all.h>
#include "MyClasses.h"
using namespace std;
using namespace itensor;

vector<MyBondGate>
gates_spin_model(const SiteSet sites , const vector<double> J, const vector<double> h, const double dt);

vector<BondGate>
gates_spin_model_bondgate(const SiteSet sites , const vector<double> J, const vector<double> h, const double dt);

vector<BondGate>
gates_spin_eff_model_bondgate(const SiteSet sites , const vector<double> J, const vector<double> h, const vector<ITensor> Lj, const vector<int> Lj_sites, const vector<double> gamma, const double dt);


vector<BondGate>
gates_free_spinful_fermions(const SiteSet sites , const vector<double> J, const vector<double> hup, const vector<double> hdn, const double dt);


vector<MyBondGate>
gates_pxp(const SiteSet sites , const double omega, const double dt);

vector<MyBondGate>
gates_rydberg_up_to_VNN(const SiteSet sites , const vector<double> Deltaj, const vector<double> Omegaj, const vector<double> Vj, const double dt);

vector<MyBondGate>
gates_rydberg_up_to_VNNN(const SiteSet sites , const vector<double> Deltaj, const vector<double> Omegaj, const vector<double> Vj, const double dt);

vector<MyBondGate>
gates_rydberg_up_to_VNNN_deprecated(const SiteSet sites , const vector<double> Deltaj, const vector<double> Omegaj, const vector<double> Vj, const double dt);




vector<MyBondGate>
gates_spin_local_field(const SiteSet sites , vector<double> omegaj, const double dt);


// override : enables to specify where the jump operators are acting. 
// If lj_sites are specified, it means that the dissipation does not act necessarily globally.

vector<MyBondGateDiss>
gates_local_lindbland(const SiteSet sites , vector<ITensor> Lj, vector<int> lj_sites, vector<double> gammaj , const double dt);

vector<MyBondGateDiss>
gates_local_lindbland(const SiteSet sites , vector<ITensor> Lj, vector<double> gammaj , const double dt);

vector<MyBondGateDiss>
gates_nearest_neighbour_local_lindbland(const SiteSet sites , vector<MyTrainITensor> TTrain, const double dt);

MPS
TEBD_lindbland_time_evolve(MPS psi_t, vector<BondGate> gates , vector<MyBondGateDiss> gates_D , Args TEBD_args, bool dissipative , double dt , double T , int steps_save_state, bool normalize, string file_root, double t_start = 0.);


#endif
