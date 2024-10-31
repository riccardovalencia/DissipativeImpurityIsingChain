#ifndef TEBD_EDGE_DISSIPATION_H
#define TEBD_EDGE_DISSIPATION_H

#include <itensor/all.h>
#include "MyClasses.h"
using namespace std;
using namespace itensor;


vector<BondGate>
gates_coherent_unfolded_kondo_impurity_model(const SiteSet sites , const vector<double> J, const vector<double> hup, const vector<double> hdn, const double dt);

vector<MyBondGateDiss>
gates_dissipative_impurity(const SiteSet sites , const vector<ITensor> Lj, const double gamma, const double dt);

vector<BondGate>
gates_dissipative_impurity_high_pade(const SiteSet sites , const vector<ITensor> Lj, const double gamma, const double dt);

vector<BondGate>
gates_coherent_part_spin_dissipative_impurity_model(const SiteSet sites , const vector<double> J, const vector<double> h, const vector<ITensor> Lj, const double gamma, const double dt);

vector<MyBondGate>
gates_coherent_part_spin_dissipative_NNN_interactions_impurity_model(const SiteSet sites , const vector<double> J, const vector<double> J_NNN, const vector<double> h, const double dt);

vector<BondGate>
gates_coherent_unfolded_kondo_impurity_model_energy_basis(const SiteSet sites , const vector<double> J, const vector<double> hup, const vector<double> hdn, const double dt);

vector<MyBondGate>
doubling_space_gates(const vector<BondGate> gates_single, const SiteSet sites_single,  const SiteSet sites_doubled);
#endif
