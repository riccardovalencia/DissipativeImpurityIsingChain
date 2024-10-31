#ifndef STATE_MANIPULATION_H
#define STATE_MANIPULATION_H

#include <itensor/all.h>

using namespace std;
using namespace itensor;


void
swap_gate( MPS *psi, int j1, int j2,double cut_off, int maxDim);

MPO
from_MPS_to_MPDO(MPS psi);

MPO 
from_MPS_to_MPDO_v2(MPS psi );

#endif
