#include <itensor/all.h>
#include <iostream>
#include <fstream>	//output file
#include <sstream>	//for ostringstream
#include <iomanip>

using namespace std;
using namespace itensor;

// Quench dynamics starting from the ground state of the integrable Ising chain

// We evolve the dissipative Ising model
// We have a dephasing channel on the left-most site.
// The Hamiltonian is
// H = - ZZ + J_x XX + J_zzz ZIZ + h X
// L = \sqrt{gamma} Z
// For J_x \neq 0 -> non integrable

// Idea: we consider a system of size 2*N
// [1,N] -> describes bra and it evolves with -H
// [N+1,2N] -> describes ket and it evolves with +H
// We have then a non hermitian jump operator on the bond connecting bra and ket

// | | | | |
// o-o-o-o-o-   (ket)
// |
// o-o-o-o-o-   (bra)
// | | | | |

// Carefull:
// The bra is inverted with respect to the ket in space


int main(int argc , char* argv[]){
 	
    bool normalize = false;
    int N     = atoi(argv[1]);  // system size
    // local fields
    double hx_max = atof(argv[2]);
    double hy = 0.;
    double hz = -0.00001; 
    hz = 0.;
    double dhx = 0.05;
    // NN interactions
    double Jxx   = atof(argv[3]);
    double Jyy   = 0.;
    double Jzz   = -1;
    // NNN interactions
    double Jxxx = 0.;
    double Jyyy = 0.;
    double Jzzz = atof(argv[4]);
   
    string main_dir = "./data_DMRG_Ising_up_NNN/";

    string name_file_tmp = tinyformat::format("DMRG_Ising_N%d_Jxx%.3f_Jzzz%.3f_hxmax%.3f",N,Jxx,Jzzz,hx_max);
    string file_root = tinyformat::format("%s%s",main_dir,name_file_tmp);
    string file_obs    = tinyformat::format("%s%s_mzj.txt",main_dir,name_file_tmp);
    ofstream save_file( file_obs) ;
    save_file << "# hx . E . DeltaE . m_half_x . m_half_y . m_half_z\n";
    save_file << setprecision(8);

    
    SiteSet sites = SpinHalf(N,{"ConserveQNs=",false});
    // Build MPO initial Hamiltonian
    // Run DMRG for finding the ground state, to be used as initial state
    writeToFile(tinyformat::format("%s_sites",file_root),sites); 

    double hx = 1.1;
    do
    {

        cerr << hx << "\n";

        auto ampo = AutoMPO(sites);

        for(int j=1 ; j<= N ; j++)
        {
            ampo += 2 * hx , "Sx", j;
            ampo += 2 * hz , "Sz", j;
        }

        for(int j = 1; j < N; ++j)
        {
            ampo += 4 * Jzz , "Sz" , j , "Sz" , j+1;
            ampo += 4 * Jxx , "Sx" , j , "Sx" , j+1;
        }

        for(int j = 1; j < N-1; ++j)
        {
            ampo += 4 * Jzzz , "Sz" , j , "Sz" , j+2;
        }

        MPO H = toMPO(ampo);

            auto sweeps = Sweeps(20);
            sweeps.maxdim() = 10,10,10,20,20,40,40,100,200,200;
            sweeps.cutoff() = 1E-14;
            // sweeps.noise() = 1E-8,1E-10,1E-12,0;
            sweeps.noise() = 0;

        MPS psi0 = randomMPS(sites);
        auto [energy,psi] = dmrg(H,psi0,sweeps,{"Quiet",true});

        writeToFile(tinyformat::format("%s_GS_hx%.3f",file_root,hx),psi); 

        int j_meas = int(N/2);

        ITensor Sx_j = 2*op(sites,"Sx",j_meas);
        ITensor Sy_j = 2*op(sites,"Sy",j_meas);
        ITensor Sz_j = 2*op(sites,"Sz",j_meas);
                
        psi.position(j_meas);
        ITensor ket = psi(j_meas);
		ITensor bra = dag(prime(psi(j_meas),"Site"));
		
		double mx_j =  eltC(bra * Sx_j * ket).real();
        double my_j =  eltC(bra * Sy_j * ket).real();
		double mz_j =  eltC(bra * Sz_j * ket).real();

        double E2  = inner(psi, H , H , psi);

        save_file << hx << " " << energy << " " << E2-energy*energy << " " <<  mx_j << " " << my_j << " " << mz_j << "\n";
        save_file.flush();

        hx += dhx;

    }while(hx <=hx_max);

    
    return 0;
    
}
