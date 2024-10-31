#include <itensor/all.h>
#include <sys/stat.h>
#include <iostream>
#include <string>
#include <vector>
#include <typeinfo>
#include <chrono>
#include <math.h>       /* exp */
#include <fstream>	//output file
#include <sstream>	//for ostringstream
#include <iomanip>
#include "../../library_cpp/spin_boson.h"
#include <filesystem>

using namespace std;
using namespace itensor;
namespace fs = std::filesystem;
using namespace std::chrono;

// Quench dynamics starting from the ground state of the integrable Ising chain

// We evolve the dissipative Ising model
// We have a dephasing channel on the left-most site.
// The Hamiltonian is
// H = - ZZ - J_x XX - h X
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

// Coherent evolution is performed via a 2-sites TEBD
// Dissipative evolution is performed via a 2-sites TEBD

// Carefull:
// The bra has to be inverted with respect to the ket in space



int main(int argc , char* argv[]){
 	
    // directory where to save states 
    string main_dir = "./";
    // directory where to save output txt files
    string data_dir = "./";

    // true : normalize at each time step the state
    bool normalize = false;
    // system size
    int N     = atoi(argv[1]); 
    // local fields
    double hx = atof(argv[2]);
    double hy = 0.;
    double hz = 0.; 
    // nearest-neighbor interactions
    double Jxx   = atof(argv[3]);
    double Jyy   = 0.;
    double Jzz   = -1.;
    // local fields along z (longitudinal direction) to break explicitly Z2 symmetry 
    // when computing the ground state
    double eps = 0.;

    double gamma = atof(argv[4]); // decay rate
    double Tness = atof(argv[5]); // time before you measure the NESS
    double T     = atof(argv[6]); // time to reach after measurement is performed
    double dt    = atof(argv[7]); // timestep
    int maxDim   = atoi(argv[8]); // max bond dimension
    bool compute_autocorrelation = true;
    double cut_off = 1E-14;  // cut_off TEBD

    // -------------------------
    int steps_measure;
    if(dt < 0.05) steps_measure = int(0.05/dt);
    else steps_measure = 1;
    int steps_save_state = int(0.2/dt);
    int total_steps = int(Tness / dt);

    auto TEBD_args = Args("Cutoff=",cut_off,"Verbose=",false,"MaxDim=", maxDim, "Normalize=", false  );	

    cerr << "Input taken\n";
    cerr << N << "\n";
    cerr << hz << "\n";
    cerr << Jxx << "\n";
    cerr << gamma << "\n";
    cerr << Tness << "\n";
    cerr << dt << "\n";
    cerr << maxDim << "\n";

    
    SiteSet sites_tmp = SpinHalf(N,{"ConserveQNs=",false});

    // Build MPO initial Hamiltonian
    // Run DMRG for finding the ground state, to be used as initial state
    
    auto ampo = AutoMPO(sites_tmp);
    for(int j = 1; j < N; ++j)
    {
        ampo += 4 * Jzz , "Sz" , j , "Sz" , j+1;
        ampo += 4 * Jxx , "Sx" , j , "Sx" , j+1;
    }
    for(int j=1 ; j<= N ; j++)
    {
        ampo += 2 * hx , "Sx", j;
        ampo += 2 * eps, "Sz", j;
    }

    // Convert the AutoMPO object to an MPO
    MPO H = toMPO(ampo);

    cerr << "H created\n";
    auto sweeps = Sweeps(20);
    sweeps.maxdim() = 10,10,10,20,20,40,40,100,200,200;
    sweeps.cutoff() = 1E-14;
    // sweeps.noise() = 1E-8,1E-10,1E-12,0;
    sweeps.noise() = 0;
    

    MPS psi0 = randomMPS(sites_tmp);
    auto [energy,psi] = dmrg(H,psi0,sweeps,{"Quiet",true});

    // Physical dimension doubling: I map the density matrix to a 2*N MPS where the central bond is
    // the ancillary bond connecting bra and ket

    // chain of 2*N sites
    SiteSet sites = SpinHalf(2*N,{"ConserveQNs=",false});

    MPS psi_t = randomMPS(sites);    
    // inserting bra (it gets inverted and dag) between [1,N]
    insert_state(&psi_t, psi, 1,   true , true);
    // inserting ket bewteen [N+1,2*N]
    insert_state(&psi_t, psi, N+1, false, false);

    MPS psi_t0 = psi_t;
        
    vector<ITensor> Lj;

    // An impurity acting on the first site maps to operators acting on first site of bra and first of ket

    int impurity_sites[2] = {N,N+1};
    for(int j : impurity_sites)
    {
        Lj.push_back( 2. * op(sites,"Sz",j)); 
    }

    // vector<double> Jvec = {Jxx/2,Jyy/2,Jzz/2};
    vector<double> Jvec = {Jxx,Jyy,Jzz};
    vector<double> hvec = {hx , hy, hz};

    bool coherent = true;
    bool dissipative = false;
    if(abs(gamma) > 1E-10) dissipative = true;

    // STRONG DOUBT ABOUT APPLICATION OF THE GATE ON THE IMPURITY BOND
    // I AM DOING PADE APPROXIMATION OF THE EXPONENTIAL OF THE JUMPS/DISSIPATIVE
    // BUT MAYBE I SHOULD DO FIRST ORDER KRAUS APPROXIMATION?
    vector<BondGate>  gates ;
    if(coherent && dissipative)               gates = gates_coherent_part_spin_dissipative_impurity_model(sites , Jvec, hvec, Lj, gamma, dt/2.);
    else if(coherent && dissipative == false) gates = gates_coherent_part_spin_dissipative_impurity_model(sites , Jvec, hvec, Lj, gamma, dt);
    

    vector<MyBondGateDiss> gates_D;
    if(dissipative) gates_D = gates_dissipative_impurity(sites , Lj, gamma , dt);

    cerr << setprecision(14);



    
    
    string name_file_tmp = tinyformat::format("TN_purification_Ising_N%d_Jxx%.3f_hx%.3f_gamma%.3f_dt%.4f_D%d_cutoff%.2e_normalize%d",N,Jxx,hx,gamma,dt,maxDim,cut_off,normalize);
    string file_root = tinyformat::format("%s%s",main_dir,name_file_tmp);
    string file_obs    = tinyformat::format("%s%s.txt",main_dir,name_file_tmp);
    ofstream save_file( file_obs) ;
    save_file << "# t . D\n";
    save_file << setprecision(14);

    double norm = 0.;

    writeToFile(tinyformat::format("%s_sites",file_root),sites); 

    for(int k=0 ; k<total_steps ; k++)
    {
        double t = (k+1)*dt;

    	gateTEvol( gates , dt , dt , psi_t , TEBD_args); 
        if(dissipative)
        {
            for (MyBondGateDiss gate : gates_D)
            {
                vector<int> jket = gate.jnket(); // sites where it acts on ket
                ITensor g        = gate.gate();

                int j = jket[0];

                ITensor AA = psi_t(j) * psi_t(j+1);
                ITensor dpsi =  g * AA;
                dpsi.mapPrime(1,0);

                AA = AA + dpsi;

                auto [U,S,V] = svd(AA,inds(psi_t(j)),{"Cutoff=",cut_off,"MaxDim=",maxDim});
                psi_t.set(j,U);
                psi_t.set(j+1,S*V);

            }

            //norm = compute_norm_purifed_impurity(psi_t);
            //psi_t /= norm;
            gateTEvol( gates , dt , dt , psi_t , TEBD_args); 
        }
        if(normalize)
        {
        norm = compute_norm_purifed_impurity(&psi_t);
        psi_t /= norm;
        }
        
        save_file << t << " " << maxLinkDim(psi_t) << "\n";
        save_file.flush();
           
        if ( (k+1) % steps_save_state == 0)
        {
            writeToFile(tinyformat::format("%s_psi_t%.3f",file_root,t),psi_t); 
        }
    }       
    

    save_file.close();

    norm = compute_norm_purifed_impurity(&psi_t);
    psi_t /= norm;

    cout << "Starting autocorrelation function calculation.\n";

    if(compute_autocorrelation)
    {
        file_root = tinyformat::format("%sTN_purification_Ising_N%d_Jxx%.3f_hx%.3f_gamma%.3f_dt%.4f_D%d_cutoff%.2e_Tness%.1f",data_dir,N,Jxx,hx,gamma,dt,maxDim,cut_off,Tness);
        // I have to apply a gate Z_1 on first site of ket
        ITensor AA = psi_t(N+1);
        ITensor Sz = 2*op(sites,"Sz",N+1);
        AA = AA * Sz;
        AA.mapPrime(1,0);
        psi_t.set(N+1,AA);

        file_obs    = tinyformat::format("%s_Z1t_Z0.txt",file_root);

        ofstream save_file_Ct( file_obs );
        save_file_Ct << "# t Zt1_Z0\n";
        save_file_Ct << setprecision(14);

        total_steps = int(T / dt);
        for(int k=0 ; k<total_steps ; k++)
        {
            double t = (k+1)*dt;

        
    	    gateTEvol( gates , dt , dt , psi_t , TEBD_args); 
            if(dissipative)
            {
                for (MyBondGateDiss gate : gates_D)
                {
                    vector<int> jket = gate.jnket(); // sites where it acts on ket
                    ITensor g        = gate.gate();

                    int j = jket[0];

                    ITensor AA = psi_t(j) * psi_t(j+1);
                    ITensor dpsi =  g * AA;
                    dpsi.mapPrime(1,0);

                    AA = AA + dpsi;

                    auto [U,S,V] = svd(AA,inds(psi_t(j)),{"Cutoff=",cut_off,"MaxDim=",maxDim});
                    psi_t.set(j,U);
                    psi_t.set(j+1,S*V);

                }

                // norm = compute_norm_purifed_impurity(psi_t);
                // psi_t /= norm;

                gateTEvol( gates , dt , dt , psi_t , TEBD_args); 

            }

            // norm = compute_norm_purifed_impurity(psi_t);
            // psi_t /= norm;

            if (k % steps_measure == 0)
            {
                // measure magnetization on first physical site, which corresponds to the autocorrelation function
                // due to the regression theorem
                vector<complex<double> >  mj = measure_magnetization_impurity_first_site(&psi_t,"z",false,1);
                save_file_Ct << t << " " << mj[0].real() << " " << mj[0].imag() << " " << abs(mj[0]) << "\n";

                save_file_Ct.flush();
                /* if (maxLinkDim(psi_t) >= maxDim)
                {
                    cerr << "Reached max bond dimension. Interrupting simulation.\n";
                    save_file.close();
                    exit(0);
                }*/

            }

        //if ( (k+1) % steps_save_state == 0)
        //{
        //    writeToFile(tinyformat::format("%s_psi_autocorr_t%.3f",file_root,t),psi_t); 
        //}
        cerr << t << " " << maxLinkDim(psi_t) << "\n";
        }       

    }

    return 0;
    
}
