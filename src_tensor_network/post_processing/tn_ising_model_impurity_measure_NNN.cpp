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

// Code for recording the outcome of measurements starting from states.


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

// Carefull:
// The bra has to be inverted with respect to the ket in space



int main(int argc , char* argv[]){
	
    // directory where states are stored
    string states_dir = "./";
    // directory where to save the output data
    string data_dir   = "./";

    int N     = atoi(argv[1]);  // system size
    // local fields
    double hx = atof(argv[2]);
    double hy = 0.;
    double hz = 0.; 
    // NN interactions
    double Jxx   = atof(argv[3]);
    double Jyy   = 0.;
    double Jzz   = -1;
    // NNN interactions
    double Jxxx = 0.;
    double Jyyy = 0.;
    double Jzzz = atof(argv[4]);
    double eps   = -0.001; // explictly breaking Z2 symmetry when computing the ground state
    eps = 0.;
    double gamma = atof(argv[5]); // decay rate
    double Tness = atof(argv[6]); // time before you measure the NESS
    double T     = atof(argv[7]); // time to reach after measurement is performed
    double dt    = atof(argv[8]); // timestep
    int maxDim   = atoi(argv[9]); // max bond dimension
    bool compute_autocorrelation = true;
    bool normalized = false;

    // observables to measure
    vector<string> name_obs;
    // name_obs.push_back("fidelity");
    // name_obs.push_back("Sx");
    // name_obs.push_back("Sy");
    // name_obs.push_back("Sz");
    // name_obs.push_back("nj");
    name_obs.push_back("MaxD");
    name_obs.push_back("Z1t_Z0");

    double cut_off = 1E-14;  // cut_off TEBD
    int total_steps = int(100 / dt);
    
    
    string file_root_states = tinyformat::format("%sTN_purification_Ising_N%d_Jxx%.3f_Jzzz%.3f_hx%.3f_gamma%.3f_dt%.4f_D%d_cutoff%.2e_normalize%d",states_dir,N,Jxx,Jzzz,hx,gamma,dt,maxDim,cut_off,normalized);
    string file_root_data   = tinyformat::format("%sTN_purification_Ising_N%d_Jxx%.3f_Jzzz%.3f_hx%.3f_gamma%.3f_dt%.4f_D%d_cutoff%.2e",data_dir,N,Jxx,Jzzz,hx,gamma,dt,maxDim,cut_off);

    string name_file = tinyformat::format("%s_sites",file_root_states);

    SiteSet sites = SpinHalf(2*N,{"ConserveQNs=",false});

    if(fileExists(name_file))
    {
        sites = readFromFile<SiteSet>(name_file);
    }
    else{
        cerr << "SiteSet does not exists.\n";
        cerr << name_file << "\n";
        exit(0);
    }

    string file_obs    = tinyformat::format("%s.txt",file_root_data);
    ofstream save_file( file_obs) ;
    save_file << "# t";
    for(string name : name_obs) save_file << " . " + name;
    save_file << endl;
    save_file << setprecision(14);

    file_obs    = tinyformat::format("%s_nj.txt",file_root_data);
    ofstream save_file_nj( file_obs) ;
    save_file_nj << "# t";
    for(int j : range1(1,N+1)) save_file_nj << " . " << j;
    save_file_nj << endl;
    save_file_nj << setprecision(14);

    for(int k=0 ; k<total_steps ; k++)
    {
        double t = (k+1)*dt;
        name_file = tinyformat::format("%s_psi_t%.3f",file_root_states,t);
        if (fileExists(name_file) )
        {
            cerr << "Opening file " << name_file << "\n";

            MPS psi_t = readFromFile<MPS>(name_file,sites);
            double norm = compute_norm_purifed_impurity(&psi_t);
            psi_t /= norm;


            save_file << t ;
            for(string name : name_obs)
            {
                
                if(name=="Sx")
                {
                    vector<complex<double> > mj = measure_magnetization_impurity_first_site(&psi_t,"x",false);
                    double Sx = 0.;
                    for(complex<double> m : mj) Sx += m.real();
                    save_file << " " << Sx/N ;
                }

                if(name=="Sy")
                {
                    vector<complex<double> > mj = measure_magnetization_impurity_first_site(&psi_t,"y",false);
                    double Sy = 0.;
                    for(complex<double> m : mj) Sy += m.real();
                    save_file << " " << Sy/N ;
                }

                if(name=="Sz")
                {
                    vector<complex<double> > mj = measure_magnetization_impurity_first_site(&psi_t,"z",false);
                    double Sz = 0.;
                    for(complex<double> m : mj) Sz += m.real();
                    save_file << " " << Sz/N ;
                }

                if(name=="nj")
                {
                    save_file_nj << t ;
                    vector<complex<double> > mj = measure_magnetization_impurity_first_site(&psi_t, "x",false);
                    for(complex<double> m : mj) save_file_nj << " " << m.real();
                    save_file_nj << "\n";

                }
               
                if(name=="MaxD")
                {
                    save_file << " " << maxLinkDim(psi_t);
                }

            }
            save_file << " " << norm <<"\n";
            save_file.flush();
            save_file_nj.flush();
           
            cerr << t << " " << maxLinkDim(psi_t) << "\n";
        }
    }       
    
    cerr << "Measuring autocorrelation\n";
    if(compute_autocorrelation)
    {
        string file_root = tinyformat::format("%sTN_purification_Ising_N%d_Jxx%.3f_Jzzz%.3f_hx%.3f_gamma%.3f_dt%.4f_D%d_cutoff%.2e_normalize%d",states_dir,N,Jxx,Jzzz,hx,gamma,dt,maxDim,cut_off,normalized);
        file_obs    = tinyformat::format("%s_Tness%.1f_Z1t_Z0_post.txt",file_root_data,Tness);

        ofstream save_file_Ct( file_obs );
        save_file_Ct << "# t Zt1_Z0\n";
        save_file_Ct << setprecision(14);

        total_steps = int(T / dt);
        for(int k=0 ; k<total_steps ; k++)
        {
            double t = (k+1)*dt;

        
            name_file = tinyformat::format("%s_psi_autocorr_t%.3f",file_root,t);
            // cerr << name_file << "\n";
            if (fileExists(name_file) )
            {
                cerr << "Opening file " << name_file << "\n";
                MPS psi_t = readFromFile<MPS>(name_file,sites);

                // measure magnetization on first physical site, which corresponds to the autocorrelation function
                // due to the regression theorem
                vector<complex<double> >  mj = measure_magnetization_impurity_first_site(&psi_t,"z",false,1);
                save_file_Ct << t << " " << mj[0].real() << " " << mj[0].imag() << " " << abs(mj[0]) << "\n";
                save_file_Ct.flush();
                
                save_file << " " << maxLinkDim(psi_t) << " " << -1 <<"\n";
                save_file.flush();

            }
            

        }

    }
    return  0;
}
