#include <itensor/all.h>
#include <sys/stat.h>
#include <iostream>
#include <string>
#include <vector>
#include <typeinfo>
#include <math.h>       /* exp */
#include <fstream>	//output file
#include <sstream>	//for ostringstream
#include <iomanip>
#include "../../library_cpp/spin_boson.h"
#include <filesystem>

using namespace std;
using namespace itensor;
namespace fs = std::filesystem;

// We evolve the dissipative Ising model
// We have a dephasing channel on the left-most site.
// The Hamiltonian is
// H = - ZZ - J_x XX - h X
// L = \sqrt{gamma} Z
// For J_x \neq 0 -> non integrable

// Evolution is performed via a 2-sites TEBD

int main(int argc , char* argv[]){
	
    // system size
    int N        = atoi(argv[1]); 
    // local fields
    double hx = atof(argv[2]);
    double hy = 0.;
    double hz = 0.;
    // spin-spin interaction along different directions 
    double Jxx   = atof(argv[3]);
    double Jyy   = 0.;
    double Jzz   = -1.;
    // local fields along z (longitudinal direction) to break explicitly Z2 symmetry 
    // when computing the ground state
    double eps = 0.;

    double T     = atof(argv[4]);  // total time
    double dt    = atof(argv[5]); // timestep
    int maxDim   = atoi(argv[6]); //max bond dimension
    double cut_off = 1E-10;  // cut_off TEBD

    // -------------------------
    int steps_measure;
    if(dt < 0.1) steps_measure = int(0.1/dt);
    else steps_measure = 1;

    int total_steps = int(T / dt);

    cerr << "Input taken\n";
    cerr << N << "\n";
    cerr << hz << "\n";
    cerr << Jxx << "\n";
    cerr << T << "\n";
    cerr << dt << "\n";
    cerr << maxDim << "\n";

    
    SiteSet sites = SpinHalf(N,{"ConserveQNs=",false});

    // Build MPO initial Hamiltonian
    // Run DMRG for finding the ground state, to be used as initial state

    auto ampo = AutoMPO(sites);
    for(int j = 1; j < N; ++j)
    {
        ampo += 4 * Jzz , "Sz" , j , "Sz" , j+1;
        ampo += 4 * Jxx , "Sx" , j , "Sx" , j+1;
    }
    for(int j=1 ; j<= N ; j++)
    {
        ampo += 2 * hx , "Sx", j;
        // ampo += 2 * eps, "Sz", j;
    }
    
    // Convert the AutoMPO object to an MPO
    MPO H = toMPO(ampo);

    cerr << "H created\n";

    auto sweeps = Sweeps(100);
    sweeps.maxdim() = 10,10,10,20,20,40,40,100,200,200;
    sweeps.cutoff() = 1E-14;
    sweeps.noise() = 1E-8,1E-10,1E-12,0;

    MPS psi0 = randomMPS(sites);

    auto [energy,psi] = dmrg(H,psi0,sweeps,{"Quiet",false});
    MPS psi_t0 = psi;
    cerr << setprecision(12);
    cerr << energy << "\n";
    cerr << maxLinkDim(psi) << "\n";
    exit(0);
    
    // observables to measure
    vector<string> name_obs;
    name_obs.push_back("fidelity");
    name_obs.push_back("Sx");
    name_obs.push_back("Sy");
    name_obs.push_back("Sz");

    // name_obs.push_back("Szj"); // IMPLEMENT THE LOCAL MEASURE OF ALL SZ
    // name_obs.push_back("Sx");
    // name_obs.push_back("Sz");
    // name_obs.push_back("Na");
    // name_obs.push_back("entropy");
    name_obs.push_back("MaxD");


    vector<double> Jvec = {Jxx,Jyy,Jzz};
    vector<double> hvec = {hx , hy, hz};

    vector<MyBondGate> gates;
    gates = gates_spin_model(sites,Jvec,hvec,dt);

    cerr << setprecision(10);

    string file_obs    = tinyformat::format("test_ising/TN_Ising_N%d_Jxx%.3f_hx%.3f_dt%.4f_coherent.txt",N,Jxx,hx,dt);
    ofstream save_file( file_obs) ;

    save_file << "# t";
    for(string name : name_obs) save_file << " . " + name;
    save_file << endl;

    save_file << setprecision(10);

    for(int k=0 ; k<total_steps ; k++)
    {
        double t = (k+1)*dt;

        for (MyBondGate g : gates)
        {

            vector<int> jn = g.jn();

            int j = jn[0];
            ITensor AA = g.gate();
            psi.position(j);
            // psi.normalize();

            for(int q : jn) AA *= psi(q);  
            AA.mapPrime(1,0);

            if(jn.size() == 1)
            {
                psi.set(j,AA);
            }

            else if(jn.size() == 2)
            {
                auto [U,S,V] = svd(AA,inds(psi(j)),{"Cutoff=",cut_off,"MaxDim=",maxDim});
                psi.set(j,U);
                psi.set(j+1,S*V);
            }

            else if(jn.size() == 3)
            {
                auto [U,S,V] = svd(AA,inds(psi(j)),{"Cutoff=",cut_off,"MaxDim=",maxDim});
                Index l =  commonIndex(U,S);
                psi.set(j,U);

                auto [U1,S1,V1] = svd(S*V,{inds(psi(j+1)),l},{"Cutoff=",cut_off,"MaxDim=",maxDim});
                psi.set(j+1,U1);
                psi.set(j+2,S1*V1);
            }

            else
            {
            cerr << jn.size() <<"-TEBD not implemented yet!\n";
            exit(0);
            }
            
        }


        if (k % steps_measure == 0)
        {

            save_file << t ;
            for(string name : name_obs)
            {
                if(name=="fidelity")
                {
                    double fidelity = abs(innerC(psi_t0,psi));
                    fidelity *= fidelity;
                    save_file << " " << fidelity;
                    
                }

                if(name=="Sx")
                {
                    vector<double> mj = measure_magnetization(&psi,sites,"x");
                    double Sx = 0.;
                    for(long unsigned int idx=1; idx < mj.size() ; idx++) Sx += mj[idx];
                    save_file << " " << Sx/N ;
                }

                if(name=="Sy")
                {
                    vector<double> mj = measure_magnetization(&psi,sites,"y");
                    double Sy = 0.;
                    for(long unsigned int idx=1; idx < mj.size() ; idx++) Sy += mj[idx];
                    save_file << " " << Sy/N ;
                }



                if(name=="Sz")
                {
                    vector<double> mj = measure_magnetization(&psi,sites,"z");
                    double Sz = 0.;
                    for(long unsigned int idx=1; idx < mj.size() ; idx++) Sz += mj[idx];
                    save_file << " " << Sz/N ;
                }

                
                if(name=="MaxD")
                {
                    save_file << " " << maxLinkDim(psi);
                }

                
            }
            save_file << "\n";
            save_file.flush();

            if (maxLinkDim(psi) == maxDim)
            {
                cerr << "Reached max bond dimension. Interrupting simulation.\n";
                save_file.close();
                exit(0);
            }

        }

        
        cerr << t << " " << maxLinkDim(psi) << "\n";

        
    }       
    save_file.close();
    return 0;
    
}
