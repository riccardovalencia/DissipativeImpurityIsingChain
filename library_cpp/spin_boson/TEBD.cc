#include "TEBD.h"
#include "observables.h"
#include "MyClasses.h"
#include "state_manipulation.h"
#include <itensor/all.h>
#include <iostream>
#include <math.h>       
#include <complex>
using namespace std;
using namespace itensor;

// -----------------------------------------------------------------
// Gates of exp(-i dt H) with spin Hamiltonian H
//  H  =  - hx \sum_j X_j - Jxx \sum_j X_j X_{j+1} 
//        - hy \sum_j X_j - Jyy \sum_j Y_j Y_{j+1}  
//        - hz \sum_j X_j - Jzz \sum_j Z_j Z_{j+1}
// where J = [Jxx,Jyy,Jzz] and h = [hx,hy,hz]

vector<MyBondGate>
gates_spin_model(const SiteSet sites , const vector<double> J, const vector<double> h, const double dt)
{

	int N = length(sites);

    vector<MyBondGate> gates;

	cerr << "vector J = (J_xx, J_yy , J_zz)\n";
	double Jxx = J[0];
	double Jyy = J[1];
	double Jzz = J[2];
	cerr << Jxx << "\n" << Jyy << "\n" << Jzz << "\n";

	double hx  = h[0];
	double hy  = h[1];
	double hz  = h[2];

	for(int j=1 ; j <= N-1 ; j+=1)
	{
		vector<ITensor> X;
		vector<ITensor> Y;
		vector<ITensor> Z;
		vector<ITensor> Id;

		for(int q=j ; q<=j+1; q++)
		{
			Id.push_back(      op(sites,"Id",q) );
			X.push_back(  2 * op(sites,"Sx",q) );
			Y.push_back(  2 * op(sites,"Sy",q) );
			Z.push_back(  2 * op(sites,"Sz",q) );
		}

		ITensor H_S , H_SS;

		if(j==1) H_S = (hx * X[0] + hy * Y[0] + hz * Z[0]) * Id[1] ;
		else     H_S = (hx * X[0] + hy * Y[0] + hz * Z[0]) * Id[1] / 2.;

		if(j <  N-1) H_S += Id[0] * (hx * X[1] + hy * Y[1] + hz * Z[1]) / 2. ;
		else         H_S += Id[0] * (hx * X[1] + hy * Y[1] + hz * Z[1])      ;

		H_SS  = Jxx * X[0]*X[1] + Jyy * Y[0]*Y[1] + Jzz * Z[0]*Z[1];		
		
		ITensor H = H_S + H_SS;

		vector<int> jn = {j,j+1};
		// BondGate g = BondGate(sites,j,j+1,BondGate::tReal,dt/2.,H); 
		MyBondGate g = MyBondGate(sites,jn,dt/2.,H);
		gates.push_back(g);
	}
	
	vector<MyBondGate> gates_ = gates;
	reverse(gates_.begin(), gates_.end());
	for(MyBondGate gate : gates_) gates.push_back(gate);

	return gates;
}



// -----------------------------------------------------------------
// Gates of exp(-i dt H) with spin Hamiltonian H
//  H  =  - hx \sum_j X_j - Jxx \sum_j X_j X_{j+1} 
//        - hy \sum_j X_j - Jyy \sum_j Y_j Y_{j+1}  
//        - hz \sum_j X_j - Jzz \sum_j Z_j Z_{j+1}
// where J = [Jxx,Jyy,Jzz] and h = [hx,hy,hz]
// Same as the one retuning <MyBondGate>: overload of the function. 
// Depending on the degree of flexibility and control needed could be better to use one over the other

vector<BondGate>
gates_spin_model_bondgate(const SiteSet sites , const vector<double> J, const vector<double> h, const double dt)
{

	int N = length(sites);

    vector<BondGate> gates;

	cerr << "vector J = (J_xx, J_yy , J_zz)\n";
	double Jxx = J[0];
	double Jyy = J[1];
	double Jzz = J[2];
	cerr << Jxx << "\n" << Jyy << "\n" << Jzz << "\n";

	double hx  = h[0];
	double hy  = h[1];
	double hz  = h[2];
	// cerr << hx << "\n" << hy << "\n" << hz << "\n";
	// exit(0);
	for(int j=1 ; j <= N-1 ; j+=1)
	{
		vector<ITensor> X;
		vector<ITensor> Y;
		vector<ITensor> Z;
		vector<ITensor> Id;

		for(int q=j ; q<=j+1; q++)
		{
			Id.push_back(      op(sites,"Id",q) );
			X.push_back(  2 * op(sites,"Sx",q) );
			Y.push_back(  2 * op(sites,"Sy",q) );
			Z.push_back(  2 * op(sites,"Sz",q) );
		}

		ITensor H_S , H_SS;

		if(j==1) H_S = (hx * X[0] + hy * Y[0] + hz * Z[0]) * Id[1] ;
		else     H_S = (hx * X[0] + hy * Y[0] + hz * Z[0]) * Id[1] / 2.;

		if(j <  N-1) H_S += Id[0] * (hx * X[1] + hy * Y[1] + hz * Z[1]) / 2. ;
		else         H_S += Id[0] * (hx * X[1] + hy * Y[1] + hz * Z[1])      ;

		H_SS  = Jxx * X[0] * X[1] + Jyy * Y[0] * Y[1] + Jzz * Z[0] * Z[1];
				
		ITensor H = H_S + H_SS;

		vector<int> jn = {j,j+1};
		BondGate g = BondGate(sites,j,j+1,BondGate::tReal,dt/2.,H); 
		// MyBondGate g = MyBondGate(sites,jn,dt/2.,H);
		gates.push_back(g);
	}
	
	vector<BondGate> gates_ = gates;
	reverse(gates_.begin(), gates_.end());
	for(BondGate gate : gates_) gates.push_back(gate);

	return gates;
}

// effective Hamiltonian of a local dissipative process
// The coherent dynamics is given by the generic short range spin model

vector<BondGate>
gates_spin_eff_model_bondgate(const SiteSet sites , const vector<double> J, const vector<double> h, const vector<ITensor> Lj, const vector<int> Lj_sites, const vector<double> gamma, const double dt)
{

	int N = length(sites);

    vector<BondGate> gates;

	// cerr << "vector J = (J_xx, J_yy , J_zz)\n";
	double Jxx = J[0];
	double Jyy = J[1];
	double Jzz = J[2];
	// cerr << Jxx << "\n" << Jyy << "\n" << Jzz << "\n";

	double hx  = h[0];
	double hy  = h[1];
	double hz  = h[2];
	// cerr << hx << "\n" << hy << "\n" << hz << "\n";
	// exit(0);
	for(int j=1 ; j <= N-1 ; j+=1)
	{
		vector<ITensor> X;
		vector<ITensor> Y;
		vector<ITensor> Z;
		vector<ITensor> Id;

		for(int q=j ; q<=j+1; q++)
		{
			Id.push_back(      op(sites,"Id",q) );
			X.push_back(  2 * op(sites,"Sx",q) );
			Y.push_back(  2 * op(sites,"Sy",q) );
			Z.push_back(  2 * op(sites,"Sz",q) );
		}

		ITensor H_S , H_SS;

		if(j==1) H_S = (hx * X[0] + hy * Y[0] + hz * Z[0]) * Id[1] ;
		else     H_S = (hx * X[0] + hy * Y[0] + hz * Z[0]) * Id[1] / 2.;

		if(j <  N-1) H_S += Id[0] * (hx * X[1] + hy * Y[1] + hz * Z[1]) / 2. ;
		else         H_S += Id[0] * (hx * X[1] + hy * Y[1] + hz * Z[1])      ;

		for(int k=0 ; k<= Lj_sites.size(); k++)
		{
			if(Lj_sites[k]==j)
			{
				ITensor lj  = Lj[k];
				ITensor ljd = conj(lj);
				ljd.mapPrime(0,2); // I have to do L^dag L, which is (L^*)^T L (this is why I make the 'row' index the 'column' one)
				ITensor ljdlj = ljd * lj; 
				ljdlj.mapPrime(2,1);
				H_S -= 0.5 * gamma[k] * Cplx_i * ljdlj * Id[1]; 
			}
			if(j==N-1 && Lj_sites[k] == N)
			{
				ITensor lj = Lj[k];
				ITensor ljd = conj(lj);
				ljd.mapPrime(0,2); // I have to do L^dag L, which is (L^*)^T L (this is why I make the 'row' index the 'column' one)
				ITensor ljdlj = ljd * lj; 
				ljdlj.mapPrime(2,1);
				H_S -= 0.5 * gamma[k] * Cplx_i * Id[0] * ljdlj;
			}
		}

		H_SS  = Jxx * X[0] * X[1];
		H_SS += Jyy * Y[0] * Y[1];
		H_SS += Jzz * Z[0] * Z[1];		
		
		ITensor H = H_S + H_SS;

		vector<int> jn = {j,j+1};
		BondGate g = BondGate(sites,j,j+1,BondGate::tReal,dt/2.,H); 
		// MyBondGate g = MyBondGate(sites,jn,dt/2.,H);
		gates.push_back(g);
	}
	
	vector<BondGate> gates_ = gates;
	reverse(gates_.begin(), gates_.end());
	for(BondGate gate : gates_) gates.push_back(gate);

	return gates;
}



// Simulation of free spinful fermions 

vector<BondGate>
gates_free_spinful_fermions(const SiteSet sites , const vector<double> J, const vector<double> hup, const vector<double> hdn, const double dt)
{ 


	int N = length(sites);

    vector<BondGate> gates;
	
	for(int j=1 ; j <= N-1 ; j+=1)
	{

	
		vector<ITensor> Adagup;
		vector<ITensor> Aup;

		vector<ITensor> Adagdn;
		vector<ITensor> Adn;

		vector<ITensor> Nup;
		vector<ITensor> Ndn;

		vector<ITensor> Id;

		vector<ITensor> AdagupFi;
		vector<ITensor>	AupFi;
		vector<ITensor>	FiAdn;
		vector<ITensor>	FiAdagdn;

			
		for(int q=j ; q<=j+1; q++)
		{
			Adagup.push_back(op(sites,"Adagup",q) );
			Aup.push_back(   op(sites,"Aup",q) );
			Adagdn.push_back(op(sites,"Adagdn",q) );
			Adn.push_back(   op(sites,"Adn",q) );
			Nup.push_back(   op(sites,"Nup",q) );
			Ndn.push_back(   op(sites,"Ndn",q) );
			Id.push_back(op(sites,"Id",q));

			// see https://itensor.org/docs.cgi?page=tutorials/fermions
		
			AdagupFi.push_back(op(sites,"Adagup*F",q));
			AupFi.push_back(op(sites,"Aup*F",q));
			FiAdn.push_back(op(sites,"F*Adn",q));
			FiAdagdn.push_back(op(sites,"F*Adagdn",q));


		}

		ITensor H, H_S , H_SS;

		// build single site
	
		if(j ==1)    H_S = (hup[j-1] * Nup[0] + hdn[j-1] * Ndn[0]) * Id[1] ;
		else         H_S = (hup[j-1] * Nup[0] + hdn[j-1] * Ndn[0]) * Id[1] / 2.;

		if(j == N-1) H_S += Id[0] * (hup[j] * Nup[1] + hdn[j] * Ndn[1]) ;
		else         H_S += Id[0] * (hup[j] * Nup[1] + hdn[j] * Ndn[1]) / 2.      ;

		// two sites hopping

		H_SS   = J[j-1] * (  AdagupFi[0] * Aup[1]   - AupFi[0] * Adagup[1] );
		H_SS  += J[j-1] * (  Adagdn[0]   * FiAdn[1] - Adn[0] * FiAdagdn[1] );
		
		
		H = H_S + H_SS ;
		BondGate g = BondGate(sites,j,j+1,BondGate::tReal,dt/2.,H); 
		gates.push_back(g);
			
	}


	vector<BondGate> gates_reversed = gates;
	
	reverse(gates_reversed.begin(), gates_reversed.end());

	for(BondGate g : gates_reversed) gates.push_back(g);

	return gates;
}



// Gates of the PXP Hamiltonian
// H = omega \sum_j P_j X_{j-1} P_{j+1}
// where P_j = (1+Z_j)/2

vector<MyBondGate>
gates_pxp(const SiteSet sites , const double omega, const double dt)
{

	int N = length(sites);

	
	vector<MyBondGate> gates;

	// first layer (acts on sites [1,2,3] , [4,5,6] , ... )
	for(int j=1 ; j <= N-2 ; j+=3)
	{
		ITensor P1 = (op(sites,"Id",j) + 2*op(sites,"Sz",j))/2;
		ITensor X2 = 2*op(sites,"Sx",j+1);
		ITensor P3 = (op(sites,"Id",j+2) + 2*op(sites,"Sz",j+2))/2;
		vector<int> jn = {j,j+1,j+2};
		MyBondGate g = MyBondGate(sites,jn,dt/2.,omega*P1*X2*P3);
		gates.push_back(g);
	}

	// second layer (acts on sites [2,3,4] , [5,6,7] , ... )
	for(int j=2 ; j <= N-2 ; j+=3)
	{
		ITensor P1 = (op(sites,"Id",j) + 2*op(sites,"Sz",j))/2;
		ITensor X2 = 2*op(sites,"Sx",j+1);
		ITensor P3 = (op(sites,"Id",j+2) + 2*op(sites,"Sz",j+2))/2;
		vector<int> jn = {j,j+1,j+2};
		MyBondGate g = MyBondGate(sites,jn,dt/2.,omega*P1*X2*P3);
		gates.push_back(g);
	}

	// third layer (acts on sites [3,4,5] , [6,7,8] , ... )
	for(int j=3 ; j <= N-2 ; j+=3)
	{
		ITensor P1 = (op(sites,"Id",j) + 2*op(sites,"Sz",j))/2;
		ITensor X2 = 2*op(sites,"Sx",j+1);
		ITensor P3 = (op(sites,"Id",j+2) + 2*op(sites,"Sz",j+2))/2;
		vector<int> jn = {j,j+1,j+2};
		MyBondGate g = MyBondGate(sites,jn,dt/2.,omega*P1*X2*P3);
		gates.push_back(g);
	}


	// if(open_system == true) cerr << "TO DO : Include non-hermitian part" << endl;

	vector<MyBondGate> gates_ = gates;
	reverse(gates_.begin(), gates_.end());

	for(MyBondGate gate : gates_) gates.push_back(gate);
	
	return gates;
}


// Rydberg Hamiltonian - we keep up to nearest neighbor interactions

vector<MyBondGate>
gates_rydberg_up_to_VNN(const SiteSet sites , const vector<double> Deltaj, const vector<double> Omegaj, const vector<double> Vj, const double dt)
{

	int N = length(sites);

	vector<MyBondGate> gates;


	for(int j=1 ; j <= N-1 ; j+=1)
	{
		
		vector<ITensor> Nj;
		vector<ITensor> Ij;
		vector<ITensor> Xj;
		for(int q=j ; q<=j+1; q++)
		{
			Nj.push_back(  (op(sites,"Id",q)   - 2*op(sites,"Sz",q))  /2. );
			Ij.push_back(   op(sites,"Id",q) );
			Xj.push_back(   op(sites,"Sx",q) );
		}

		// for(ITensor A : Nj) PrintData(A);

		
		double V = Vj[j-1];
		double Omega1 = Omegaj[j-1];
		double Omega2 = Omegaj[j];
		double Delta1 = Deltaj[j-1];
		double Delta2 = Deltaj[j];

		if(j<N-1)
		{
			Omega2 /= 2.;
		 	Delta2 /= 2.;
		}

		if(j>1)
		{
			Omega1 /= 2.;
			Delta1 /= 2.;
		}



		ITensor H_om, H_N, H_NN; 
		H_om = Omega1 * Xj[0] * Ij[1] + Omega2 * Ij[0] * Xj[1];
		H_N  = Delta1 * Nj[0] * Ij[1] + Delta2 * Ij[0] * Nj[1];
		H_NN = V * Nj[0] * Nj[1];

		ITensor H = H_NN + H_N + H_om;
	


		vector<int> jn = {j,j+1};
		MyBondGate g = MyBondGate(sites,jn,dt/2.,H);
		gates.push_back(g);
	}
	

	vector<MyBondGate> gates_ = gates;
	reverse(gates_.begin(), gates_.end());

	for(MyBondGate gate : gates_) gates.push_back(gate);
	
	return gates;
}

// Rydberg Hamiltonian - we keep up to next-nearest neighbor interactions
// 1. We split H = H_1 + H_2 + H_3, so that [H_i,H_j] \neq 0 while the elements within each H_i commute.
// 2. We prepare the gates for H_j, and then we put them inside a time-evolving operator U_j (of time step dt/2) via SVDs. Namely: we construct the gates and then the resulting MPO
// 3. Either we return the vector [U_1,U_2,U_3,U_3,U_2,U_1]. Or we multiply the MPOs in order to have a single one.

// PLUS: it does not split the single-site terms separately. You earn ~30% in computation time

vector<MyBondGate>
gates_rydberg_up_to_VNNN(const SiteSet sites , const vector<double> Deltaj, const vector<double> Omegaj, const vector<double> Vj, const double dt)
{

	int N = length(sites);

	vector<MyBondGate> gates;
	vector<double> omega;
	vector<double> delta;

	// first layer (acts on sites [1,2,3] , [4,5,6] , ... )
	for(int j=1 ; j <= N-2 ; j+=3)
	{
		int js = j;
		int jf = js + 2;
		
		vector<ITensor> Nj;
		vector<ITensor> Ij;
		vector<ITensor> Xj;

		if(js==1)
			{
			omega = {Omegaj[j-1] , Omegaj[j]/2. , Omegaj[j+1]/3.};
			delta = {Deltaj[j-1] , Deltaj[j]/2. , Deltaj[j+1]/3.};
		}
		else if(js==2)
		{
			omega = {Omegaj[j-1]/2. , Omegaj[j]/3. , Omegaj[j+1]/3.};
			delta = {Deltaj[j-1]/2. , Deltaj[j]/3. , Deltaj[j+1]/3.};
		}
		else if(jf==N-1)
		{
			omega = {Omegaj[j-1]/3. , Omegaj[j]/3. , Omegaj[j+1]/2.};
			delta = {Deltaj[j-1]/3. , Deltaj[j]/3. , Deltaj[j+1]/2.};
		}
		else if(jf==N)
		{
			omega = {Omegaj[j-1]/3. , Omegaj[j]/2. , Omegaj[j+1]};
			delta = {Deltaj[j-1]/3. , Deltaj[j]/2. , Deltaj[j+1]};
		}
		else
		{
			omega = {Omegaj[j-1]/3. , Omegaj[j]/3. , Omegaj[j+1]/3.};
			delta = {Deltaj[j-1]/3. , Deltaj[j]/3. , Deltaj[j+1]/3.};
		}

		for(int q=j ; q<=j+2; q++)
		{
			Nj.push_back(  (op(sites,"Id",q)   - 2*op(sites,"Sz",q))  /2. );
			Ij.push_back(   op(sites,"Id",q) );
			Xj.push_back(   op(sites,"Sx",q)) ; 
		}

		double V12 = Vj[j-1];
		double V23 = Vj[j];

		double r1 = pow(1/V12, 1./6);
		double r2 = pow(1/V23, 1./6);
		double V13 = pow(1/(r1+r2),6.);

		if(j<N-2) V23 /= 2.;		
		if(j>1)   V12 /= 2.;

		ITensor H1, H_NN ;
		H_NN  = V12 * Nj[0] * Nj[1] * Ij[2] ;
		H_NN += V23 * Ij[0] * Nj[1] * Nj[2] ;
		H_NN += V13 * Nj[0] * Ij[1] * Nj[2] ;


		H1  =  omega[0] * Xj[0] * Ij[1] * Ij[2];
		H1  += omega[1] * Ij[0] * Xj[1] * Ij[2];
		H1  += omega[2] * Ij[0] * Ij[1] * Xj[2];

		H1  += delta[0] * Nj[0] * Ij[1] * Ij[2];
		H1  += delta[1] * Ij[0] * Nj[1] * Ij[2];
		H1  += delta[2] * Ij[0] * Ij[1] * Nj[2];


		ITensor H = H_NN + H1;

		vector<int> jn = {j,j+1,j+2};
		MyBondGate g = MyBondGate(sites,jn,dt/2.,H);
		gates.push_back(g);
	}

	// second layer (acts on sites [2,3,4] , [5,6,7] , ... )
	for(int j=2 ; j <= N-2 ; j+=3)
	{
		int js = j;
		int jf = js + 2;
		
		vector<ITensor> Nj;
		vector<ITensor> Ij;
		vector<ITensor> Xj;

		if(js==1)
			{
			omega = {Omegaj[j-1] , Omegaj[j]/2. , Omegaj[j+1]/3.};
			delta = {Deltaj[j-1] , Deltaj[j]/2. , Deltaj[j+1]/3.};
		}
		else if(js==2)
		{
			omega = {Omegaj[j-1]/2. , Omegaj[j]/3. , Omegaj[j+1]/3.};
			delta = {Deltaj[j-1]/2. , Deltaj[j]/3. , Deltaj[j+1]/3.};
		}
		else if(jf==N-1)
		{
			omega = {Omegaj[j-1]/3. , Omegaj[j]/3. , Omegaj[j+1]/2.};
			delta = {Deltaj[j-1]/3. , Deltaj[j]/3. , Deltaj[j+1]/2.};
		}
		else if(jf==N)
		{
			omega = {Omegaj[j-1]/3. , Omegaj[j]/2. , Omegaj[j+1]};
			delta = {Deltaj[j-1]/3. , Deltaj[j]/2. , Deltaj[j+1]};
		}
		else
		{
			omega = {Omegaj[j-1]/3. , Omegaj[j]/3. , Omegaj[j+1]/3.};
			delta = {Deltaj[j-1]/3. , Deltaj[j]/3. , Deltaj[j+1]/3.};
		}

		for(int q=j ; q<=j+2; q++)
		{
			Nj.push_back(  (op(sites,"Id",q)   - 2*op(sites,"Sz",q))  /2. );
			Ij.push_back(   op(sites,"Id",q) );
			Xj.push_back(   op(sites,"Sx",q)) ; 

		}

		
		
		double V12 = Vj[j-1];
		double V23 = Vj[j];

		double r1 = pow(1/V12, 1./6);
		double r2 = pow(1/V23, 1./6);
		double V13 = pow(1/(r1+r2),6.);

		if(j < N-2) V23 /= 2.;
		if(j > 1)   V12 /= 2.;

		ITensor H1, H_NN ;
		H_NN  = V12 * Nj[0] * Nj[1] * Ij[2] ;
		H_NN += V23 * Ij[0] * Nj[1] * Nj[2] ;
		H_NN += V13 * Nj[0] * Ij[1] * Nj[2] ;


		H1  =  omega[0] * Xj[0] * Ij[1] * Ij[2];
		H1  += omega[1] * Ij[0] * Xj[1] * Ij[2];
		H1  += omega[2] * Ij[0] * Ij[1] * Xj[2];

		H1  += delta[0] * Nj[0] * Ij[1] * Ij[2];
		H1  += delta[1] * Ij[0] * Nj[1] * Ij[2];
		H1  += delta[2] * Ij[0] * Ij[1] * Nj[2];

		ITensor H = H_NN + H1;

		vector<int> jn = {j,j+1,j+2};
		MyBondGate g = MyBondGate(sites,jn,dt/2.,H);
		gates.push_back(g);
	}

	// third layer (acts on sites [3,4,5] , [6,7,8] , ... )
	for(int j=3 ; j <= N-2 ; j+=3)
	{
		int js = j;
		int jf = js + 2;
		
		vector<ITensor> Nj;
		vector<ITensor> Ij;
		vector<ITensor> Xj;

		if(js==1)
			{
			omega = {Omegaj[j-1] , Omegaj[j]/2. , Omegaj[j+1]/3.};
			delta = {Deltaj[j-1] , Deltaj[j]/2. , Deltaj[j+1]/3.};
		}

		else if(js==2)
		{
			omega = {Omegaj[j-1]/2. , Omegaj[j]/3. , Omegaj[j+1]/3.};
			delta = {Deltaj[j-1]/2. , Deltaj[j]/3. , Deltaj[j+1]/3.};
		}
		else if(jf==N-1)
		{
			omega = {Omegaj[j-1]/3. , Omegaj[j]/3. , Omegaj[j+1]/2.};
			delta = {Deltaj[j-1]/3. , Deltaj[j]/3. , Deltaj[j+1]/2.};
		}
		else if(jf==N)
		{
			omega = {Omegaj[j-1]/3. , Omegaj[j]/2. , Omegaj[j+1]};
			delta = {Deltaj[j-1]/3. , Deltaj[j]/2. , Deltaj[j+1]};
		}
		else
		{
			omega = {Omegaj[j-1]/3. , Omegaj[j]/3. , Omegaj[j+1]/3.};
			delta = {Deltaj[j-1]/3. , Deltaj[j]/3. , Deltaj[j+1]/3.};
		}


		for(int q=j ; q<=j+2; q++)
		{
			Nj.push_back(  (op(sites,"Id",q)   - 2*op(sites,"Sz",q))  /2. );
			Ij.push_back(   op(sites,"Id",q) );
			Xj.push_back(   op(sites,"Sx",q)) ; 
		}
		
		double V12 = Vj[j-1];
		double V23 = Vj[j];

		double r1 = pow(1/V12, 1./6);
		double r2 = pow(1/V23, 1./6);
		double V13 = pow(1/(r1+r2),6.);

		if(j<N-2) V23 /= 2.;
		if(j>1)   V12 /= 2.;

		ITensor H1, H_NN ;
		H_NN  = V12 * Nj[0] * Nj[1] * Ij[2] ;
		H_NN += V23 * Ij[0] * Nj[1] * Nj[2] ;
		H_NN += V13 * Nj[0] * Ij[1] * Nj[2] ;


		H1  =  omega[0] * Xj[0] * Ij[1] * Ij[2];
		H1  += omega[1] * Ij[0] * Xj[1] * Ij[2];
		H1  += omega[2] * Ij[0] * Ij[1] * Xj[2];

		H1  += delta[0] * Nj[0] * Ij[1] * Ij[2];
		H1  += delta[1] * Ij[0] * Nj[1] * Ij[2];
		H1  += delta[2] * Ij[0] * Ij[1] * Nj[2];

		ITensor H = H_NN + H1;

		vector<int> jn = {j,j+1,j+2};
		MyBondGate g = MyBondGate(sites,jn,dt/2.,H);
		gates.push_back(g);
	}


	vector<MyBondGate> gates_ = gates;
	reverse(gates_.begin(), gates_.end());

	for(MyBondGate gate : gates_) gates.push_back(gate);
	
	return gates;
}


// Rydberg Hamiltonian - we keep up to next-nearest neighbor interactions
// 1. We split H = H_1 + H_2 + H_3, so that [H_i,H_j] \neq 0 while the elements within each H_i commute.
// 2. We prepare the gates for H_j, and then we put them inside a time-evolving operator U_j (of time step dt/2) via SVDs. Namely: we construct the gates and then the resulting MPO
// 3. Either we return the vector [U_1,U_2,U_3,U_3,U_2,U_1]. Or we multiply the MPOs in order to have a single one.

// deprecated in favour of gates_rydberg_up_to_VNNN: in the new version we have single-site terms applied together with the 3-site one.

vector<MyBondGate>
gates_rydberg_up_to_VNNN_deprecated(const SiteSet sites , const vector<double> Deltaj, const vector<double> Omegaj, const vector<double> Vj, const double dt)
{

	int N = length(sites);

	vector<MyBondGate> gates;

	// on site terms
	for(int j=1 ; j<= N ; j++)
	{
		ITensor Nj =   (op(sites,"Id",j)   - 2*op(sites,"Sz",j))  /2. ;
		ITensor Xj =    op(sites,"Sx",j);

		ITensor H = Omegaj[j-1] * Xj + Deltaj[j-1] * Nj;
		vector<int> jn = {j};
		MyBondGate g = MyBondGate(sites,jn,dt/2.,H);
		gates.push_back(g);
	}

	// first layer (acts on sites [1,2,3] , [4,5,6] , ... )
	for(int j=1 ; j <= N-2 ; j+=3)
	{
		
		vector<ITensor> Nj;
		vector<ITensor> Ij;

		for(int q=j ; q<=j+2; q++)
		{
			Nj.push_back(  (op(sites,"Id",q)   - 2*op(sites,"Sz",q))  /2. );
			Ij.push_back(   op(sites,"Id",q) );
		}

		
		double V12 = Vj[j-1];
		double V23 = Vj[j];

		double r1 = pow(1/V12, 1./6);
		double r2 = pow(1/V23, 1./6);
		double V13 = pow(1/(r1+r2),6.);
		// cerr << V13 << "\n";

		if(j<N-2) V23 /= 2.;		
		if(j>1)   V12 /= 2.;

		ITensor H_NN ;
		H_NN  = V12 * Nj[0] * Nj[1] * Ij[2] ;
		H_NN += V23 * Ij[0] * Nj[1] * Nj[2] ;
		H_NN += V13 * Nj[0] * Ij[1] * Nj[2] ;


		vector<int> jn = {j,j+1,j+2};
		MyBondGate g = MyBondGate(sites,jn,dt/2.,H_NN);
		gates.push_back(g);
	}
	
	// exit(0);

	// second layer (acts on sites [2,3,4] , [5,6,7] , ... )
	for(int j=2 ; j <= N-2 ; j+=3)
	{
		vector<ITensor> Nj;
		vector<ITensor> Ij;

		for(int q=j ; q<=j+2; q++)
		{
			Nj.push_back(  (op(sites,"Id",q)   - 2*op(sites,"Sz",q))  /2. );
			Ij.push_back(   op(sites,"Id",q) );
		}

		
		
		double V12 = Vj[j-1];
		double V23 = Vj[j];

		double r1 = pow(1/V12, 1./6);
		double r2 = pow(1/V23, 1./6);
		double V13 = pow(1/(r1+r2),6.);
		// V13 = 0.;

		if(j < N-2) V23 /= 2.;
		if(j > 1)   V12 /= 2.;

		ITensor H_NN ;
		H_NN  = V12 * Nj[0] * Nj[1] * Ij[2] ;
		H_NN += V23 * Ij[0] * Nj[1] * Nj[2] ;
		H_NN += V13 * Nj[0] * Ij[1] * Nj[2] ;

		vector<int> jn = {j,j+1,j+2};
		MyBondGate g = MyBondGate(sites,jn,dt/2.,H_NN);
		gates.push_back(g);
	}

	// third layer (acts on sites [3,4,5] , [6,7,8] , ... )
	for(int j=3 ; j <= N-2 ; j+=3)
	{
		vector<ITensor> Nj;
		vector<ITensor> Ij;

		for(int q=j ; q<=j+2; q++)
		{
			Nj.push_back(  (op(sites,"Id",q)   - 2*op(sites,"Sz",q))  /2. );
			Ij.push_back(   op(sites,"Id",q) );

		}
		
		double V12 = Vj[j-1];
		double V23 = Vj[j];

		double r1 = pow(1/V12, 1./6);
		double r2 = pow(1/V23, 1./6);
		double V13 = pow(1/(r1+r2),6.);
		// V13 = 0.;

		if(j<N-2) V23 /= 2.;
		if(j>1)   V12 /= 2.;

		ITensor H_NN ;
		H_NN  = V12 * Nj[0] * Nj[1] * Ij[2] ;
		H_NN += V23 * Ij[0] * Nj[1] * Nj[2] ;
		H_NN += V13 * Nj[0] * Ij[1] * Nj[2] ;

		vector<int> jn = {j,j+1,j+2};
		MyBondGate g = MyBondGate(sites,jn,dt/2.,H_NN);
		gates.push_back(g);
	}

	vector<MyBondGate> gates_ = gates;
	reverse(gates_.begin(), gates_.end());

	for(MyBondGate gate : gates_) gates.push_back(gate);
	
	return gates;
}




vector<MyBondGate>
gates_spin_local_field(const SiteSet sites , vector<double> omegaj, const double dt)
{

	int N = length(sites);
	vector<MyBondGate> gates;

	// H = \sum_j omveja
	for(int j=1 ; j <= N; j++)
	{
		ITensor Sx = 2*op(sites,"Sx",j);
		ITensor Sy = 2*op(sites,"Sy",j);
		ITensor Sz = 2*op(sites,"Sz",j);

		ITensor hj = omegaj[0] * Sx + omegaj[1] * Sy + omegaj[2] * Sz;

		vector<int> jn = {j};

		MyBondGate g = MyBondGate(sites,jn,dt/2.,hj);
		gates.push_back(g);
	}

	
	vector<MyBondGate> gates_ = gates;
	reverse(gates_.begin(), gates_.end());

	for(MyBondGate gate : gates_) gates.push_back(gate);
	
	return gates;
}



// using this, we have the more standard usage of indices, but we have to use the ITensor convention
// for S^- , which corresponds to the standard S^+ convention. 
// CHECK: 03.09.23 IT COULD BE THAT I AM MESSING UP INDICES TECHNICALLY, SINCE I AM ASSOCIATING THE INDEX 0 TO THE KET,
// AND PRIME TO THE BRA. BUT FROM ITENSOR DEFAULT CONVENTION IT COULD BE THAT THEY ARE SWAPPED. THIS IS WHY IT LOOKS LIKE
// MY CONVENTION OF S^- IS THE OPPOSITE OF THE ONE OF ITENSOR (IN REALITY THEY ARE NOT DIFFERENT). I SHOULD CHECK THIS
// REWRITING A PIECE OF CODE CONCERNING THIS AND TESTING WITH A NON-HERMITIAN JUMP.
vector<MyBondGateDiss>
gates_local_lindbland(const SiteSet sites , vector<ITensor> Lj, vector<int> lj_sites, vector<double> gammaj , const double dt)
{

	int N = length(sites);
	vector<MyBondGateDiss> gates;

	// ket has index sj
	// bra has index sj'
	// we need a gate  input (sj,sj') -> (sj'',sj''') output
	// 		   _	
	// 	sj''- | | - sj
	// 		  |	|
	// sj'''- | | - sj'
	// 	

	//  The procedure is: (0,1) -> (2,3) (input-output)
	//   sj''
	//   |
	//   Lj - 1/2 (Ljdag \otimes Lj)
	//   | sj
	//   o-
	//   | sj'
	//   LjdagT - 1/2 (LjdagT \otimes LjT)
	//   |
	//   sj'''

	for(int j : lj_sites)
	{
		Index sj  = sites(j);
		Index sj1 = prime(sj);
		Index sj2 = prime(sj,2);
		Index sj3 = prime(sj,3);

		ITensor Idket = ITensor(sj,sj2);
		ITensor Idbra = ITensor(sj1,sj3);

		for(int q=1 ; q<=dim(sj) ; q++)
		{
			Idket.set(sj(q),sj2(q),1.);
			Idbra.set(sj1(q),sj3(q),1.);
		}


		ITensor lj = Lj[j-1];
		ITensor ljd = conj(lj);
		ITensor lj_ = lj;
		ITensor ljd_ = ljd;

		// non-hermitian Hamiltonian

		// ket 
		ljd_.mapPrime(0,2); // I have to do L^dag L, which is (L^*)^T L (this is why I make the 'row' index the 'column' one)
		ITensor ljdlj_I = ljd_ * lj_ * Idbra; 

		// reset
		lj_ = lj;
		ljd_ = ljd;
	
		// bra
		ljd_.mapPrime(0,3); 
		ITensor I_ljdlj = ljd_ * lj_; // acts on bra  - (3,0)
		I_ljdlj.mapPrime(3,1);      // (3,0) -> (1,0)
		I_ljdlj.mapPrime(0,3);      // (1,0) -> (1,3) I have performed transposition
		I_ljdlj *= Idket;			// acts on bra from (0,1) to (2,3) as desired
		
		// reset
		lj_ = lj;
		ljd_ = ljd;

		// jumps 
		lj_.mapPrime(1,2);
		ljd_.mapPrime(1,3);
		ljd_.mapPrime(0,1);
		ITensor lj_ljd = lj_ * ljd_;


		ITensor Dj = gammaj[j-1] * (lj_ljd - 0.5 * ljdlj_I - 0.5 * I_ljdlj);

		// cerr << inds(Dj) << "\n";
		vector<int> jnket = {j};
		vector<int> jnbra = {j};

		MyBondGateDiss g = MyBondGateDiss(sites,jnket,jnbra,dt,Dj);
	
		gates.push_back(g);

	}

	return gates;
}


// using this, we have the more standard usage of indices, but we have to use the ITensor convention
// for S^- , which corresponds to the standard S^+ convention. 

vector<MyBondGateDiss>
gates_local_lindbland(const SiteSet sites , vector<ITensor> Lj, vector<double> gammaj , const double dt)
{

	int N = length(sites);
	vector<MyBondGateDiss> gates;

	// ket has index sj
	// bra has index sj'
	// we need a gate  input (sj,sj') -> (sj'',sj''') output
	// 		   _	
	// 	sj''- | | - sj
	// 		  |	|
	// sj'''- | | - sj'
	// 	

	//  The procedure is: (0,1) -> (2,3) (input-output)
	//   sj''
	//   |
	//   Lj - 1/2 (Ljdag \otimes Lj)
	//   | sj
	//   o-
	//   | sj'
	//   LjdagT - 1/2 (LjdagT \otimes LjT)
	//   |
	//   sj'''


	for(int j=1 ; j <= N; j++)
	{
		
		Index sj  = sites(j);
		Index sj1 = prime(sj);
		Index sj2 = prime(sj,2);
		Index sj3 = prime(sj,3);

		ITensor Idket = ITensor(sj,sj2);
		ITensor Idbra = ITensor(sj1,sj3);

		for(int q=1 ; q<=dim(sj) ; q++)
		{
			Idket.set(sj(q),sj2(q),1.);
			Idbra.set(sj1(q),sj3(q),1.);
		}


		ITensor lj = Lj[j-1];
		ITensor ljd = conj(lj);
		ITensor lj_ = lj;
		ITensor ljd_ = ljd;

		// non-hermitian Hamiltonian

		// ket 
		ljd_.mapPrime(0,2); // I have to do L^dag L, which is (L^*)^T L (this is why I make the 'row' index the 'column' one)
		ITensor ljdlj_I = ljd_ * lj_ * Idbra; 

		// reset
		lj_ = lj;
		ljd_ = ljd;
	
		// bra
		ljd_.mapPrime(0,3); 
		ITensor I_ljdlj = ljd_ * lj_; // acts on bra  - (3,0)
		I_ljdlj.mapPrime(3,1);      // (3,0) -> (1,0)
		I_ljdlj.mapPrime(0,3);      // (1,0) -> (1,3) I have performed transposition
		I_ljdlj *= Idket;			// acts on bra from (0,1) to (2,3) as desired
		
		// reset
		lj_ = lj;
		ljd_ = ljd;

		// jumps 
		lj_.mapPrime(1,2);
		ljd_.mapPrime(1,3);
		ljd_.mapPrime(0,1);
		ITensor lj_ljd = lj_ * ljd_;


		ITensor Dj = gammaj[j-1] * (lj_ljd - 0.5 * ljdlj_I - 0.5 * I_ljdlj);

		// cerr << inds(Dj) << "\n";
		vector<int> jnket = {j};
		vector<int> jnbra = {j};

		MyBondGateDiss g = MyBondGateDiss(sites,jnket,jnbra,dt,Dj);
	
		gates.push_back(g);

	}

	
	// vector<MyBondGate> gates_ = gates;
	// reverse(gates_.begin(), gates_.end());

	// for(MyBondGate gate : gates_) gates.push_back(gate);
	
	return gates;
}

// keeping as backup - 4.05.23
// vector<MyBondGateDiss>
// gates_local_lindbland(const SiteSet sites , vector<ITensor> Lj, vector<double> gammaj , const double dt)
// {

// 	int N = length(sites);
// 	vector<MyBondGateDiss> gates;

// 	// ket has index sj
// 	// bra has index sj'
// 	// we need a gate  input (sj,sj') -> (sj'',sj''') output
// 	// 		   _	
// 	// 	sj''- | | - sj
// 	// 		  |	|
// 	// sj'''- | | - sj'
// 	// 	

// 	//  The procedure is: (0,1) -> (2,3) (input-output)
// 	//   sj''
// 	//   |
// 	//   Lj - 1/2 (Ljdag \otimes Lj)
// 	//   | sj
// 	//   o-
// 	//   | sj'
// 	//   LjdagT - 1/2 (LjdagT \otimes LjT)
// 	//   |
// 	//   sj'''


// 	for(int j=1 ; j <= N; j++)
// 	{
		
// 		Index sj  = sites(j);
// 		Index sj1 = prime(sj);
// 		Index sj2 = prime(sj,2);
// 		Index sj3 = prime(sj,3);

// 		ITensor Idket = ITensor(sj,sj2);
// 		ITensor Idbra = ITensor(sj1,sj3);

// 		for(int q=1 ; q<=dim(sj) ; q++)
// 		{
// 			Idket.set(sj(q),sj2(q),1.);
// 			Idbra.set(sj1(q),sj3(q),1.);
// 		}


// 		ITensor lj = Lj[j-1];
// 		ITensor ljd = conj(lj);

// 		// non-hermitian Hamiltonian

// 		// v2: I think correct version

// 		ljd.mapPrime(0,2);
// 		ITensor ljdlj_I = ljd * lj * Idbra; // acts on ket (sj,sj') -> (sj'',sj''') as desired
	
// 		ljd.mapPrime(2,0);
// 		ljd.mapPrime(0,3); // (0,1) -> (3,1) (prime order)
// 		ITensor I_ljdlj = ljd * lj; // acts on bra  - (3,0)
// 		// I_ljdlj.mapPrime(0,1);      // (3,0) -> (3,1)
// 		I_ljdlj.mapPrime(3,1);      // (3,0) -> (1,0)
// 		I_ljdlj.mapPrime(0,3);      // (1,0) -> (1,3) I have performed transposition
// 		I_ljdlj *= Idket;			// acts on bra from (0,1) to (2,3) as desired
// 		ljd.mapPrime(3,0); // (0,1) -> (3,1) (prime order)
		
// 		// end v2

// 		// start v1: I think it does not do correctly the transposition
// 		/*

// 		ljd.mapPrime(0,2);
// 		ITensor ljdlj_I = ljd * lj * Idbra; // acts on ket (sj,sj') -> (sj'',sj''') as desired
	
// 		ljd.mapPrime(2,0);
// 		ljd.mapPrime(0,3); // (0,1) -> (3,1) (prime order)
// 		ITensor I_ljdlj = ljd * lj; // acts on bra  - (3,0)
// 		I_ljdlj.mapPrime(0,1);      // (3,0) -> (3,1)
// 		I_ljdlj *= Idket;			// acts on bra from (0,1) to (2,3) as desired
// 		ljd.mapPrime(3,0); // (0,1) -> (3,1) (prime order)
// `		
// 		// end v1
// 		*/ 

// 		// jumps 
// 		lj.mapPrime(1,2);
// 		ljd.mapPrime(1,3);
// 		ljd.mapPrime(0,1);
// 		ITensor lj_ljd = lj * ljd;

	
// 		ITensor Dj = gammaj[j-1] * (lj_ljd - 0.5 * ljdlj_I - 0.5 * I_ljdlj);

// 		// cerr << inds(Dj) << "\n";
// 		vector<int> jnket = {j};
// 		vector<int> jnbra = {j};

// 		MyBondGateDiss g = MyBondGateDiss(sites,jnket,jnbra,dt,Dj);
	
// 		gates.push_back(g);

// 	}

	
// 	// vector<MyBondGate> gates_ = gates;
// 	// reverse(gates_.begin(), gates_.end());

// 	// for(MyBondGate gate : gates_) gates.push_back(gate);
	
// 	return gates;
// }

// Non-diagonal local Lindland 
// We consider Lindbland of the form: Li \rho L_{i+1}^\dagger + 0.5 * {Li L_{i+1}, \rho}
// It appears as a 4-sites gate if we apply the jumps and the non-hermitian part at the same time.
// A possibility is to split differently: the non hermitiain part in the hermitian one
// In this way I have at most 2-sites gates. The number of gates that have to be applied is the same.  <- POSSIBLE EFFICIENCY GAIN(?)
// Drawback: we would have long-range interactions in the final case study both in the Hamiltonian part 
// and jump part -> MULTIPLE LOOPS NECESSARY                                                           -> HUGE INEFFICENCY FROM LOOPING

// MyTrainITensor is a personalized class containing ITensors which have to act either on 


// It is a 4-sites object
// // We apply Li \rho L_j^\dagger + L_j \rho L_i^\dagger - 1/2( {L_i^\dagger L_j , \rho} + {L_j^\dagger L_i,\rho} ) (OR SIMILAR)



vector<MyBondGateDiss>
gates_nearest_neighbour_local_lindbland(const SiteSet sites , vector<MyTrainITensor> TTrain, const double dt)
{

	int N = length(sites);
	vector<MyBondGateDiss> gates;
	// check size of the two containers

	for(MyTrainITensor T : TTrain)
	{
		
		int i = T.i();
		int j = T.j();

		ITensor li = T.Ti();
		ITensor lj = T.Tj();
		ITensor lid = dag(li);
		ITensor ljd = dag(lj);

		double gamma = T.gamma();

		// site index

		cerr << "Sites : " << i << " " << j << "\n";

		if( abs(i-j)>1 )
		{
			cerr << "non-local dissipation still not implemented! Returning empty set of gates.\n";
			return gates;
		}

		if(abs(i-j)==0)
		{

			ITensor lj_ = lj;
			ITensor ljd_ = ljd;

			Index sj  = sites(j);
			Index sj1 = prime(sj);
			Index sj2 = prime(sj,2);
			Index sj3 = prime(sj,3);

			ITensor Idket = ITensor(sj,sj2);
			ITensor Idbra = ITensor(sj1,sj3);

			for(int q=1 ; q<=dim(sj) ; q++)
			{
				Idket.set(sj(q),sj2(q),1.);
				Idbra.set(sj1(q),sj3(q),1.);
			}

			// non-hermitian Hamiltonian

			// ket
			ljd_.mapPrime(0,2);
			ITensor ljdlj_I = ljd_ * lj_ * Idbra; // acts on ket (sj,sj') -> (sj'',sj''') as desired
		
			// reset
			lj_ = lj;
			ljd_ = ljd;

			// bra
			ljd_.mapPrime(0,3); // (0,1) -> (3,1) (prime order)
			ITensor I_ljdlj = ljd_ * lj_; // acts on bra  - (3,0)
			I_ljdlj.mapPrime(3,1);      // (3,0) -> (1,0)
			I_ljdlj.mapPrime(0,3);      // (1,0) -> (1,3) I have performed transposition
			I_ljdlj *= Idket;			// acts on bra from (0,1) to (2,3) as desired
			
			// reset
			lj_ = lj;
			ljd_ = ljd;
			
			// jumps 
			lj_.mapPrime(1,2);
			ljd_.mapPrime(1,3);
			ljd_.mapPrime(0,1);
			ITensor lj_ljd = lj_ * ljd_;

			// all together

			ITensor Dj = gamma * (lj_ljd - 0.5 * ljdlj_I - 0.5 * I_ljdlj);
			// ITensor Dj = gamma * ( - 0.5 * ljdlj_I - 0.5 * I_ljdlj);

			// cerr << inds(Dj) << "\n";
			vector<int> jnket = {j};
			vector<int> jnbra = {j};

			MyBondGateDiss g = MyBondGateDiss(sites,jnket,jnbra,dt/2.,Dj);
		
			gates.push_back(g);

		}

		if(abs(i-j)==1)
		{
			Index si  = sites(i);
			Index si1 = prime(si);
			Index si2 = prime(si,2);
			Index si3 = prime(si,3);

			Index sj  = sites(j);
			Index sj1 = prime(sj);
			Index sj2 = prime(sj,2);
			Index sj3 = prime(sj,3);

			ITensor Idket = ITensor(si,sj,si2,sj2);
			ITensor Idbra = ITensor(si1,sj1,si3,sj3);

			ITensor Id_iket_jbra = ITensor(si,si2,sj1,sj3);
			ITensor Id_ibra_jket = ITensor(sj,sj2,si1,si3);
						
			for(int q=1 ; q<=dim(sj) ; q++)
			{
				Idket.set( si(q) , si2(q) , sj(q)  , sj2(q) , 1.);
				Idbra.set(si1(q) , si3(q) , sj1(q) , sj3(q) , 1.);
				Id_iket_jbra.set(si(q),si2(q),sj1(q),sj3(q) , 1.);
				Id_ibra_jket.set(sj(q),sj2(q),si1(q),si3(q) , 1.);
			}

			// effective Hamiltonian part - jumps act either on the ket or bra, but not both

			// acts on ket

			ITensor lid_ = lid;
			ITensor ljd_ = ljd;
			ITensor li_  = li;
			ITensor lj_  = lj;

			// ket

			lid_.mapPrime(0,2);
			lid_.mapPrime(1,0);
			ljd_.mapPrime(0,2);
			ljd_.mapPrime(1,0);
			li_.mapPrime(1,2);
			lj_.mapPrime(1,2);
			
			ITensor ljd_li_I = ljd_ * li_ ;
			ITensor lid_lj_I = lid_ * lj_ ;
			ITensor LdL_I = (ljd_li_I + lid_lj_I ) * Idbra ; // acts on ket (sj,sj') -> (sj'',sj''') as desired

		
			// reset
			lid_ = lid;
			ljd_ = ljd;
			li_  = li;
			lj_  = lj;

			// bra
			ljd_.mapPrime(0,3); // (0,1) -> (3,1) (prime order)
			lid_.mapPrime(0,3);
			li_.mapPrime(1,3);
			lj_.mapPrime(1,3);
			li_.mapPrime(0,1);
			lj_.mapPrime(0,1);

			// second attempt
			// ljd_.mapPrime(1,3);
			// ljd_.mapPrime(0,1); 
			// lid_.mapPrime(1,3);
			// lid_.mapPrime(0,1); 
			// li_.mapPrime(0,3);
			// lj_.mapPrime(0,3);
			// end second attempt
		
			ITensor I_LdL = ljd_ * li_ + lid_ * lj_; // acts on bra  - (3,0)
			// cerr << I_LdL << endl;
			I_LdL = swapPrime(I_LdL,1,3); // transposition
			// cerr << I_LdL << endl;
			// exit(0);
			// I_LdL.mapPrime(0,1);      // (3,0) -> (3,1)
			// I_LdL.mapPrime(3,1);      // (3,0) -> (1,0)
			// I_LdL.mapPrime(0,3);      // (1,0) -> (1,3) I have performed transposition
			I_LdL *= Idket;			// acts on bra from (0,1) to (2,3) as desired
			
			// reset

			lid_ = lid;
			ljd_ = ljd;
			li_  = li;
			lj_  = lj;

			// jumps

			lj_.mapPrime(1,2);
			ljd_.mapPrime(1,3);
			ljd_.mapPrime(0,1);

			li_.mapPrime(1,2);
			lid_.mapPrime(1,3);
			lid_.mapPrime(0,1);

			ITensor L_Ld = li_ * ljd_ * Id_ibra_jket + lj_ * lid_ * Id_iket_jbra;
 
			ITensor Dij = gamma * (L_Ld - 0.5 * LdL_I - 0.5 * I_LdL);
			// ITensor Dij = gamma * (- 0.5 * LdL_I - 0.5 * I_LdL);
			

			vector<int> jnket = {i,j};
			vector<int> jnbra = {i,j};

			MyBondGateDiss g = MyBondGateDiss(sites,jnket,jnbra,dt/2.,Dij);
		
			gates.push_back(g);

		}

		

	}

	
	vector<MyBondGateDiss> gates_ = gates;
	reverse(gates_.begin(), gates_.end());

	for(MyBondGateDiss gate : gates_) gates.push_back(gate);
	
	return gates;
}


// vector<MyBondGateDiss>
// gates_nearest_neighbour_local_lindbland(const SiteSet sites , vector<MyTrainITensor> TTrain, const double dt)
// {

// 	int N = length(sites);
// 	vector<MyBondGateDiss> gates;
// 	// check size of the two containers

// 	for(MyTrainITensor T : TTrain)
// 	{
		
// 		int i = T.i();
// 		int j = T.j();

// 		ITensor li = T.Ti();
// 		ITensor lj = T.Tj();
// 		ITensor lid = dag(li);
// 		ITensor ljd = dag(lj);

// 		double gamma = T.gamma();

// 		// site index

// 		cerr << "Sites : " << i << " " << j << "\n";

// 		if( abs(i-j)>1 )
// 		{
// 			cerr << "non-local dissipation still not implemented! Returning empty set of gates.\n";
// 			return gates;
// 		}

// 		if(abs(i-j)==0)
// 		{
// 			Index sj  = sites(j);
// 			Index sj1 = prime(sj);
// 			Index sj2 = prime(sj,2);
// 			Index sj3 = prime(sj,3);

// 			ITensor Idket = ITensor(sj,sj2);
// 			ITensor Idbra = ITensor(sj1,sj3);

// 			for(int q=1 ; q<=dim(sj) ; q++)
// 			{
// 				Idket.set(sj(q),sj2(q),1.);
// 				Idbra.set(sj1(q),sj3(q),1.);
// 			}

// 			// non-hermitian Hamiltonian

// 			ljd.mapPrime(0,2);
// 			ITensor ljdlj_I = ljd * lj * Idbra; // acts on ket (sj,sj') -> (sj'',sj''') as desired
		
// 			ljd.mapPrime(2,0);
// 			ljd.mapPrime(0,3); // (0,1) -> (3,1) (prime order)
// 			ITensor I_ljdlj = ljd * lj; // acts on bra  - (3,0)
// 			I_ljdlj.mapPrime(3,1);      // (3,0) -> (1,0)
// 			I_ljdlj.mapPrime(0,3);      // (1,0) -> (1,3) I have performed transposition
// 			I_ljdlj *= Idket;			// acts on bra from (0,1) to (2,3) as desired
// 			ljd.mapPrime(3,0); // (0,1) -> (3,1) (prime order)
			
// 			// jumps 
// 			lj.mapPrime(1,2);
// 			ljd.mapPrime(1,3);
// 			ljd.mapPrime(0,1);
// 			ITensor lj_ljd = lj * ljd;

// 			ITensor Dj = gamma * (lj_ljd - 0.5 * ljdlj_I - 0.5 * I_ljdlj);

// 			// cerr << inds(Dj) << "\n";
// 			vector<int> jnket = {j};
// 			vector<int> jnbra = {j};

// 			MyBondGateDiss g = MyBondGateDiss(sites,jnket,jnbra,dt/2.,Dj);
		
// 			gates.push_back(g);

// 		}

// 		if(abs(i-j)==1)
// 		{
// 			Index si  = sites(i);
// 			Index si1 = prime(si);
// 			Index si2 = prime(si,2);
// 			Index si3 = prime(si,3);

// 			Index sj  = sites(j);
// 			Index sj1 = prime(sj);
// 			Index sj2 = prime(sj,2);
// 			Index sj3 = prime(sj,3);

// 			ITensor Idket = ITensor(si,sj,si2,sj2);
// 			ITensor Idbra = ITensor(si1,sj1,si3,sj3);

// 			ITensor Id_iket_jbra = ITensor(si,si2,sj1,sj3);
// 			ITensor Id_ibra_jket = ITensor(sj,sj2,si1,si3);
						
// 			for(int q=1 ; q<=dim(sj) ; q++)
// 			{
// 				Idket.set( si(q) , si2(q) , sj(q)  , sj2(q) , 1.);
// 				Idbra.set(si1(q) , si3(q) , sj1(q) , sj3(q) , 1.);
// 				Id_iket_jbra.set(si(q),si2(q),sj1(q),sj3(q) , 1.);
// 				Id_ibra_jket.set(sj(q),sj2(q),si1(q),si3(q) , 1.);
// 			}

// 			// effective Hamiltonian part - jumps act either on the ket or bra, but not both

// 			// acts on ket

// 			ITensor lid_ = lid;
// 			ITensor ljd_ = ljd;
// 			ITensor li_  = li;
// 			ITensor lj_  = lj;


// 			lid_.mapPrime(0,2);
// 			lid_.mapPrime(1,0);
// 			ljd_.mapPrime(0,2);
// 			ljd_.mapPrime(1,0);
			
// 			ITensor ljd_li_I = ljd_ * li_ ;
// 			ITensor lid_lj_I = lid_ * lj_ ;
			
// 			ljd_li_I.mapPrime(1,2);
// 			lid_lj_I.mapPrime(1,2);

// 			ITensor LdL_I = (ljd_li_I + lid_lj_I ) * Idbra ; // acts on ket (sj,sj') -> (sj'',sj''') as desired

		
// 			// acts on bra

// 			lid_ = lid;
// 			ljd_ = ljd;
// 			li_  = li;
// 			lj_  = lj;

// 			// first attempt
// 			ljd_.mapPrime(0,3); // (0,1) -> (3,1) (prime order)
// 			lid_.mapPrime(0,3);
// 			li_.mapPrime(1,3);
// 			lj_.mapPrime(1,3);
// 			li_.mapPrime(0,1);
// 			lj_.mapPrime(0,1);

// 			// second attempt
// 			// ljd_.mapPrime(1,3);
// 			// ljd_.mapPrime(0,1); 
// 			// lid_.mapPrime(1,3);
// 			// lid_.mapPrime(0,1); 
// 			// li_.mapPrime(0,3);
// 			// lj_.mapPrime(0,3);
// 			// end second attempt
		
// 			ITensor I_LdL = ljd_ * li_ + lid_ * lj_; // acts on bra  - (3,0)
// 			// I_LdL.mapPrime(0,1);      // (3,0) -> (3,1)
// 			// I_LdL.mapPrime(3,1);      // (3,0) -> (1,0)
// 			// I_LdL.mapPrime(0,3);      // (1,0) -> (1,3) I have performed transposition
// 			I_LdL *= Idket;			// acts on bra from (0,1) to (2,3) as desired
			
// 			// jumps

// 			lid_ = lid;
// 			ljd_ = ljd;
// 			li_  = li;
// 			lj_  = lj;

// 			lj_.mapPrime(1,2);
// 			ljd_.mapPrime(1,3);
// 			ljd_.mapPrime(0,1);

// 			li_.mapPrime(1,2);
// 			lid_.mapPrime(1,3);
// 			lid_.mapPrime(0,1);

// 			ITensor L_Ld = li_ * ljd_ * Id_ibra_jket + lj_ * lid_ * Id_iket_jbra;
 
// 			ITensor Dij = gamma * (L_Ld - 0.5 * LdL_I - 0.5 * I_LdL);

// 			vector<int> jnket = {i,j};
// 			vector<int> jnbra = {i,j};

// 			MyBondGateDiss g = MyBondGateDiss(sites,jnket,jnbra,dt/2.,Dij);
		
// 			gates.push_back(g);

// 		}

		

// 	}

	
// 	vector<MyBondGateDiss> gates_ = gates;
// 	reverse(gates_.begin(), gates_.end());

// 	for(MyBondGateDiss gate : gates_) gates.push_back(gate);
	
// 	return gates;
// }




// Dissipative gate : we have a list of tensors which act 
// We have a Lidbland of the form L_{i,j} \rho L_{i,j}^\dagger + ...

// NOT USEFUL AT THE MOMENT - NOT TESTED (SHOULD WORK)
vector<MyBondGateDiss>
gates_local_nsites_lindbland(const SiteSet sites , vector<ITensor> Lij_list, vector<vector<int> > Lj_sites, vector<double> gammaj , const double dt)
{

	int N = length(sites);
	vector<MyBondGateDiss> gates;


	// check size of the two containers

	if(Lj_sites.size() != Lij_list.size()){
		cerr << "vectors containing jumps and sites where they act have different length!\n";
		cerr << "Lj_sites has length " << Lj_sites.size() << "\n";
		cerr << "Lj has length " << Lij_list.size() << "\n";
		cerr << "Returning empty gates\n";
		return gates;
	}
	

	int M = Lj_sites.size();

	for(int k=0 ; k < M; k++)
	{
		vector<int> jn = Lj_sites[k];
		if(jn.size() > 2){
			cerr << "lindlbland acting on more than two sites not yet implemented!\n Returning empty gates";
			return gates;
		} 

		// sites where it acts
		int i = jn[0];
		int j = jn[1];

		// list of jump operators

		ITensor lij  = Lij_list[k];
		ITensor lijd = dag(lij);  // it is equal to complex conjugation - it does not swap indices to make the transpose

		// vector<ITensor> ljn = Lj_list[k];
		// ITensor lij = ljn[0];
		// ITensor lj = ljn[1];
		// ITensor lid = dag(li);
		// ITensor ljd = dag(lj);

		// site index

		Index si  = sites(i);
		Index si1 = prime(si);
		Index si2 = prime(si,2);
		Index si3 = prime(si,3);

		Index sj  = sites(j);
		Index sj1 = prime(sj);
		Index sj2 = prime(sj,2);
		Index sj3 = prime(sj,3);

		// Identity for the non-hermitian Hamiltonian part
		
		ITensor Idket = ITensor(si,sj,si2,sj2);
		ITensor Idbra = ITensor(si1,sj1,si3,sj3);
		
		for(int q=1 ; q<=dim(sj) ; q++)
		{
			Idket.set( si(q) , si2(q) , sj(q)  , sj2(q) , 1.);
			Idbra.set(si1(q) , si3(q) , sj1(q) , sj3(q) , 1.);
		}

		
		// v2: I think correct version

		lijd.mapPrime(0,2);
		ITensor ljdlj_I = lijd * lij * Idbra; // acts on ket
	
		lijd.mapPrime(2,0);
		lijd.mapPrime(0,3); // (0,1) -> (3,1) (prime order)
		ITensor I_ljdlj = lijd * lij; // acts on bra  - (3,0)
		// I_ljdlj.mapPrime(0,1);      // (3,0) -> (3,1)
		I_ljdlj.mapPrime(3,1);      // (3,0) -> (1,0)
		I_ljdlj.mapPrime(0,3);      // (1,0) -> (1,3) I have performed transposition
		I_ljdlj *= Idket;			// acts on bra from (0,1) to (2,3)
		lijd.mapPrime(3,0); // (0,1) -> (3,1) (prime order)
		
		// start v1: I think it does not do correctly the transposition
		/*

		lijd.mapPrime(0,2);
		ITensor ljdlj_I = lijd * lij * Idbra; // acts on ket
	
		lijd.mapPrime(2,0);
		lijd.mapPrime(0,3); // (0,1) -> (3,1) (prime order)
		ITensor I_ljdlj = lijd * lij; // acts on bra  - (3,0)
		I_ljdlj.mapPrime(0,1);      // (3,0) -> (3,1)
		I_ljdlj *= Idket;			// acts on bra from (0,1) to (2,3) as desired
		lijd.mapPrime(3,0); // (0,1) -> (3,1) (prime order)
`		
		// end v1
		*/ 

		// jumps 
		lij.mapPrime(1,2);
		lijd.mapPrime(1,3);
		lijd.mapPrime(0,1);
		ITensor lj_ljd = lij * lijd;

	
		ITensor Dj = gammaj[k] * (lj_ljd - 0.5 * ljdlj_I - 0.5 * I_ljdlj);

		// cerr << inds(Dj) << "\n";
		vector<int> jnket = {i,j};
		vector<int> jnbra = {i,j};

		MyBondGateDiss g = MyBondGateDiss(sites,jnket,jnbra,dt/2.,Dj);
	
		gates.push_back(g);

	}

	
	vector<MyBondGateDiss> gates_ = gates;
	reverse(gates_.begin(), gates_.end());

	for(MyBondGateDiss gate : gates_) gates.push_back(gate);
	
	return gates;
}


// vector<MyBondGate>
// MPO_global_dissipation_plane_waves(const double gamma, const double k0, const douple sign)
// {
// 	complex<double> ci;
// 	complex<double> i(0,sign*1);
// 	auto ampo = AutoMPO(sites);
// 	for(int j = 1; j < N; ++j)
// 		{
// 		ampo += exp(i * k0 * (j-1)),"S-",j;
// 		}

// 	//Convert the AutoMPO object to an MPO
// 	MPO L = toMPO(ampo);

// }




// vector<MyBondGate>
// MPO_global_dissipation_plane_waves(const double gamma, const double k0, const douple sign)
// {
// 	complex<double> ci;
// 	complex<double> i(0,sign*1);
// 	auto ampo = AutoMPO(sites);
// 	for(int j = 1; j < N; ++j)
// 		{
// 		ampo += exp(i * k0 * (j-1)),"S-",j;
// 		}

// 	//Convert the AutoMPO object to an MPO
// 	MPO L = toMPO(ampo);

// }

MPS
TEBD_lindbland_time_evolve(MPS psi_t, vector<BondGate> gates , vector<MyBondGateDiss> gates_D , Args TEBD_args, bool dissipative , double dt , double T , int steps_save_state, bool normalize, string file_root, double t_start)
{
	int total_steps = int(T/dt);
	int MaxDim = TEBD_args.getInt("MaxDim");
	double cut_off = TEBD_args.getReal("Cutoff");
    for(int k=0 ; k<total_steps ; k++)
    {
        double t = t_start + (k+1)*dt;

    	gateTEvol( gates , dt , dt , psi_t , TEBD_args); 

        if(dissipative)
        {
            for (MyBondGateDiss gate : gates_D)
            {
                vector<int> jket = gate.jnket(); // sites where it acts on ket
                ITensor g        = gate.gate();
                

                int j = jket[0];

                cerr <<  j << " "; 

                ITensor AA = psi_t(j) * psi_t(j+1);
                ITensor dpsi =  g * AA;
                dpsi.mapPrime(1,0);

                AA = AA + dpsi;

                auto [U,S,V] = svd(AA,inds(psi_t(j)),{"Cutoff=",cut_off,"MaxDim=",MaxDim});
                psi_t.set(j,U);
                psi_t.set(j+1,S*V);

            }

            gateTEvol( gates , dt , dt , psi_t , TEBD_args); 
        }


        if(normalize)
        {
            double norm = compute_norm_purifed_impurity(&psi_t);
            psi_t /= norm;
        }
        


        if ( (k+1) % steps_save_state == 0)
        {
			if( maxLinkDim(psi_t) > MaxDim)
			{
				cerr << "Reached max bond dimension. Abort.\n";
				return psi_t;
			}
            writeToFile(tinyformat::format("%s_psi_t%.3f",file_root,t),psi_t); 
        }
    }       
    

	return psi_t;
}


