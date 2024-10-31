#include "TEBD_edge_dissipation.h"
#include "observables.h"
#include "MyClasses.h"
// #include "/home/ricval/Documenti/MyLibrary/TDVP/tdvp.h"
// #include "/home/ricval/Documenti/MyLibrary/TDVP/basisextension.h"
#include "state_manipulation.h"
#include <itensor/all.h>
#include <iostream>
#include <math.h>       
#include <complex>
using namespace std;
using namespace itensor;


// Simulation of an impurity problem located on the first physical site.
// Can be applied to the unfolded density matrix: 
// the first half sites represent the bra and evolve via  -H
// the second half sites represent the ket and evolve via +H
// the bond in between site N and N+1 is where jump/nonunitary dynamics take place
// This part takes care of the coherent dynamics - we have hopping of free spinful fermions

vector<BondGate>
gates_coherent_unfolded_kondo_impurity_model(const SiteSet sites , const vector<double> J, const vector<double> hup, const vector<double> hdn, const double dt)
{ 
	// careful with local fields if not homogeneous - I have to flip the array. Remember that [1-N] corresponds to bra 
	vector<double> Jket = J;
	vector<double> Jbra = J;

	vector<double> hupket = hup;
	vector<double> hupbra = hup;

	vector<double> hdnket = hdn;
	vector<double> hdnbra = hdn;

	reverse(Jbra.begin()  , Jbra.end());
	reverse(hupbra.begin(), hupbra.end());
	reverse(hdnbra.begin(), hdnbra.end());

	// hopping in the doubled space
	vector<double> J2;
	J2.insert( J2.end(), Jbra.begin(), Jbra.end() );
	// acting on the bond connecting bra - ket (so it should be 0)
	J2.push_back(0.);
	J2.insert( J2.end(), Jket.begin(), Jket.end() );

	// local fields in the doubled space

	vector<double> hup2;
	vector<double> hdn2;

	hup2.insert( hup2.end(), hupbra.begin(), hupbra.end() );
	hup2.insert( hup2.end(), hupket.begin(), hupket.end() );
	hdn2.insert( hdn2.end(), hdnbra.begin(), hdnbra.end() );
	hdn2.insert( hdn2.end(), hdnket.begin(), hdnket.end() );

	// for(double j : hup2) cerr << j << " ";
	// cerr << "\n";
	// for(double j : hdn2) cerr << j << " ";
	// exit(0);
	// N is always even and corresponds to the physical size
	int N2 = length(sites);
	int N = N2/2;

    vector<BondGate> gates;
	vector<BondGate> gates_ket;
	vector<BondGate> gates_bra;
	
	for(int j=1 ; j <= N2-1 ; j+=1)
	{
		// act on bra j \in [1,N] or ket space j \in [N+1,2*N]
		if(j!=N)
		{
			cerr << "Site : " << j << "\n";

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

				// ITensor o = prime(op(sites,"Adagup",q)) * op(sites,"F",q);
				// o.mapPrime(2,1);
				// AdagupFi.push_back(o);

				// o = prime(op(sites,"Aup",q)) * op(sites,"F",q) ;
				// o.mapPrime(2,1);
				// AupFi.push_back(o);

				// o = prime(op(sites,"F",q)) * op(sites,"Adn",q);
				// o.mapPrime(2,1);
				// FiAdn.push_back(o);

				// o = prime(op(sites,"F",q)) * op(sites,"Adagdn",q);
				// o.mapPrime(2,1);
				// FiAdagdn.push_back(o);

			
				AdagupFi.push_back(op(sites,"Adagup*F",q));
				AupFi.push_back(op(sites,"Aup*F",q));
				FiAdn.push_back(op(sites,"F*Adn",q));
				FiAdagdn.push_back(op(sites,"F*Adagdn",q));


			}

			ITensor H, H_S , H_SS;

			// build single site
		
			if(j == 1 || j == (N+1) ) H_S = (hup2[j-1] * Nup[0] + hdn2[j-1] * Ndn[0]) * Id[1] ;
			else                      H_S = (hup2[j-1] * Nup[0] + hdn2[j-1] * Ndn[0]) * Id[1] / 2.;

			if(j == N-1 || j == 2*N-1) H_S += Id[0] * (hup2[j] * Nup[1] + hdn2[j] * Ndn[1]) ;
			else                       H_S += Id[0] * (hup2[j] * Nup[1] + hdn2[j] * Ndn[1]) / 2.      ;

			cerr << "Site : " << j     << " -> h : " << hup2[j-1] << "\n";
			cerr << "Site : " << j + 1 << " -> h : " << hup2[j] << "\n";

			// two sites hopping
			
			if(j>N)
			{
				H_SS   = J2[j-1] * (  AdagupFi[0] * Aup[1]   - AupFi[0] * Adagup[1] );
				H_SS  += J2[j-1] * (  Adagdn[0]   * FiAdn[1] - Adn[0] * FiAdagdn[1] );
			}
			else
			{
				// the ordering of the physical indices is the opposite with respect to the indices
				// used here (the bra is inverted)
				H_SS   = J2[j-1] * (  Aup[0] * AdagupFi[1] - Adagup[0] * AupFi[1] );
				H_SS  += J2[j-1] * (  FiAdn[0] * Adagdn[1] - FiAdagdn[0]  * Adn[1] );
			}
			
			
			cerr << "Site : " << j << " -> " <<  J2[j-1] << endl;
			if(j<N){
				// action on bra (it is exp(+i H^T t))
				// attempt - the sign is due to the convention of the sites 
				H = - H_S - H_SS ;
				H = swapPrime(H,0,1); // NOT SURE
				BondGate g = BondGate(sites,j,j+1,BondGate::tReal,dt/2.,H); 
				gates_bra.push_back(g);
			}
			// action on ket
			else
			{
				H = H_S + H_SS ;
				BondGate g = BondGate(sites,j,j+1,BondGate::tReal,dt/2.,H); 
				gates_ket.push_back(g);
			}
	
		}
		
	}
	

	vector<BondGate> gates_ket_reversed = gates_ket;
	vector<BondGate> gates_bra_reversed = gates_bra;
	
	reverse(gates_ket_reversed.begin(), gates_ket_reversed.end());
	reverse(gates_bra_reversed.begin(), gates_bra_reversed.end());

	for(BondGate g : gates_bra)  		 gates.push_back(g);
	for(BondGate g : gates_ket) 		 gates.push_back(g);
	for(BondGate g : gates_ket_reversed) gates.push_back(g);
	for(BondGate g : gates_bra_reversed) gates.push_back(g);

	return gates;
}



// Dissipative impurity acting on the unfolded density matrix in an impurity problem.
// the first half sites represent the bra and evolve via -H
// the second half sites represent the ket and evolve via +H
// the bond in between site N and N+1 is where jump/nonunitary dynamics take place
// Here we apply the jump/nonunitary part on the bond in between

// There was an error - the swap done at the end was a mistake (referring to modification 03.09.2023)
// for hermitian jump it was not a problem. For non hermitian one yes.

vector<MyBondGateDiss>
gates_dissipative_impurity(const SiteSet sites , const vector<ITensor> Lj, const double gamma, const double dt)
{

	int N = length(sites)/2;

	vector<MyBondGateDiss> gates;

	vector<ITensor> Id;

	for(int q=N ; q<=N+1; q++)
	{
		Id.push_back(     op(sites,"Id",q) );
	}
	// the bond gate acts on site [N,N+1]
	// lj1 acts on bra
	// lj2 acts on ket
	ITensor lj1 = Lj[0];
	ITensor lj2 = Lj[1];

	ITensor lj1d = conj(lj1);
	ITensor lj2d = conj(lj2);
	
	lj1.mapPrime(0,2); // I have to do L^* L
	lj2d.mapPrime(0,2); // I have to do L^dag L, which is (L^*)^T L (this is why I make the 'row' index the 'column' one)

	ITensor ljdlj1 = lj1d * lj1; 
	ITensor ljdlj2 = lj2d * lj2; 
	
	// acting on bra (L^dag L)^T = (L^T L*)
	ljdlj1.mapPrime(2,1);
	// acting on ket (L^dag L)
	ljdlj2.mapPrime(2,1);
	
	// I am here using convention that LdL_I = (L^\dag L) \otimes I is: first operator act on ket, the second on bra
	// Here the first half describe the bra , and the second half ket. This is why I have Id[0] * ljdlj2
	ITensor LdL_I = Id[0] * ljdlj2;
	ITensor I_LdL = ljdlj1 * Id[1];

	lj1 = Lj[0];
	lj2 = Lj[1];
	lj1d = conj(lj1);

	// test 03.09.2023 (it is not relevant for symmetric jumps) - WAS A MISTAKE!
	// lj1d = swapPrime(lj1d,0,1);


	ITensor D = gamma * (lj1d * lj2 - 0.5 * LdL_I - 0.5 * I_LdL);

	vector<int> jnket = {N,N+1};
	vector<int> jnbra = {N,N+1};

	MyBondGateDiss g = MyBondGateDiss(sites,jnket,jnbra,dt,D);

	gates.push_back(g);

	return gates;
}



// Dissipative impurity acting on the unfolded density matrix in an impurity problem.
// the first half sites represent the bra and evolve via -H
// the second half sites represent the ket and evolve via +H
// the bond in between site N and N+1 is where jump/nonunitary dynamics take place
// Here we apply the jump/nonunitary part on the bond in between.

// Differences with gates_dissipative_impurity: above we used a first order Kraus approximation of the 
// dissipative part. Here, I use the class BondGate of ITensor which is able to exponentiate
// also non hermitian things since it uses a high grade Pade approximation

vector<BondGate>
gates_dissipative_impurity_high_pade(const SiteSet sites , const vector<ITensor> Lj, const double gamma, const double dt)
{

	int N = length(sites)/2;

	vector<BondGate> gates;

	vector<ITensor> Id;

	for(int q=N ; q<=N+1; q++)
	{
		Id.push_back(     op(sites,"Id",q) );
	}
	// the bond gate acts on site [N,N+1]
	// lj1 acts on bra
	// lj2 acts on ket
	ITensor lj1 = Lj[0];
	ITensor lj2 = Lj[1];

	ITensor lj1d = conj(lj1);
	ITensor lj2d = conj(lj2);
	
	lj1.mapPrime(0,2); // I have to do L^* L
	lj2d.mapPrime(0,2); // I have to do L^dag L, which is (L^*)^T L (this is why I make the 'row' index the 'column' one)

	ITensor ljdlj1 = lj1d * lj1; 
	ITensor ljdlj2 = lj2d * lj2; 
	
	// acting on bra (L^dag L)^T = (L^T L*)
	ljdlj1.mapPrime(2,1);
	// acting on ket (L^dag L)
	ljdlj2.mapPrime(2,1);
	
	// I am here using convention that LdL_I = (L^\dag L) \otimes I is: first operator act on ket, the second on bra
	// Here the first half describe the bra , and the second half ket. This is why I have Id[0] * ljdlj2
	ITensor LdL_I = Id[0] * ljdlj2;
	ITensor I_LdL = ljdlj1 * Id[1];


	lj1 = Lj[0];
	lj2 = Lj[1];
	lj1d = conj(lj1);

	ITensor D = gamma * (lj1d * lj2 - 0.5 * LdL_I - 0.5 * I_LdL);

	vector<int> jnket = {N,N+1};
	vector<int> jnbra = {N,N+1};

	BondGate g = BondGate(sites,N,N+1,BondGate::tImag,-1*dt,D); 
	gates.push_back(g);

	return gates;
}


// Simulation of an impurity problem located on the first physical site.
// The system is made of N spin-1/2 (no mixed basis).
// Can be applied to the unfolded density matrix: 
// the first half sites represent the bra and evolve via -H
// the second half sites represent the ket and evolve via +H
// the bond in between site N and N+1 is where jump/nonunitary dynamics take place


vector<BondGate>
gates_coherent_part_spin_dissipative_impurity_model(const SiteSet sites , const vector<double> J, const vector<double> h, const vector<ITensor> Lj, const double gamma, const double dt)
{

	// N is always even and corresponds to the physical size
	int N2 = length(sites);
	int N = N2/2;

    vector<BondGate> gates;

	double Jxx = J[0];
	double Jyy = J[1];
	double Jzz = J[2];

	double hx  = h[0];
	double hy  = h[1];
	double hz  = h[2];

	for(int j=1 ; j <= N2-1 ; j+=1)
	{
		cerr << "Site : " << j << "\n";

		vector<ITensor> X;
		vector<ITensor> Y;
		vector<ITensor> Z;
		vector<ITensor> Id;

		for(int q=j ; q<=j+1; q++)
		{
			Id.push_back(     op(sites,"Id",q) );
			X.push_back(  2 * op(sites,"Sx",q) );
			Y.push_back(  2 * op(sites,"Sy",q) );
			Z.push_back(  2 * op(sites,"Sz",q) );
		}

		ITensor H, H_S , H_SS;

		// act on bra j \in [1,N] or ket space j \in [N+1,2*N]
		if(j!=N)
		{
			if(j % N ==1) H_S = (hx * X[0] + hy * Y[0] + hz * Z[0]) * Id[1] ;
			else          H_S = (hx * X[0] + hy * Y[0] + hz * Z[0]) * Id[1] / 2.;

			if(j == N-1 || j == 2*N-1) H_S += Id[0] * (hx * X[1] + hy * Y[1] + hz * Z[1]) ;
			else                       H_S  += Id[0] * (hx * X[1] + hy * Y[1] + hz * Z[1]) / 2.      ;

			H_SS  = Jxx * X[0] * X[1] + Jyy * Y[0] * Y[1] + Jzz * Z[0] * Z[1];

			// action on bra (it is exp(+i H^T t))
			if(j<N){
				H = -H_S - H_SS;
				H = swapPrime(H,0,1);
			}
			// action on ket
			else    H =  H_S + H_SS;
		}


		if(j !=N)
		{
			BondGate g = BondGate(sites,j,j+1,BondGate::tReal,dt/2.,H); 
			gates.push_back(g);
		}
		
		
	}

	for(int j=N2-1 ; j >= 1; j-=1)
	{
		cerr << "Site : " << j << "\n";

		vector<ITensor> X;
		vector<ITensor> Y;
		vector<ITensor> Z;
		vector<ITensor> Id;

		for(int q=j ; q<=j+1; q++)
		{
			Id.push_back(     op(sites,"Id",q) );
			X.push_back(  2 * op(sites,"Sx",q) );
			Y.push_back(  2 * op(sites,"Sy",q) );
			Z.push_back(  2 * op(sites,"Sz",q) );
		}

		ITensor H, H_S , H_SS;

		// act on bra j \in [1,N] or ket space j \in [N+1,2*N]
		if(j!=N)
		{
			if(j ==1 || j==N+1) H_S = (hx * X[0] + hy * Y[0] + hz * Z[0]) * Id[1] ;
			else       H_S = (hx * X[0] + hy * Y[0] + hz * Z[0]) * Id[1] / 2.;

			if(j == N-1 || j == 2*N-1) H_S += Id[0] * (hx * X[1] + hy * Y[1] + hz * Z[1]) ;
			else           H_S   += Id[0] * (hx * X[1] + hy * Y[1] + hz * Z[1]) / 2.      ;

			H_SS  = Jxx * X[0] * X[1] + Jyy * Y[0] * Y[1] + Jzz * Z[0] * Z[1];
	

			// action on bra
			if(j<N){
				H = -H_S - H_SS;
				H = swapPrime(H,0,1);
			}
			// action on ket
			else    H =  H_S + H_SS;
		}

	
		if(j !=N)
		{
			BondGate g = BondGate(sites,j,j+1,BondGate::tReal,dt/2.,H); 
			gates.push_back(g);
		}
		
	}


	
	// vector<BondGate> gates_ = gates;
	// reverse(gates_.begin(), gates_.end());
	// for(BondGate gate : gates_) gates.push_back(gate);

	return gates;
}




// Simulation of an impurity problem located on the first physical site.
// Can be applied to the unfolded density matrix: 
// the first half sites represent the bra and evolve via -H
// the second half sites represent the ket and evolve via +H
// the bond in between site N and N+1 is where jump/nonunitary dynamics take place


vector<MyBondGate>
gates_coherent_part_spin_dissipative_NNN_interactions_impurity_model(const SiteSet sites , const vector<double> J, const vector<double> J_NNN, const vector<double> h, const double dt)
{

	// N is always even and corresponds to the physical size
	int N2 = length(sites);
	int N = N2/2;


    vector<MyBondGate> gates, gates_bra, gates_ket;

	double Jxx = J[0];
	double Jyy = J[1];
	double Jzz = J[2];

	double hx  = h[0];
	double hy  = h[1];
	double hz  = h[2];


	vector<double> hx_tot;
	vector<double> Jzz_tot;
	for(int j : range1(N))   hx_tot.push_back(0);
	for(int j : range1(N-1)) Jzz_tot.push_back(0);

	if(N<3)
	{
		cerr << "Less than 3-sites. Not possible to implement 3-site TEBD.\n";
		exit(0);
	}

	else if(N==3)
	{
		for(int j = 1; j < N2 ; j+=3)
		{
			ITensor H, H_S , H_SS, H_SSS;

			int js = j;
			int jf = js+2;
				
			vector<double> hx_vec = { hx, hx, hx};
			vector<double> hy_vec = { hy, hy, hy};
			vector<double> hz_vec = { hz, hz, hz};
			
			// nearest-neighbor interactions - when acting in the bulk
			vector<double> Jxx_vec = {Jxx,Jxx};
			vector<double> Jyy_vec = {Jyy,Jyy};
			vector<double> Jzz_vec = {Jzz,Jzz};

			vector<ITensor> Xj;
			vector<ITensor> Yj;
			vector<ITensor> Zj;
			vector<ITensor> Ij;

			for(int q=js ; q<= jf; q++)
			{
				Ij.push_back(      op(sites,"Id",q) );
				Xj.push_back(  2 * op(sites,"Sx",q) );
				Yj.push_back(  2 * op(sites,"Sy",q) );
				Zj.push_back(  2 * op(sites,"Sz",q) );
			}

			H_S   = hx_vec[0] * Xj[0] * Ij[1] * Ij[2];
			H_S  += hx_vec[1] * Ij[0] * Xj[1] * Ij[2];
			H_S  += hx_vec[2] * Ij[0] * Ij[1] * Xj[2];

			H_S  += hy_vec[0] * Yj[0] * Ij[1] * Ij[2];
			H_S  += hy_vec[1] * Ij[0] * Yj[1] * Ij[2];
			H_S  += hy_vec[2] * Ij[0] * Ij[1] * Yj[2];

			H_S  += hz_vec[0] * Zj[0] * Ij[1] * Ij[2];
			H_S  += hz_vec[1] * Ij[0] * Zj[1] * Ij[2];
			H_S  += hz_vec[2] * Ij[0] * Ij[1] * Zj[2];

			H_SS   = Jxx_vec[0] * Xj[0] * Xj[1] * Ij[2] ;
			H_SS  += Jxx_vec[1] * Ij[0] * Xj[1] * Xj[2] ;

			H_SS  += Jyy_vec[0] * Yj[0] * Yj[1] * Ij[2] ;
			H_SS  += Jyy_vec[1] * Ij[0] * Yj[1] * Yj[2] ;

			H_SS  += Jzz_vec[0] * Zj[0] * Zj[1] * Ij[2] ;
			H_SS  += Jzz_vec[1] * Ij[0] * Zj[1] * Zj[2] ;

			H_SSS  = J_NNN[0] * Xj[0] * Ij[1] * Xj[2] ;
			H_SSS += J_NNN[1] * Yj[0] * Ij[1] * Yj[2] ;
			H_SSS += J_NNN[2] * Zj[0] * Ij[1] * Zj[2] ;
			
	
			vector<int> jn = {js,js+1,js+2};
			if(j<N){
				// action on bra (it is exp(+i H^T t))

				H = -H_S - H_SS - H_SSS;
				H = swapPrime(H,0,1);

				MyBondGate g = MyBondGate(sites,jn,dt/2.,H);
				gates_bra.push_back(g);
			}
			// action on ket
			else
			{
				H = H_S + H_SS + H_SSS;
				MyBondGate g = MyBondGate(sites,jn,dt/2.,H);
				gates_ket.push_back(g);

			}
		}
	}

	else
	{
		for(int layer = 1 ; layer <= 3 ; layer ++)
		{
			for(int j = layer ; j <= N-2 ; j+=3)
			{
			
				cerr << "Layer : " << layer << " site : " << j << "\n";

				ITensor H, H_S , H_SS, H_SSS;

				int js = j;
				int jf = js + 2;
				
				vector<double> hx_vec,   hy_vec,  hz_vec; 
				// nearest-neighbor interactions - when acting in the bulk
				vector<double> Jxx_vec = {Jxx/2.,Jxx/2.};
				vector<double> Jyy_vec = {Jyy/2.,Jyy/2.};
				vector<double> Jzz_vec = {Jzz/2.,Jzz/2.};

				vector<ITensor> Xj;
				vector<ITensor> Yj;
				vector<ITensor> Zj;
				vector<ITensor> Ij;

				for(int q=js ; q<= jf; q++)
				{
					Ij.push_back(     op(sites,"Id",q) );
					Xj.push_back(  2 * op(sites,"Sx",q) );
					Yj.push_back(  2 * op(sites,"Sy",q) );
					Zj.push_back(  2 * op(sites,"Sz",q) );
				}

				
				if( js % N == 1 )
				{
					hx_vec = {hx, hx/2. , hx/3.};
					hy_vec = {hy, hy/2. , hy/3.};
					hz_vec = {hz, hz/2. , hz/3.};
				}
				else if( js % N == 2 )
				{
					hx_vec = {hx/2. , hx/3. , hx/3.};
					hy_vec = {hy/2. , hy/3. , hy/3.};
					hz_vec = {hz/2. , hz/3. , hz/3.};
				}
				else if(jf % (N-1) == 0)
				{
					hx_vec = {hx/3. , hx/3. , hx/2.};
					hy_vec = {hy/3. , hy/3. , hy/2.};
					hz_vec = {hz/3. , hz/3. , hz/2.};
				}
				else if(jf % N == 0)
				{
					hx_vec = {hx/3. , hx/2. , hx};
					hy_vec = {hy/3. , hy/2. , hy};
					hz_vec = {hz/3. , hz/2. , hz};
				}
				else
				{
					hx_vec = {hx/3. , hx/3. , hx/3.};
					hy_vec = {hy/3. , hy/3. , hy/3.};
					hz_vec = {hz/3. , hz/3. , hz/3.};
				}

				if( js%N == 1) // First physical site (it technically correspond to the last site of the bra, but since couplings are homogeneous it does not matter)
				{
					Jxx_vec[0] = Jxx;
					Jyy_vec[0] = Jyy;
					Jzz_vec[0] = Jzz;
				}

				if( jf%N == 0) // Last physical site
				{
					Jxx_vec[1] = Jxx;
					Jyy_vec[1] = Jyy;
					Jzz_vec[1] = Jzz;
				}


				for(double hj : hx_vec) cerr << hj << " ";
				cerr << "\n";


				Jzz_tot[j-1] += Jzz_vec[0];
				Jzz_tot[j]   += Jzz_vec[1];

				H_S   = hx_vec[0] * Xj[0] * Ij[1] * Ij[2];
				H_S  += hx_vec[1] * Ij[0] * Xj[1] * Ij[2];
				H_S  += hx_vec[2] * Ij[0] * Ij[1] * Xj[2];

				H_S  += hy_vec[0] * Yj[0] * Ij[1] * Ij[2];
				H_S  += hy_vec[1] * Ij[0] * Yj[1] * Ij[2];
				H_S  += hy_vec[2] * Ij[0] * Ij[1] * Yj[2];

				H_S  += hz_vec[0] * Zj[0] * Ij[1] * Ij[2];
				H_S  += hz_vec[1] * Ij[0] * Zj[1] * Ij[2];
				H_S  += hz_vec[2] * Ij[0] * Ij[1] * Zj[2];

				H_SS   = Jxx_vec[0] * Xj[0] * Xj[1] * Ij[2] ;
				H_SS  += Jxx_vec[1] * Ij[0] * Xj[1] * Xj[2] ;

				H_SS  += Jyy_vec[0] * Yj[0] * Yj[1] * Ij[2] ;
				H_SS  += Jyy_vec[1] * Ij[0] * Yj[1] * Yj[2] ;

				H_SS  += Jzz_vec[0] * Zj[0] * Zj[1] * Ij[2] ;
				H_SS  += Jzz_vec[1] * Ij[0] * Zj[1] * Zj[2] ;

				H_SSS  = J_NNN[0] * Xj[0] * Ij[1] * Xj[2] ;
				H_SSS += J_NNN[1] * Yj[0] * Ij[1] * Yj[2] ;
				H_SSS += J_NNN[2] * Zj[0] * Ij[1] * Zj[2] ;
				
		
				vector<int> jn = {js,js+1,js+2};
				if(j<N){
					// action on bra (it is exp(+i H^T t))

					H = -H_S - H_SS - H_SSS;
					H = swapPrime(H,0,1);

					MyBondGate g = MyBondGate(sites,jn,dt/2.,H);
					gates_bra.push_back(g);
				}
				// action on ket
				else
				{
					H = H_S + H_SS + H_SSS;
					MyBondGate g = MyBondGate(sites,jn,dt/2.,H);
					gates_ket.push_back(g);

				}
			}

			// acting on ket
			for(int j = N + layer ; j <= N2-2 ; j+=3)
			{
			
				cerr << "Layer : " << layer << " site : " << j << "\n";

				ITensor H, H_S , H_SS, H_SSS;

				int js = j;
				int jf = js + 2;
				
				vector<double> hx_vec,   hy_vec,  hz_vec; 
				// nearest-neighbor interactions - when acting in the bulk
				vector<double> Jxx_vec = {Jxx/2.,Jxx/2.};
				vector<double> Jyy_vec = {Jyy/2.,Jyy/2.};
				vector<double> Jzz_vec = {Jzz/2.,Jzz/2.};

				vector<ITensor> Xj;
				vector<ITensor> Yj;
				vector<ITensor> Zj;
				vector<ITensor> Ij;

				for(int q=js ; q<= jf; q++)
				{
					Ij.push_back(     op(sites,"Id",q) );
					Xj.push_back(  2 * op(sites,"Sx",q) );
					Yj.push_back(  2 * op(sites,"Sy",q) );
					Zj.push_back(  2 * op(sites,"Sz",q) );
				}

				

				if( js == N + 1 )
				{
					hx_vec = {hx, hx/2. , hx/3.};
					hy_vec = {hy, hy/2. , hy/3.};
					hz_vec = {hz, hz/2. , hz/3.};
				}
				else if( js == N+2 )
				{
					hx_vec = {hx/2. , hx/3. , hx/3.};
					hy_vec = {hy/2. , hy/3. , hy/3.};
					hz_vec = {hz/2. , hz/3. , hz/3.};
				}
				else if(jf == N2 - 1)
				{
					hx_vec = {hx/3. , hx/3. , hx/2.};
					hy_vec = {hy/3. , hy/3. , hy/2.};
					hz_vec = {hz/3. , hz/3. , hz/2.};
				}
				else if(jf == N2)
				{
					hx_vec = {hx/3. , hx/2. , hx};
					hy_vec = {hy/3. , hy/2. , hy};
					hz_vec = {hz/3. , hz/2. , hz};
				}
				else
				{
					hx_vec = {hx/3. , hx/3. , hx/3.};
					hy_vec = {hy/3. , hy/3. , hy/3.};
					hz_vec = {hz/3. , hz/3. , hz/3.};
				}

				if( js == N+1) // First physical site (it technically correspond to the last site of the bra, but since couplings are homogeneous it does not matter)
				{
					Jxx_vec[0] = Jxx;
					Jyy_vec[0] = Jyy;
					Jzz_vec[0] = Jzz;
				}

				if( jf == N2) // Last physical site
				{
					Jxx_vec[1] = Jxx;
					Jyy_vec[1] = Jyy;
					Jzz_vec[1] = Jzz;
				}

				for(double hj : hx_vec) cerr << hj << " ";
				cerr << "\n";
				
				hx_tot[j%N-1]   += hx_vec[0];
				hx_tot[j%N] += hx_vec[1];
				hx_tot[j%N+1] += hx_vec[2];

				// Jzz_tot[j%N-1] += Jzz_vec[0];
				// Jzz_tot[j%N]   += Jzz_vec[1];
						
				
				

				H_S   = hx_vec[0] * Xj[0] * Ij[1] * Ij[2];
				H_S  += hx_vec[1] * Ij[0] * Xj[1] * Ij[2];
				H_S  += hx_vec[2] * Ij[0] * Ij[1] * Xj[2];

				H_S  += hy_vec[0] * Yj[0] * Ij[1] * Ij[2];
				H_S  += hy_vec[1] * Ij[0] * Yj[1] * Ij[2];
				H_S  += hy_vec[2] * Ij[0] * Ij[1] * Yj[2];

				H_S  += hz_vec[0] * Zj[0] * Ij[1] * Ij[2];
				H_S  += hz_vec[1] * Ij[0] * Zj[1] * Ij[2];
				H_S  += hz_vec[2] * Ij[0] * Ij[1] * Zj[2];

				H_SS   = Jxx_vec[0] * Xj[0] * Xj[1] * Ij[2] ;
				H_SS  += Jxx_vec[1] * Ij[0] * Xj[1] * Xj[2] ;

				H_SS  += Jyy_vec[0] * Yj[0] * Yj[1] * Ij[2] ;
				H_SS  += Jyy_vec[1] * Ij[0] * Yj[1] * Yj[2] ;

				H_SS  += Jzz_vec[0] * Zj[0] * Zj[1] * Ij[2] ;
				H_SS  += Jzz_vec[1] * Ij[0] * Zj[1] * Zj[2] ;

				H_SSS  = J_NNN[0] * Xj[0] * Ij[1] * Xj[2] ;
				H_SSS += J_NNN[1] * Yj[0] * Ij[1] * Yj[2] ;
				H_SSS += J_NNN[2] * Zj[0] * Ij[1] * Zj[2] ;
				
		
				vector<int> jn = {js,js+1,js+2};
				if(j<N)
				{
					// action on bra (it is exp(+i H^T t))
					H = -H_S - H_SS - H_SSS;
					H = swapPrime(H,0,1);

					MyBondGate g = MyBondGate(sites,jn,dt/2.,H);
					gates_bra.push_back(g);
				}
				// action on ket
				else
				{
					H = H_S + H_SS + H_SSS;
					MyBondGate g = MyBondGate(sites,jn,dt/2.,H);
					gates_ket.push_back(g);
				}
			}
				
		}
	}

	// total magnetic field on each site

	for(double q : hx_tot) cerr << q << " ";
	cerr << "\n";
	for(double q : Jzz_tot) cerr << q << " ";
	cerr << "\n";
	// exit(0);

	vector<MyBondGate> gates_ket_reversed = gates_ket;
	vector<MyBondGate> gates_bra_reversed = gates_bra;
	
	reverse(gates_ket_reversed.begin(), gates_ket_reversed.end());
	reverse(gates_bra_reversed.begin(), gates_bra_reversed.end());

	for(MyBondGate g : gates_bra)  		   gates.push_back(g);
	for(MyBondGate g : gates_ket) 		   gates.push_back(g);
	for(MyBondGate g : gates_ket_reversed) gates.push_back(g);
	for(MyBondGate g : gates_bra_reversed) gates.push_back(g);

	return gates;
}


// Gates of impurity problem (bath treated exactly in energy basis)
// this results in a hihgly non local Hamiltonian. Specifically
// H = 
// we do not implement the swap gates as gates, but directly in the TEBD algorithm.

// We move from the center of the chain (where the bond connecting bra and ket is located)
// moving towards the outer part

// we feed in input J_eps, hup, hdn in the right order (acting on sites from 1 to N) in the TRUE system
// wee perform first evolution of the bra [1,N]
// then, we perform the evolution of the ket [N+1,2*N]
// we start from the center, so that the first interaction is nearest-neighbor and then
// we should apply swapgates

vector<BondGate>
gates_coherent_unfolded_kondo_impurity_model_energy_basis(const SiteSet sites , const vector<double> J, const vector<double> hup, const vector<double> hdn, const double dt)
{
	// N is always even and corresponds to the physical size
	int N2 = length(sites);
	int N = N2/2;


    vector<BondGate> gates;
	vector<BondGate> gates_ket;

	// collection of gates

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

	
	for(int q=1 ; q<=N2; q++)
	{
		Adagup.push_back(op(sites,"Adagup",q) );
		Aup.push_back(   op(sites,"Aup",q) );
		Adagdn.push_back(op(sites,"Adagdn",q) );
		Adn.push_back(   op(sites,"Adn",q) );
		Nup.push_back(   op(sites,"Nup",q) );
		Ndn.push_back(   op(sites,"Ndn",q) );
		Id.push_back(op(sites,"Id",q));
		AdagupFi.push_back(op(sites,"Adagup*F",q));
		AupFi.push_back(op(sites,"Aup*F",q));
		FiAdn.push_back(op(sites,"F*Adn",q));
		FiAdagdn.push_back(op(sites,"F*Adagdn",q));

	}
	
	cerr << "Couplings.\n\n";

	for(double z : J) cerr << z << " ";
	cerr << "\n\n";

	// Action of the Hamiltonian on the bra
	// we start from the center of the chain

	cerr << "Entering bra cycle.\n\n";
	for(int j=N-1 ; j>=1 ; j--)
	{

		ITensor H , H_S , H_SS;

		// build single site - the impurity is set only at the first bond
		if(j==N-1)
		{
			H_S = (hup[N-j-1] * Nup[N-1] + hdn[N-j-1] * Ndn[N-1]) * Id[N-2] ;
			H_S += Id[N-1] * (hup[N-j] * Nup[j-1] + hdn[N-j] * Ndn[j-1]);
		}
		else  H_S = Id[N-1] * (hup[N-j] * Nup[j-1] + hdn[N-j] * Ndn[j-1]);
		
		

		// cerr << "Site : " << j     << " -> h : " << hup[j-1] << "\n";
		// cerr << "Site : " << j + 1 << " -> h : " << hup[j] << "\n";

		// two sites hopping
		
		H_SS   = J[N-j-1] * (  Aup[j-1] * AdagupFi[N-1] - Adagup[j-1]   * AupFi[N-1] );
		H_SS  += J[N-j-1] * (  FiAdn[j-1] * Adagdn[N-1] - FiAdagdn[j-1] * Adn[N-1]   );
		
		cerr << J[N-j-1] << " ";

		
		H = - H_S - H_SS ;
		H = swapPrime(H,0,1); 
		BondGate g = BondGate(sites,j,N,BondGate::tReal,dt/2.,H); 
		gates.push_back(g);
			
	
    }
	cerr << "\n\n";

	vector<BondGate> gates_reversed = gates;
	reverse(gates_reversed.begin(), gates_reversed.end());
	for(BondGate g : gates_reversed)  gates.push_back(g);


	cerr << "Entering ket cycle.\n\n";

	for(int j=N+2 ; j<=N2 ; j++)
	{

		ITensor H , H_S , H_SS;

		// build single site - the impurity is set only at the first bond
		if(j==N+2)
		{
			H_S =  (hup[abs(N-j)-1] * Nup[N] + hdn[abs(N-j)-1] * Ndn[N]) * Id[N+1] ;
			H_S += Id[N] * (hup[abs(N-j)] * Nup[j-1] + hdn[abs(N-j)] * Ndn[j-1]);
		}
		else H_S = Id[N] * (hup[abs(N-j)] * Nup[j-1] + hdn[abs(N-j)] * Ndn[j-1]);
		

		// cerr << "Site : " << j     << " -> h : " << hup[j-1] << "\n";
		// cerr << "Site : " << j + 1 << " -> h : " << hup[j] << "\n";

		// two sites hopping
		
		H_SS   = J[abs(N-j)-2] * (  AdagupFi[N] * Aup[j-1]   - AupFi[N] * Adagup[j-1] );
		H_SS  += J[abs(N-j)-2] * (  Adagdn[N]   * FiAdn[j-1] - Adn[N] * FiAdagdn[j-1] );
		
		cerr << J[abs(N-j)-2] << " ";

		H = H_S + H_SS ;
		BondGate g = BondGate(sites,N+1,j,BondGate::tReal,dt/2.,H); 
		gates_ket.push_back(g);
			
	
    }

	gates_reversed = gates_ket;
	reverse(gates_reversed.begin(), gates_reversed.end());
	for(BondGate g : gates_reversed)  gates_ket.push_back(g);
	for(BondGate g : gates_ket) gates.push_back(g);

	// cerr << "\n\n";

	// cerr << "Returning gates\n";
	
	// for(BondGate g : gates) cerr << g.i1() << " ";
	// cerr << "\n\n";

	// for(BondGate g : gates) cerr << g.i2() << " ";

	// exit(0);

	return gates;

}


// Gates of impurity problem (bath treated exactly in energy basis)
// this results in a hihgly non local Hamiltonian. Specifically
// H = 
// we do not implement the swap gates as gates, but directly in the TEBD algorithm.

// We move from the center of the chain (where the bond connecting bra and ket is located)
// moving towards the outer part

// we feed in input J_eps, hup, hdn in the right order (acting on sites from 1 to N) in the TRUE system
// wee perform first evolution of the bra [1,N]
// then, we perform the evolution of the ket [N+1,2*N]
// we start from the center, so that the first interaction is nearest-neighbor and then
// we should apply swapgates

vector<MyBondGate>
doubling_space_gates(const vector<BondGate> gates_single, const SiteSet sites_single,  const SiteSet sites_doubled)
{

    vector<MyBondGate> gates_doubled;
	int N =  length(sites_single);
	cerr << N << "\n";
	cerr << "Entering ket\n";
	// ket dynamics
    for( BondGate g : gates_single)
    {
		cerr << "Here.\n";
        // physical space position
        int i1 = g.i1();
        int i2 = g.i2();
		cerr << i1 << "\n";
		cerr << i2 << "\n";
        // position along the doubled space
        int inew_1 = N + 1 + i1;
        int inew_2 = N + 1 + i2;
        
        ITensor gate 	= g.gate();
        Index si1 		= sites_single(i1);
        Index si2 		= sites_single(i2);
        Index sinew1 	= sites_doubled(inew_1);
        Index sinew2 	= sites_doubled(inew_2); 

        // changed sites
        gate *= delta(si1,sinew1);
        gate *= delta(si2,sinew2);
        gate *= delta(prime(si1),prime(sinew1));
        gate *= delta(prime(si2),prime(sinew2));
        
        MyBondGate gnew = MyBondGate(sites_doubled,{inew_1,inew_2},0,gate);
        gnew.modify_gate(gate);
        gates_doubled.push_back(gnew);
		
    }

	cerr << "Entering bra\n";

    // bra dynamics
    for( BondGate g : gates_single)
    {
        // physical space position
        int i1 = g.i1();
        int i2 = g.i2();
		cerr << i1 << "\n";

        // position along the doubled space
        int inew_1 = N + 2 - i1;
        int inew_2 = N + 2 - i2;
        
        ITensor gate 	= g.gate();
        Index si1 		= sites_single(i1);
        Index si2 		= sites_single(i2);
        Index sinew1 	= sites_doubled(inew_1);
		Index sinew2 	= sites_doubled(inew_2); 

        // changed sites
        gate *= delta(si1,sinew1);
        gate *= delta(si2,sinew2);
        gate *= delta(prime(si1),prime(sinew1));
        gate *= delta(prime(si2),prime(sinew2));
        // gate = swapPrime(gate,0,1); // NOT SURE

        MyBondGate gnew = MyBondGate(sites_doubled,{inew_1,inew_2},0,gate);
        gnew.modify_gate(dag(gate));
        gates_doubled.push_back(gnew);
    }

	return gates_doubled;

}