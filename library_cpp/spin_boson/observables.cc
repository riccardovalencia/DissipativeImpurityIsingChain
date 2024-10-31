#include "observables.h"
#include <itensor/all.h>
#include <math.h>       
#include <iostream>

// ----------------------------------------------------------
// Given a mized spin-boson or spin-1/2 system it measures:
// - occupation number for bosons
// - magnetization along direction (x,y,z) for spin-1/2 
 
vector<double>
measure_magnetization(MPS* psi, const SiteSet sites , string direction)
{

    int N = length(sites);
    vector<double> mj;

    for(int j=1 ; j<=N ; j++)
    {
        Index sj = sites(j);
        Index sjp = prime(sites(j));

        ITensor S_j  = ITensor(sj ,sjp );
		
	
		if(hasTags(sj,"Site,Boson"))
		{
			for(int d=1; d <= dim(sj) ; d++) S_j.set(sj(d),sjp(d),d-1.);
        }
				
		if(hasTags(sj,"Site,S=1/2"))
		{
            if (direction == "x")
            {
                S_j.set(sj(1),sjp(2),1.);
			    S_j.set(sj(2),sjp(1),1.);	
                // S_j = 2*op(sites,"Sx",j);
                
            }
            else if(direction == "y")
            {
                S_j.set(sj(1),sjp(2),-1*Cplx_i);
			    S_j.set(sj(2),sjp(1), 1*Cplx_i);
                // S_j = 2*op(sites,"Sy",j);

            }
            else if(direction == "z")
            {
                S_j.set(sj(1),sjp(1),1.);
                S_j.set(sj(2),sjp(2),-1.);
                // S_j = 2*op(sites,"Sz",j);
            }
            else
            {
                cerr << "Direction choses is neither 'x' , 'y' or 'z'" << endl;
                return mj;
            }
		}
        (*psi).position(j);
        ITensor ket = (*psi)(j);
		ITensor bra = dag(prime((*psi)(j),"Site"));
		
		complex<double> exp_Sj = eltC(bra * S_j * ket);
        // cerr << "Magnetization site : " << j << " : " << real(exp_Sj) << "\n";
		mj.push_back(real(exp_Sj));
        
    }

    // for(double m : mj) cerr << " " << m ;
    // cerr << "\n\n";
    // exit(0);
    return mj;
}

// ----------------------------------------------------------
// Measure single-site magnetization along direction (x,y,z) on site q on a density matrix unfolded as an MPS.
// The structure of the unfolded density matrix is 
// | | | | |
// o-o-o-o-o-   (ket)
// |
// o-o-o-o-o-   (bra)
// | | | | |
// namely the first [1,N] sites describe the bra, and [N+1,2*N] the ket
// To compute Tr(rho O), it is necessary to contract symmetrically wrt the center of the chain the bra and ket, 
// inserting where necessary, the operator O

// q is the site on the physical site (not the one along the unfolded density matrix) on which I want to measure.
// -> q gets suitable transformed into the site in the MPS formalism.

vector<complex<double> >
measure_magnetization_impurity_first_site(MPS *psi , string direction, bool compute_normalization, int q)
{
    IndexSet sites = siteInds((*psi));
    int N = length(sites)/2;
    vector<complex<double> > mj;

    Index s_ket_j, s_ket_k;
    Index s_bra_j, s_bra_k;

    // compute normalization

    ITensor M;
    double norm = 1.;
    if(compute_normalization)
    {
        norm = compute_norm_purifed_impurity(psi);
    }
    
    if(q!=-1)
    {
        int jbra = N-q+1; // Ncorresponding site on bra part of the MPS
        int jket = N+q;
        s_ket_j = sites(jket);
        s_bra_j = sites(jbra);

        // Observable to measure
        ITensor S_j  = ITensor(s_ket_j ,s_bra_j );

        if(hasTags(s_ket_j,"Site,S=1/2"))
		{
            if (direction == "x")
            {
                S_j.set(s_ket_j(1),s_bra_j(2),1.);
			    S_j.set(s_ket_j(2),s_bra_j(1),1.);	
            }
            else if(direction == "y")
            {
                S_j.set(s_ket_j(1),s_bra_j(2),-1*Cplx_i);
			    S_j.set(s_ket_j(2),s_bra_j(1), 1*Cplx_i);
            }
            else if(direction == "z")
            {
                S_j.set(s_ket_j(1),s_bra_j(1),1.);
                S_j.set(s_ket_j(2),s_bra_j(2),-1.);
            }
            else
            {
                cerr << "Direction choses is neither 'x' , 'y' or 'z'" << endl;
                return mj;
            }
		}

        // until I reach the site where I measure
        for(int k : range1(jbra-1))
        {
            int k_bra = k; // Ncorresponding site on bra part of the MPS
            int k_ket = 2*N-k+1;
            s_ket_k = sites(k_bra);
            s_bra_k = sites(k_ket);
            
            if(k==1) M  = (*psi)(k_bra) * delta(s_ket_k,s_bra_k) * (*psi)(k_ket);
            else     
            {
                M *= (*psi)(k_bra);
                M *= (*psi)(k_ket);
                M *= delta(s_ket_k,s_bra_k);
            }
        }

        if( jbra>=1)
        {
            M *= (*psi)(jbra); 
            M *= (*psi)(jket);
            M *= S_j;
        }
        else
        {
            M  = (*psi)(jbra);
            M *= (*psi)(jket);
            M *= S_j;
        }

        for(int k : range1(jbra+1,N))
        {
            int k_bra = k; // Ncorresponding site on bra part of the MPS
            int k_ket = 2*N-k+1;
            s_ket_k = sites(k_bra);
            s_bra_k = sites(k_ket);
            M *= (*psi)(k_bra);
            M *= (*psi)(k_ket);
            M *= delta(s_ket_k,s_bra_k);

        }

		complex<double> oj = eltC(M);
		mj.push_back(oj/norm);
        return mj;
    }

    for(int j : range1(N))
    {
        s_ket_j = sites(j);
        s_bra_j = sites(2*N-j+1);

        // Observable to measure
        ITensor S_j  = ITensor(s_ket_j ,s_bra_j );


        if(hasTags(s_ket_j,"Site,S=1/2"))
		{
            if (direction == "x")
            {
                S_j.set(s_ket_j(1),s_bra_j(2),1.);
			    S_j.set(s_ket_j(2),s_bra_j(1),1.);	
            }
            else if(direction == "y")
            {
                S_j.set(s_ket_j(1),s_bra_j(2),-1*Cplx_i);
			    S_j.set(s_ket_j(2),s_bra_j(1), 1*Cplx_i);
            }
            else if(direction == "z")
            {
                S_j.set(s_ket_j(1),s_bra_j(1),1.);
                S_j.set(s_ket_j(2),s_bra_j(2),-1.);
            }
            else
            {
                cerr << "Direction choses is neither 'x' , 'y' or 'z'" << endl;
                return mj;
            }
		}


        // until I reach the site where I measure
        for(int k : range1(j-1))
        {
            s_ket_k = sites(k);
            s_bra_k = sites(2*N-k+1);
            
            if(k==1) M  = (*psi)(k) * delta(s_ket_k,s_bra_k) * (*psi)(2*N-k+1);
            else     
            {
                M *= (*psi)(k);
                M *= (*psi)(2*N-k+1);
                M *= delta(s_ket_k,s_bra_k);
            }
        }

        if(j>1)
        {
            M *= (*psi)(j);
            M *= (*psi)(2*N-j+1);
            M *= S_j;
        }
        else
        {
            M = (*psi)(j);
            M *= (*psi)(2*N-j+1);
            M *= S_j;
        }

        for(int k : range1(j+1,N))
        {
            s_ket_k = sites(k);
            s_bra_k = sites(2*N-k+1);
            M *= (*psi)(k);
            M *= (*psi)(2*N-k+1);
            M *= delta(s_ket_k,s_bra_k);

        }

   
		complex<double> oj = eltC(M);
		mj.push_back(oj/norm);

    }

    reverse(mj.begin(), mj.end());
    return mj;
}


// ----------------------------------------------------------
// Measure single-site operator O on site q on a density matrix unfolded as an MPS.
// The structure of the unfolded density matrix is 
// | | | | |
// o-o-o-o-o-   (ket)
// |
// o-o-o-o-o-   (bra)
// | | | | |
// namely the first [1,N] sites describe the bra, and [N+1,2*N] the ket
// To compute Tr(rho O), it is necessary to contract symmetrically wrt the center of the chain the bra and ket, 
// inserting where necessary, the operator O

// q is the site on the physical site (not the one along the unfolded density matrix) on which I want to measure.
// -> q gets suitable transformed into the site in the MPS formalism.


vector<complex<double> >
measure_local_obs_impurity_first_site(MPS *psi , const ITensor O, bool compute_normalization, int q)
{
    IndexSet sites = siteInds((*psi));
    int N = int(length(sites)/2);
    vector<complex<double> > mj;
    Index s_ket_j, s_ket_k;
    Index s_bra_j, s_bra_k;

    // compute normalization

    double norm = 1.;
    if(compute_normalization)  norm = compute_norm_purifed_impurity(psi);
    
    ITensor M;

    // getting indices of ITensor 
    IndexSet inds_O = inds(O);

    Index col = noPrime(inds_O[0]);
    Index row = prime(col);

    if(q!=-1)
    {
        int jbra = N-q+1; // Ncorresponding site on bra part of the MPS
        int jket = N+q;
        s_ket_j = sites(jket);
        s_bra_j = sites(jbra);

        // Observable to measure (convert the indices so that it can be suitably contracted)        
        ITensor O_q = O;
        if( s_ket_j != col) O_q *= delta(s_ket_j,col);
        if( s_bra_j != row) O_q *= delta(s_bra_j,row);


        // until I reach the site where I measure
        for(int k : range1(jbra-1))
        {
            int k_bra = k; // Ncorresponding site on bra part of the MPS
            int k_ket = 2*N-k+1;
            s_ket_k = sites(k_bra);
            s_bra_k = sites(k_ket);
            
            if(k==1) M  = (*psi)(k_bra) * delta(s_ket_k,s_bra_k) * (*psi)(k_ket);
            else     
            {
                M *= (*psi)(k_bra);
                M *= (*psi)(k_ket);
                M *= delta(s_ket_k,s_bra_k);
            }
        }

        if( jbra>1)
        {
            M *= (*psi)(jbra); 
            M *= (*psi)(jket);
            M *= O_q;
        }
        else
        {
            M  = (*psi)(jbra);
            M *= (*psi)(jket);
            M *= O_q;
        }

        for(int k : range1(jbra+1,N))
        {
            int k_bra = k; // Ncorresponding site on bra part of the MPS
            int k_ket = 2*N-k+1;
            s_ket_k = sites(k_bra);
            s_bra_k = sites(k_ket);
            M *= (*psi)(k_bra);
            M *= (*psi)(k_ket);
            M *= delta(s_ket_k,s_bra_k);

        }

		complex<double> oj = eltC(M);

		mj.push_back(oj/norm);
        return mj;
    }

    for(int j : range1(N))
    {
        s_ket_j = sites(j);
        s_bra_j = sites(2*N-j+1);

        // Observable to measure

        ITensor O_j = O;
        if( s_ket_j != col) O_j *= delta(s_ket_j,col);
        if( s_bra_j != row) O_j *= delta(s_bra_j,row);

        // I was using s_bra_j - col, but I think it is wrong.
        // ITensor O_j  =  delta(s_bra_j,col) * O *  delta(s_ket_j,row);

        // until I reach the site where I measure
        for(int k : range1(j-1))
        {
            s_ket_k = sites(k);
            s_bra_k = sites(2*N-k+1);
            
            if(k==1) M  = (*psi)(k) * delta(s_ket_k,s_bra_k) * (*psi)(2*N-k+1);
            else     
            {
                M *= (*psi)(k);
                M *= (*psi)(2*N-k+1);
                M *= delta(s_ket_k,s_bra_k);
            }
        }

        if(j>1)
        {
            M *= (*psi)(j);
            M *= (*psi)(2*N-j+1);
            M *= O_j;
        }
        else
        {
            M = (*psi)(j);
            M *= (*psi)(2*N-j+1);
            M *= O_j;
        }

        for(int k : range1(j+1,N))
        {
            s_ket_k = sites(k);
            s_bra_k = sites(2*N-k+1);
            M *= (*psi)(k);
            M *= (*psi)(2*N-k+1);
            M *= delta(s_ket_k,s_bra_k);

        }

   
		complex<double> oj = eltC(M);
		mj.push_back(oj/norm);

    }

    reverse(mj.begin(), mj.end());
    return mj;
}


// ----------------------------------------------------------
// Compute Tr(rho) where rho is unfolded as an MPS.
double
compute_norm_purifed_impurity(MPS (*psi))
{
    IndexSet sites = siteInds((*psi));
    int N = length(sites)/2;

    Index s_ket_j, s_ket_k;
    Index s_bra_j, s_bra_k;

    ITensor M;
    for(int j : range1(N))
    {
        s_ket_j = sites(j);
        s_bra_j = sites(2*N-j+1);
        if(j==1) M  = (*psi)(j) * delta(s_ket_j,s_bra_j) * (*psi)(2*N-j+1);
        else     
        {
            // cerr << "Site " << j << "\n";
            M *= (*psi)(j);
            M *= (*psi)(2*N-j+1);
            M *= delta(s_ket_j,s_bra_j);
        }
    }

    double norm = real(eltC(M));

    return norm;

}

// ----------------------------------------------------------
// Compute Tr(rho) where rho is unfolded as an MPS with quantum numbers QN

double
compute_norm_purifed_impurity_QN(MPS (*psi))
{
    IndexSet sites = siteInds((*psi));
    int N = length(sites)/2;

    Index s_ket_j, s_ket_k;
    Index s_bra_j, s_bra_k;

    ITensor M;
    for(int j : range1(N))
    {
        s_ket_j = sites(j);
        s_bra_j = sites(2*N-j+1);
        if(j==1) M  = (*psi)(j) * delta(s_ket_j,s_bra_j) * (*psi)(2*N-j+1);
        else     
        {
            // cerr << "Site " << j << "\n";
            M *= (*psi)(j);
            M *= (*psi)(2*N-j+1);
            M *= delta(s_ket_j,s_bra_j);
        }
    }

    double norm = real(eltC(M));

    return norm;

}

// ----------------------------------------------------------
// Compute entanglement entropy along the bond [site,site+1]

double
entanglement_entropy( MPS* psi , int site)
	{
	(*psi).position(site); 
	ITensor wf = (*psi)(site) * (*psi)(site+1);
	ITensor U  = (*psi)(site);
	ITensor S,V;
	auto spectrum = svd(wf,U,S,V);
	
	double SvN = 0.;
	for(auto p : spectrum.eigs())
		{
		if(p > 1E-12) SvN += -p*log2(p);
		}
	return SvN;
	}


// ----------------------------------------------------------
// Compute number of kinks (|\up_z \dw_z>) on a state psi

double 
measure_kink( MPS* psi, const SiteSet sites)
{
    int N = length(sites);

    double kink = 0.;

    if(N==1)
    {
        cerr << "Cannot measure number of kinks in a single-site system.\n";
        cerr << "Returnin 0.\n";
        return kink;
    }

    for(int j=1 ; j<N ; j++)
    {
        (*psi).position(j);
        
        ITensor N_1 = (op(sites,"Id",j)   - 2*op(sites,"Sz",j))    /2.;
        ITensor N_2 = (op(sites,"Id",j+1) + 2*op(sites,"Sz",j+1))  /2.;
        
        ITensor ket = (*psi)(j)*(*psi)(j+1);
		ITensor bra = dag(prime((*psi)(j),"Site"))*dag(prime((*psi)(j+1),"Site"));
		
        complex<double> n_j = eltC(bra * N_1 * N_2 * ket);
        kink += n_j.real();
    }

    return kink;
}


// ----------------------------------------------------------
// Compute correlation functions <N_start N_(start+i)> (both connected and disconnected).
// Where N = (1-2*S^z)/2 = |down_z> <down_z|

vector<double>
measure_correlations(MPS* psi, const SiteSet sites, const int start, const bool connected)
{
    int N = length(sites);
    vector<double> C;

    // reference site
    ITensor Ns = (op(sites,"Id",start) - 2*op(sites,"Sz",start))  /2.;

    vector<double> nj;
    if(connected)
    {
        vector<double> mz = measure_magnetization( psi,sites,"z");
        for(double m : mz) nj.push_back((1-m)/2.);
    }

    for(int j=1 ; j<= N; j++)
    {
        double Cjs;
        ITensor M;
        
        ITensor N1, N2;
        int jmax = max(j,start);
        int jmin = min(j,start);

        if(jmin == j)
        {
            N1 = (op(sites,"Id",j) - 2*op(sites,"Sz",j))  /2.;
            N2 = Ns;
        }
        else
        {
            N1 = Ns;
            N2 = (op(sites,"Id",j) - 2*op(sites,"Sz",j))  /2.;
        }

        (*psi).position(jmin);
        ITensor ket = (*psi)(jmin);


        if(j==start)
        {
		    ITensor bra = dag(prime((*psi)(j),"Site"));
            Cjs = eltC(ket*N1*bra).real(); //nb N^2 = N (it is a projector)
        }


        else
        {    
            Index ir = commonIndex( (*psi)(jmin) , (*psi)(jmin + 1) ,"Link");
			M = ket * N1 * dag( prime( prime( ket , "Site") , ir ) );

            for(int q = jmin + 1 ; q < jmax ; q++)
            {
                M *= (*psi)(q);
                M *= dag(prime( (*psi)(q) , "Link"));
            }

            Index il = commonIndex( (*psi)( jmax-1 ), (*psi)(jmax), "Link");
            M *= (*psi)(jmax);
            M *= N2;
            M *= dag( prime( prime((*psi)( jmax ), il) , "Site") );
            Cjs = eltC(M).real();
        }
			
		
        if(connected)
        {
            Cjs = Cjs - nj[jmin-1] * nj[jmax-1];
        }

        C.push_back(Cjs);

    }

    return C;
}

// ----------------------------------------------------------
// Compute correlation functions <Q_q1 O_q2> (both connected and disconnected) of a generic one-site observable O.
// The input MPS represents a density matrix unfolded as an MPS.
// The structure of the unfolded density matrix is 
// | | | | |
// o-o-o-o-o-   (ket)
// |
// o-o-o-o-o-   (bra)
// | | | | |
// namely the first [1,N] sites describe the bra, and [N+1,2*N] the ket

// q1 and q2 are the sites on the physical site (not the one along the unfolded density matrix) on which I want to measure.
// -> q1 and q2 get suitable transformed into the site in the MPS formalism.

complex<double>
measure_correlation_impurity_first_site(MPS *psi , const ITensor O, bool compute_normalization, int q1, int q2, bool connected)
{
    IndexSet sites = siteInds((*psi));
    int N = int(length(sites)/2);
    complex<double> Oq1q2;
    Index s_ket_q1, s_ket_q2, s_ket_k;
    Index s_bra_q1, s_bra_q2, s_bra_k;

    // swapping so that q2>q1 always
    if(q1 > q2)
    {
        int q = q1;
        q1 = q2;
        q2 = q;
    }
    else if(q1==q2)
    {
        cerr << "Measuring autocorrelation function. Not implemented.\n";
        return Oq1q2;
    }
    else if(q1 < 0 && q2 < 0 && q1 > N && q2 > N)
    {
        cerr << "Measuring correlation function on unphysical sites.\n";
        return Oq1q2;
    }

    // compute normalization
    double norm = 1.;
    if(compute_normalization)  norm = compute_norm_purifed_impurity(psi);
    
    ITensor M;

    // getting indices of ITensor 
    IndexSet inds_O = inds(O);

    Index col = noPrime(inds_O[0]);
    Index row = prime(col);

    // measuring correlation function

    // corresponding site on bra part of the MPS
    int q1bra = N-q1+1;
    int q1ket = N+q1;
    // corresponding site on bra part of the MPS
    int q2bra = N-q2+1;
    int q2ket = N+q2;

     
    s_ket_q1 = sites(q1ket);
    s_bra_q1 = sites(q1bra);

    s_ket_q2 = sites(q2ket);
    s_bra_q2 = sites(q2bra);

    // Observable to measure (convert the indices so that it can be suitably contracted)
    // ITensor O_q  =  delta(s_bra_j,row) * O *  delta(s_ket_j,col);
    
    ITensor O_q1 = O;
    if( s_ket_q1 != col) O_q1 *= delta(s_ket_q1,col);
    if( s_bra_q1 != row) O_q1 *= delta(s_bra_q1,row);

    ITensor O_q2 = O;
    if( s_ket_q2 != col) O_q2 *= delta(s_ket_q2,col);
    if( s_bra_q2 != row) O_q2 *= delta(s_bra_q2,row);

    // until I reach the site where I measure
    for(int k : range1(q2bra-1))
    {
        int k_bra = k; // Ncorresponding site on bra part of the MPS
        int k_ket = 2*N-k+1;
        s_ket_k = sites(k_bra);
        s_bra_k = sites(k_ket);
        
        if(k==1) M  = (*psi)(k_bra) * delta(s_ket_k,s_bra_k) * (*psi)(k_ket);
        else     
        {
            M *= (*psi)(k_bra);
            M *= (*psi)(k_ket);
            M *= delta(s_ket_k,s_bra_k);
        }
    }

    if( q2bra>1)
    {
        M *= (*psi)(q2bra); 
        M *= (*psi)(q2ket);
        M *= O_q2;
    }
    else
    {
        M  = (*psi)(q2bra);
        M *= (*psi)(q2ket);
        M *= O_q2;
    }

    for(int k : range1(q2bra+1,q1bra-1))
    {
        int k_bra = k; // Ncorresponding site on bra part of the MPS
        int k_ket = 2*N-k+1;
        s_ket_k = sites(k_bra);
        s_bra_k = sites(k_ket);
        M *= (*psi)(k_bra);
        M *= (*psi)(k_ket);
        M *= delta(s_ket_k,s_bra_k);

    }

    M *= (*psi)(q1bra); 
    M *= (*psi)(q1ket);
    M *= O_q1;

    for(int k : range1(q1bra+1,N))
    {
        int k_bra = k; // Ncorresponding site on bra part of the MPS
        int k_ket = 2*N-k+1;
        s_ket_k = sites(k_bra);
        s_bra_k = sites(k_ket);
        M *= (*psi)(k_bra);
        M *= (*psi)(k_ket);
        M *= delta(s_ket_k,s_bra_k);

    }

    Oq1q2 = eltC(M)/norm;

    if(connected)
    {  
        cerr<< "Measuring connected\n";
        vector<complex<double> > Oq1 = measure_local_obs_impurity_first_site( psi , O, compute_normalization, q1);
        vector<complex<double> > Oq2 = measure_local_obs_impurity_first_site( psi , O, compute_normalization, q2);
        Oq1q2 = Oq1q2 - Oq1[0] * Oq2[0];
    }

    return Oq1q2;
   
}