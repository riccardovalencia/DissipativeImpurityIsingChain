#include "initial_state.h"
#include <itensor/all.h>
#include <math.h>       
#include <iostream>
#include <vector>


// ----------------------------------------------------------
// Given a mixed spin_boson SiteSet, initialize product state |n_photon > \otimes |\theta,\phi>^(N-1)
// where |n_photon> is a Fock state
//       |theta,\phi> is a spin-coherent state of a spin-1/2 pointing on the Bloch sphere 
// 
// TO DO : do a function ITensor spin_coherent_state(IndexSet,theta,phi) which returns a spin coherent state
//       : do a function ITensor fock_state(IndexSet,n) which return |n>

MPS
initialize_spin_boson_state(const SiteSet sites , const int n_photon , double theta, double phi)
{

    MPS psi = randomMPS(sites);
    int N   = length(sites);

    Index sj ,rj, lj;

    // first site
	sj = sites(1);
	rj = commonIndex(psi(1),psi(2));
	ITensor wf = ITensor(sj,rj);
	
    if(hasTags(sj,"Site,Boson"))
    {
        for( int d=1; d <= n_photon; d++) wf.set(sj(d),rj(1), 0);
        wf.set(sj(n_photon+1),rj(1),1);
        for( int d=n_photon+2; d <= dim(sj); d++) wf.set(sj(d),rj(1), 0);
    }
				
    else if(hasTags(sj,"Site,S=1/2"))
    {
        cerr << "Inserting spin coherent state" << endl;
        wf.set(sj(1),rj(1),cos(theta/2.));
        // wf.set(sj(2),lj(1),rj(1),sin(theta)*exp(i*phi));
        wf.set(sj(2),rj(1),sin(theta/2.));
    }
    
    else{
        cerr << "SiteSet not recognize : " << sj << endl;
        cerr << "Return a random initial state" << endl;
        return psi; 
    }

	psi.set(1,wf);
	
    cerr << "Inserted spin coherent state" << endl;

    for(int j=2 ; j < N ; j++)
    {
        sj = sites(j);
        lj = commonIndex(psi(j-1),psi(j));
		rj = commonIndex(psi(j),psi(j+1));
		wf = ITensor(sj,lj,rj);

        if(hasTags(sj,"Site,Boson"))
        {
            for( int d=1; d <= n_photon; d++) wf.set(sj(d),rj(1),lj(1), 0);
            wf.set(sj(n_photon+1),rj(1),lj(1),1);
            for( int d=n_photon+2; d <= dim(sj); d++) wf.set(sj(d),rj(1),lj(1), 0);
        }
                    
        else if(hasTags(sj,"Site,S=1/2"))
        {
            wf.set(sj(1),lj(1),rj(1),cos(theta/2.));
            // wf.set(sj(2),lj(1),rj(1),sin(theta)*exp(i*phi));
            wf.set(sj(2),lj(1),rj(1),sin(theta/2.));
        }

		psi.set(j,wf); 
    }

    sj = sites(N);
    lj = commonIndex(psi(N-1),psi(N));
    wf = ITensor(sj,lj);

    wf.set(sj(1),lj(1),cos(theta/2.));
    // wf.set(sj(2),lj(1),rj(1),sin(theta)*exp(i*phi));
    wf.set(sj(2),lj(1),sin(theta/2.));


    if(hasTags(sj,"Site,Boson"))
    {
        for( int d=1; d <= n_photon; d++) wf.set(sj(d),lj(1), 0);
        wf.set(sj(n_photon+1),lj(1),1);
        for( int d=n_photon+2; d <= dim(sj); d++) wf.set(sj(d),lj(1), 0);
    }
                    
    else if(hasTags(sj,"Site,S=1/2"))
    {
        wf.set(sj(1),lj(1),cos(theta/2.));
        // wf.set(sj(2),lj(1),rj(1),sin(theta)*exp(i*phi));
        wf.set(sj(2),lj(1),sin(theta/2.));
    }


    psi.set(N,wf); 

    return psi;
}


// ----------------------------------------------------------
// Override previous function - it can accept vectors as theta and phi
// Given a mixed spin_boson SiteSet, initialize product state |n_photon > \otimes |\theta,\phi>^(N-1)
// where |n_photon> is a Fock state
//       |theta,\phi> is a spin-coherent state of a spin-1/2 pointing on the Bloch sphere 
// 
// TO DO : do a function ITensor spin_coherent_state(IndexSet,theta,phi) which returns a spin coherent state
//       : do a function ITensor fock_state(IndexSet,n) which return |n>
MPS
initialize_spin_boson_state(const SiteSet sites , const int n_photon , const vector<double> theta, const vector<double> phi)
{

    MPS psi = randomMPS(sites);
    int N   = length(sites);

    if(N==1)
    {
        Index sj = sites(1);
        ITensor wf = ITensor(sj);
        
        if(hasTags(sj,"Site,Boson"))
        {
            for( int d=1; d <= n_photon; d++) wf.set(sj(d), 0);
            wf.set(sj(n_photon+1),1);
            for( int d=n_photon+2; d <= dim(sj); d++) wf.set(sj(d), 0);
        }
                    
        else if(hasTags(sj,"Site,S=1/2"))
        {
            cerr << "Inserting spin coherent state" << endl;
            wf.set(sj(1),cos(theta[0]/2.));
            // wf.set(sj(2),lj(1),rj(1),sin(theta)*exp(i*phi));
            wf.set(sj(2),sin(theta[0]/2.));
        }
        
        else{
            cerr << "SiteSet not recognize : " << sj << endl;
            cerr << "Return a random initial state" << endl;
            return psi; 
        }

        psi.set(1,wf);
    }

    if(N>1)
    {

        Index sj ,rj, lj;

        // first site
        sj = sites(1);
        rj = commonIndex(psi(1),psi(2));
        ITensor wf = ITensor(sj,rj);
        
        if(hasTags(sj,"Site,Boson"))
        {
            for( int d=1; d <= n_photon; d++) wf.set(sj(d),rj(1), 0);
            wf.set(sj(n_photon+1),rj(1),1);
            for( int d=n_photon+2; d <= dim(sj); d++) wf.set(sj(d),rj(1), 0);
        }
                    
        else if(hasTags(sj,"Site,S=1/2"))
        {
            cerr << "Inserting spin coherent state" << endl;
            wf.set(sj(1),rj(1),cos(theta[0]/2.));
            // wf.set(sj(2),lj(1),rj(1),sin(theta)*exp(i*phi));
            wf.set(sj(2),rj(1),sin(theta[0]/2.));
        }
        
        else{
            cerr << "SiteSet not recognize : " << sj << endl;
            cerr << "Return a random initial state" << endl;
            return psi; 
        }

        psi.set(1,wf);
        
        cerr << "Inserted spin coherent state" << endl;

        for(int j=2 ; j < N ; j++)
        {
            sj = sites(j);
            lj = commonIndex(psi(j-1),psi(j));
            rj = commonIndex(psi(j),psi(j+1));
            wf = ITensor(sj,lj,rj);

            if(hasTags(sj,"Site,Boson"))
            {
                for( int d=1; d <= n_photon; d++) wf.set(sj(d),rj(1),lj(1), 0);
                wf.set(sj(n_photon+1),rj(1),lj(1),1);
                for( int d=n_photon+2; d <= dim(sj); d++) wf.set(sj(d),rj(1),lj(1), 0);
            }
                        
            else if(hasTags(sj,"Site,S=1/2"))
            {
                wf.set(sj(1),lj(1),rj(1),cos(theta[j-1]/2.));
                // wf.set(sj(2),lj(1),rj(1),sin(theta)*exp(i*phi));
                wf.set(sj(2),lj(1),rj(1),sin(theta[j-1]/2.));
            }

            psi.set(j,wf); 
        }

        sj = sites(N);
        lj = commonIndex(psi(N-1),psi(N));
        wf = ITensor(sj,lj);

        // wf.set(sj(1),lj(1),cos(theta/2.));
        // wf.set(sj(2),lj(1),rj(1),sin(theta)*exp(i*phi));
        // wf.set(sj(2),lj(1),sin(theta/2.));


        if(hasTags(sj,"Site,Boson"))
        {
            for( int d=1; d <= n_photon; d++) wf.set(sj(d),lj(1), 0);
            wf.set(sj(n_photon+1),lj(1),1);
            for( int d=n_photon+2; d <= dim(sj); d++) wf.set(sj(d),lj(1), 0);
        }
                        
        else if(hasTags(sj,"Site,S=1/2"))
        {
            wf.set(sj(1),lj(1),cos(theta[N-1]/2.));
            // wf.set(sj(2),lj(1),rj(1),sin(theta)*exp(i*phi));
            wf.set(sj(2),lj(1),sin(theta[N-1]/2.));
        }


        psi.set(N,wf); 
    }

    return psi;
}


// ----------------------------------------------------------
// Initialize product states in the computation basis
// |psi> = |0/1> |0/1> ...
// where |0> = |up_z> and |1> = |down_z>
MPS
initial_computational_state(const SiteSet sites , const vector<int> initial_state)
{

    MPS psi = randomMPS(sites);
    int N   = length(sites);

    if(N != initial_state.size())
    {
        cerr << "Lenght of initial state is different from N!\nReturning random MPS.\n";
        return psi;
    }

    // for(int q : initial_state) cerr << q << " ";
    // exit(0);


    if(N==1)
    {
        Index sj = sites(1);
        ITensor wf = ITensor(sj);
        
        
        
        if(hasTags(sj,"Site,S=1/2"))
        {
            cerr << "Inserting spin coherent state" << endl;
            if (initial_state[0]==0)
            {
                wf.set(sj(1),1);
                wf.set(sj(2),0);
            }
            else
            {
                wf.set(sj(1),0);
                wf.set(sj(2),1);
            }
            
        }
        
        else{
            cerr << "SiteSet not recognize : " << sj << endl;
            cerr << "Return a random initial state" << endl;
            return psi; 
        }

        psi.set(1,wf);
    }

    if(N>1)
    {

        Index sj ,rj, lj;

        // first site
        sj = sites(1);
        rj = commonIndex(psi(1),psi(2));
        ITensor wf = ITensor(sj,rj);
        
        if(hasTags(sj,"Site,S=1/2"))
        {
            if (initial_state[0]==0)
            {
                wf.set(sj(1),rj(1),1);
                wf.set(sj(2),rj(1),0);
            }
            else
            {
                wf.set(sj(1),rj(1),0);
                wf.set(sj(2),rj(1),1);
            }

        }
        
        else{
            cerr << "SiteSet not recognize : " << sj << endl;
            cerr << "Return a random initial state" << endl;
            return psi; 
        }

        psi.set(1,wf);
        
        for(int j=2 ; j < N ; j++)
        {
            sj = sites(j);
            lj = commonIndex(psi(j-1),psi(j));
            rj = commonIndex(psi(j),psi(j+1));
            wf = ITensor(sj,lj,rj);

            if(hasTags(sj,"Site,S=1/2"))
            {
                if (initial_state[j-1]==0)
                {
                    cerr << "Inserted spin down\n";
                    wf.set(sj(1),lj(1),rj(1),1);
                    wf.set(sj(2),lj(1),rj(1),0);
                }
                else
                {
                    cerr << "Inserted spin up\n";
                    wf.set(sj(1),lj(1),rj(1),0);
                    wf.set(sj(2),lj(1),rj(1),1);
                }

            }
        
            else
            {
                cerr << "SiteSet not recognize : " << sj << endl;
                cerr << "Return a random initial state" << endl;
                return psi; 
            }
                
            psi.set(j,wf); 
        }

        sj = sites(N);
        lj = commonIndex(psi(N-1),psi(N));
        wf = ITensor(sj,lj);

                        
        if(hasTags(sj,"Site,S=1/2"))
        {
            if (initial_state[N-1]==0)
            {
                wf.set(sj(1),lj(1),1);
                wf.set(sj(2),lj(1),0);
            }
            else
            {
                wf.set(sj(1),lj(1),0);
                wf.set(sj(2),lj(1),1);
            }
        }

        psi.set(N,wf); 
    }

    return psi;
}


// Insert a state within another state, such that you have a state |state_to_insert> that you want to put in another state |psi_t0> from site start to start+L

void
insert_state(MPS* psi, MPS psi_seed, const int start, bool inverted,bool dagger)
{

    IndexSet phys_idx = siteInds(*psi);
    IndexSet phys_idx_seed = siteInds(psi_seed);

    IndexSet link_idx = linkInds(*psi);
    IndexSet link_idx_seed = linkInds(psi_seed);


    int N      = length(phys_idx);
    int N_seed = length(phys_idx_seed);

    Index lj ;
    Index rj ;
    Index sj ;
    ITensor Tj ;

    if(inverted)
    {
        MPS psi_seed_inv = psi_seed;
        int k = 1;
    
        for(int j= N_seed ; j >= 1; j--)
        {
            if(dagger) psi_seed_inv.set(k,dag(psi_seed(j)));
            else psi_seed_inv.set(k,psi_seed(j)); 
            k += 1;
            
        }

        psi_seed = psi_seed_inv;
        // test - no need to replace siteinds in the inverted one
        // psi_seed.replaceSiteInds(phys_idx_seed);
        phys_idx_seed = siteInds(psi_seed);
        link_idx_seed = linkInds(psi_seed);
    }

    // add tags to link indices, in order to avoid issues if we insert multiple copyes of the same state
    IndexSet new_links = IndexSet(N_seed-1);
    for(int j : range1(N_seed-1))
	{
		int j_psi = start + j - 1;
        Index new_idx = addTags(link_idx_seed(j),"seed="+str(j_psi));
        new_links[j-1] = new_idx;
    }

    psi_seed.replaceLinkInds(new_links);
    link_idx_seed = linkInds(psi_seed);


    // cerr << psi_seed(2) << endl;
    // cerr << link_idx_seed << endl;
    // double a ;
    // cin >> a;

	for(int j : range1(N_seed))
	{
		int j_psi = start + j - 1;
        

        if(j==1 && j_psi != 1)
        {

            // I have to add a right index
            rj = link_idx(j_psi-1);
            lj = link_idx_seed(j);
            sj = phys_idx_seed(j);
            Tj = ITensor(sj,rj,lj);

            for(int l=1; l<= dim(lj) ; l++)
			{
            for(int r=1; r<= dim(rj); r++)
            {
            for(int d=1; d<= dim(sj); d++)
            {
                if(r==1) Tj.set(lj=l,sj=d,rj=r , eltC(psi_seed(j), sj=d, lj=l) );
                else Tj.set(lj=l,sj=d,rj=r , 0 );
            }
            }
			}

            (*psi).set(j_psi,Tj);

        }

        else if (j==N_seed && j_psi != N)
        {
            // I have to add a left index
            rj = link_idx_seed(j-1);
            lj = link_idx(j_psi);
            sj = phys_idx_seed(j);
            Tj = ITensor(sj,rj,lj);

            for(int l=1; l<= dim(lj) ; l++)
			{
            for(int r=1; r<= dim(rj); r++)
            {
            for(int d=1; d<= dim(sj); d++)
            {
                // Tj.set(lj=l,sj=d,rj=r , eltC(psi_seed(j), sj=d, rj=r) );	
                if(l==1) Tj.set(lj=l,sj=d,rj=r , eltC(psi_seed(j), sj=d, rj=r) );
                else Tj.set(lj=l,sj=d,rj=r , 0 );		
            }
            }
			}

            (*psi).set(j_psi,Tj);
        }

        else if(j_psi <= N)
        {
            (*psi).set(j_psi,psi_seed(j));
        }

        else
        {
            cerr << "There is no room for accomodating the state." << endl;
            break;
        }

    }
    (*psi).replaceSiteInds(phys_idx);
}




// // Insert a state within another state, such that you have a state |state_to_insert> that you want to put in another state |psi_t0> from site start to start+L
// As we are dealing with sitesets with QN quantities, we have to define the link indices with a flux direction (In or Out)
// -> psi(j) ->

void
insert_QN_state(MPS* psi, MPS psi_seed, const int start, bool inverted,bool dagger)
{

    IndexSet phys_idx = siteInds(*psi);
    IndexSet phys_idx_seed = siteInds(psi_seed);

    IndexSet link_idx = linkInds(*psi);
    IndexSet link_idx_seed = linkInds(psi_seed);

    int N      = length(phys_idx);
    int N_seed = length(phys_idx_seed);

    Index lj ;
    Index rj ;
    Index sj ;
    ITensor Tj ;

    if(inverted)
    {
        MPS psi_seed_inv = psi_seed;
        int k = 1;
    
        for(int j= N_seed ; j >= 1; j--)
        {
            if(dagger) psi_seed_inv.set(k,dag(psi_seed(j)));
            else psi_seed_inv.set(k,psi_seed(j)); 
            k += 1;
            
        }

        psi_seed = psi_seed_inv;
        psi_seed.replaceSiteInds(phys_idx_seed);
        psi_seed.replaceLinkInds(link_idx_seed);
        link_idx_seed = linkInds(psi_seed);
    }

    // add tags to link indices, in order to avoid issues if we insert multiple copyes of the same state
    IndexSet new_links = IndexSet(N_seed-1);
    for(int j : range1(N_seed-1))
	{
		int j_psi = start + j - 1;
        Index new_idx = addTags(link_idx_seed(j),"seed="+str(j_psi));
        new_links[j-1] = new_idx;
    }

    psi_seed.replaceLinkInds(new_links);
    link_idx_seed = linkInds(psi_seed);


    // cerr << psi_seed(2) << endl;
    // cerr << link_idx_seed << endl;
    // double a ;
    // cin >> a;

	for(int j : range1(N_seed))
	{
		int j_psi = start + j - 1;
        

        if(j==1 && j_psi != 1)
        {

            // I have to add a right index
            rj = link_idx(j_psi-1);
            lj = link_idx_seed(j);
            sj = phys_idx_seed(j);
            Tj = ITensor(sj,rj,lj);

            for(int l=1; l<= dim(lj) ; l++)
			{
            for(int r=1; r<= dim(rj); r++)
            {
            for(int d=1; d<= dim(sj); d++)
            {
                if(r==1) Tj.set(lj=l,sj=d,rj=r , eltC(psi_seed(j), sj=d, lj=l) );
                else Tj.set(lj=l,sj=d,rj=r , 0 );
            }
            }
			}

            (*psi).set(j_psi,Tj);

        }

        else if (j==N_seed && j_psi != N)
        {
            // I have to add a left index
            rj = link_idx_seed(j-1);
            lj = link_idx(j_psi);
            sj = phys_idx_seed(j);
            Tj = ITensor(sj,rj,lj);

            cerr << Tj << endl;
            exit(0);
            for(int l=1; l<= dim(lj) ; l++)
			{
            for(int r=1; r<= dim(rj); r++)
            {
            for(int d=1; d<= dim(sj); d++)
            {
                // Tj.set(lj=l,sj=d,rj=r , eltC(psi_seed(j), sj=d, rj=r) );	
                if(l==1) Tj.set(lj=l,sj=d,rj=r , eltC(psi_seed(j), sj=d, rj=r) );
                else Tj.set(lj=l,sj=d,rj=r , 0 );		
            }
            }
			}

            (*psi).set(j_psi,Tj);
        }

        else if(j_psi <= N)
        {
            (*psi).set(j_psi,psi_seed(j));
            cerr << "I have inserted the state on site : " << j_psi << "\n";
        }

        else
        {
            cerr << "There is no room for accomodating the state." << endl;
            break;
        }

    }
    (*psi).replaceSiteInds(phys_idx);
}


