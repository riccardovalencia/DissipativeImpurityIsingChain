#include "state_manipulation.h"
#include <itensor/all.h>
#include <iostream>


// move index from position j1 to position j2
// implementing virtual swap (not applying a gate)
// see http://itensor.org/support/2330/non-consecutive-swap-gates

void
swap_gate( MPS *psi, int j1, int j2, double cut_off, int maxDim)
{

    SiteSet sj = siteInds(*psi);
    ITensor T, U, V, D;
    Index sj_p;
    Index lj;


    if(j1 > j2)
    {
        int k = j1;
        j1 = j2;
        j2 = k; 
    }
   

    if( j1 < 1 || j2 > length(sj))
    {
        cerr << "Index out of physical bound" << endl;
        exit(0);
    }

    for(int j=j1 ; j<j2; j++)
    {
        T = (*psi)(j) * (*psi)(j+1);
        sj_p = sj(j+1);
        if (j > 1){
            lj = commonIndex((*psi)(j-1),(*psi)(j));
        }
        else{
            // NOT SURE. THINK ABOUT IT
            lj = commonIndex((*psi)(j),(*psi)(j+1));
            }

        U = ITensor(sj_p,lj);
        svd(T,U,D,V,{"Cutoff=",cut_off,"MaxDim=",maxDim});
        (*psi).set(j,U);
        (*psi).set(j+1,D*V);

    }


}


// ----------------------------------------------------------
// Given a pure state psi, presented as an MPS, it return its density matrix representation |psi> <psi| as an MPO

MPO 
from_MPS_to_MPDO(MPS psi )
{
    // ket and bra (bra is primed)
    MPS ket = psi;
    MPS bra = dag(prime(psi));

    // In order to merge the two MPS into an MPO (outer product), I use a similar procedure used in
    // nmultMPO -> the idea is to perform a transformation so that we introduce a new link index.

    // IndexSet 
    IndexSet sA  = siteInds(psi);
    IndexSet sB  = siteInds(bra);

    int N = length(sA);

    // MPO hosting the final MatrixProductDensityOperator
    MPO rho = MPO(sA);
    if(N==1)
    {
        rho.ref(1) = psi(1) * bra(1);
    }

    else
    {    
        IndexSet lA = linkInds(ket);
        IndexSet lB = linkInds(bra);

        // length

        ITensor clust, nfork;

        // empty ITensor as first element of rho;
        rho.ref(1) = ITensor(sA(1),sB(1),lA(1));

        for(int i : range1(N))
        {
            if(i==1) clust = psi(i) * bra(i);
            else     clust = nfork * psi(i) * bra(i);
            if(i==N-1) break;   

            // for site i=1 -> we have Tensor 
            //      |
            //      o =
            //      |
            // We want a single link index. To do so, we cut along the two link indices
            //      |
            //      o -  -o<
            //      | 
            // we need a 'fork' Tensor: -o< tensor

            nfork = ITensor(lA(i),lB(i),linkIndex(rho,i));

            // it perform a denmatDecomp (similar to SVD): will cut along the two link indeces
            denmatDecomp(clust,rho.ref(i),nfork,Fromleft,{{"MaxDim",500,"Cutoff",1E-16},"Tags=",tags(linkIndex(rho,i))});

            Index mid = commonIndex(rho(i),nfork);
            mid.dag(); // why the dag? I am taking from the nmultMPO ITensor code
            rho.ref(i+1) = ITensor(mid,sA(i+1),sB(i+1),rightLinkIndex(rho,i+1));
        }

        nfork = clust * psi(N) * bra(N);
        rho.svdBond(N-1,nfork,Fromright, {"MaxDim",500,"Cutoff",1E-16});
        rho.orthogonalize();
    }
    // Dumb way - issue with multiple link indices.
    // for(int j = 1 ; j<=N ; j++)
    // {
    //     ITensor Tj = ket(j) * bra(j);

    //     rho.ref(j) = Tj;
    // }
    return rho;

    
}


// ----------------------------------------------------------
// Given a pure state psi, presented as an MPS, it return its density matrix representation |psi> <psi| as an MPO
// Similar to above, but it uses a variation for fusing the link indices in a single one

MPO 
from_MPS_to_MPDO_v2(MPS psi )
{
    // ket and bra (bra is primed)
    MPS ket = psi;
    MPS bra = dag(prime(psi));

    // In order to merge the two MPS into an MPO (outer product), I use a similar procedure used in
    // nmultMPO -> the idea is to perform a transformation so that we introduce a new link index.

    // IndexSet 
    IndexSet sA  = siteInds(psi);
    IndexSet sB  = siteInds(bra);

    int N = length(sA);
    // cerr << "Lenght N : " << N << "\n";

    // MPO hosting the final MatrixProductDensityOperator
    MPO rho = MPO(sA);
    if(N==1)
    {
        rho.ref(1) = psi(1) * bra(1);
    }

    else
    {    
        IndexSet lA = linkInds(ket);
        IndexSet lB = linkInds(bra);
        IndexSet lrho = linkInds(rho);

        // length

        // MPO rho = MPO(sA); 
        ITensor clust, nfork; // helper ITensor

        // empty ITensor as first element of rho;
        // rho.ref(1) = ITensor(sA(1),sB(1),lA(1));

        for(int i = 1 ; i <= N ; i++)
        {
            clust = psi(i) * bra(i);
            if(i==1)
            {
                auto [C,c] = combiner(lA(i),lB(i));
                // cerr << clust << endl;
                // cerr << C << endl;
                clust = clust * C;
                // cerr << clust << endl;
                clust *= delta(c,lrho(i));
                // cerr << clust << endl;
                // exit(0);
            }

            else if(i>1 && i < N)
            {
                auto [C,c] = combiner(lA(i-1),lB(i-1));
                clust = clust * C;
                auto [C2,c2] = combiner(lA(i),lB(i));
                clust = clust * C2;
                   
                clust *= delta(c,lrho(i-1));
                clust *= delta(c2,lrho(i));                
            }

            else
            {
                auto [C,c] = combiner(lA(i-1),lB(i-1));
                clust = clust * C;
                clust *= delta(c,lrho(i-1));
            }
            rho.ref(i) = clust;
            cerr << rho(i) << endl;
            // exit(0);
        }
        // rho.orthogonalize();
    }
    // Dumb way - issue with multiple link indices.
    // for(int j = 1 ; j<=N ; j++)
    // {
    //     ITensor Tj = ket(j) * bra(j);

    //     rho.ref(j) = Tj;
    // }
    return rho;

    
}
