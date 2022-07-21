#include <stdio.h>
#include <cstdio>
#include <string>
#include <map>
#include <vector>
#include "Vector.h"
#include "ProgressBar.h"
#include <numpy/arrayobject.h>
#include <math.h>
#include <pthread.h>

using namespace std;

# define MAX_FIB_LEN 10000
# define n_threads 4


// CLASS to store the segments of one fiber
class segKey
{
    public:
    unsigned short x, y, z;
    unsigned short o;
    segKey(){}

    void set(unsigned short _x, unsigned short _y, unsigned short _z, unsigned short _o)
    {
        x  = _x;
        y  = _y;
        z  = _z;
        o = _o;
    }

    bool const operator <(const segKey& seg) const
    {
        return o < seg.o || (o==seg.o && z<seg.z) || (o==seg.o && z==seg.z && y<seg.y) || (o==seg.o && z==seg.z && y==seg.y && x<seg.x);
    }
};


class segInVoxKey
{
    public:
    unsigned short x, y, z;
    segInVoxKey(){}

    void set(unsigned short _x, unsigned short _y, unsigned short _z)
    {
        x  = _x;
        y  = _y;
        z  = _z;
    }

    bool const operator <(const segInVoxKey& o) const
    {
        return (z<o.z) || (z==o.z && y<o.y) || (z==o.z && y==o.y && x<o.x);
    }
};


// Values for threads
 struct Parameters {
    
    // To stor partial information 
    vector<unsigned int>    partialICSegments;
    vector<unsigned int>    partialFibers;
    vector<unsigned int>    partialECSegments;
    vector<unsigned int>    partialECVoxels;

    // For files
    char*           fp;             // Name of the tractrogram file
    char*           path_out;
    vector<int>     offset;         // For file descriptor
    int             threadID;   
    int             spt;            // number of Streamlines per Thread, new n_counts

    // from cython 
    int             n_scalar;
    int             n_properies;
    float*          ptrPEAKS;
    int             Np;
    float           vf_THR; 
    int             ECix; 
    int             ECiy; 
    int             ECiz;
    float*          ptrTDI;
    double*         ptrPeaksAffine;
    int             nReplicas;  
    double*         ptrBlurRho; 
    double*         ptrBlurAngle; 
    double*         ptrBlurWeights; 
    bool*           ptrBlurApplyTo;
    float*          ptrToVOXMM; 
    unsigned short  ndirs; 
    short*          ptrHashTable;

};


// per threads
pthread_t tin[n_threads];


// Dichiarazione variabili globali
vector<map<segKey,float> >              FiberSegments;
vector<float>                           FiberLen;      // length of a streamline
vector<float>                           FiberLenTot;   // length of a streamline (considering the blur)
// Vector< Vector< Vector<double> > >      P;  // In read functions aggiungere poi l'accesso in base all'id del thread


// Global variables
Vector<int>     dim;        
Vector<float>   pixdim;
float*          ptrMASK;
float           fiberShiftXmm, fiberShiftYmm, fiberShiftZmm;
bool            doIntersect;
float           minSegLen, minFiberLen, maxFiberLen;

int             isTRK;
int             hdr = 1000;


bool rayBoxIntersection( Vector<double>& origin, Vector<double>& direction, Vector<double>& vmin, Vector<double>& vmax, double & t);
void fiberForwardModel( float fiber[3][MAX_FIB_LEN], unsigned int pts, int nReplicas, double* ptrBlurRho, double* ptrBlurAngle, double* ptrBlurWeights, bool doApplyBlur, short* ptrHashTable );
void segmentForwardModel( const Vector<double>& P1, const Vector<double>& P2, int k, double w, short* ptrHashTable );
unsigned int read_fiberTRK( FILE* fp, float fiber[3][MAX_FIB_LEN], int ns, int np );
unsigned int read_fiberTCK( FILE* fp, float fiber[3][MAX_FIB_LEN] , float* toVOXMM );


// ========================================
//  Kernel function
// ========================================

void* T2DKernel ( void * structure ) {

    struct Parameters * pyValues;
    pyValues = ( struct Parameters * ) structure;  

    // threads variables
    float           fiber[3][MAX_FIB_LEN];
    float           fiberNorm;
    unsigned int    N, v, partialICSegments = 0, partialFibers = 0; 
    unsigned short  o;
    unsigned char   kept;
    string          filename;   // for the output files

    map<segKey,float>::iterator         it;
    map<segInVoxKey,float>              FiberNorm;
    map<segInVoxKey,float>::iterator    itNorm;
    segInVoxKey                         inVoxKey;
    
    Vector<Vector<double>  P;

    
    // From structure
    int idx = pyValues->threadID;
    int offset = pyValues->offset[idx];
    string OUTPUT_path( pyValues->path_out );
    int n_counts = pyValues->spt;
    

    filename = OUTPUT_path + "/dictionary_TRK_norm_" + to_string(idx) + ".dict";  
    FILE* pDict_TRK_norm = fopen(filename.c_str(),"wb");

    
     if ( !pDict_TRK_norm )
    {
        printf( "\n[trk2dictionary] Unable to create output files" );
        return 0;
    }

    // Open of the output files
    filename = OUTPUT_path + "/dictionary_IC_f_" + to_string(idx) + ".dict";          
    FILE* pDict_IC_f = fopen(filename.c_str(),"wb");
    
    filename = OUTPUT_path + "/dictionary_IC_v_" + to_string(idx) + ".dict";          
    FILE* pDict_IC_v = fopen(filename.c_str(),"wb");
    
    filename = OUTPUT_path + "/dictionary_IC_o_" + to_string(idx) + ".dict";          
    FILE* pDict_IC_o = fopen(filename.c_str(),"wb");
    
    filename = OUTPUT_path + "/dictionary_IC_len_" + to_string(idx) + ".dict";        
    FILE* pDict_IC_len = fopen(filename.c_str(),"wb");
    
    filename = OUTPUT_path + "/dictionary_TRK_len_" + to_string(idx) + ".dict";       
    FILE* pDict_TRK_len = fopen(filename.c_str(),"wb");
    
    filename = OUTPUT_path + "/dictionary_TRK_lenTot_" + to_string(idx) + ".dict";    
    FILE* pDict_TRK_lenTot = fopen(filename.c_str(),"wb");
    
    filename = OUTPUT_path + "/dictionary_TRK_kept_" + to_string(idx) + ".dict";      
    FILE* pDict_TRK_kept = fopen(filename.c_str(),"wb");   


    // Recupero l'offset per il filedescriptor
    offset = offset + hdr;  // + 1000 skip of the header

    FILE* fpTractogram = fopen(pyValues->fp,"rb"); // open for reading
    if (fpTractogram == NULL) return 0;
    fseek(fpTractogram,offset,SEEK_SET); //skip header    


    // Inizio dell'iterazione
    ProgressBar PROGRESS( n_counts );
    PROGRESS.setPrefix("     ");
    for(int f=0; f<n_counts ;f++)
    {
        PROGRESS.inc();

        if ( isTRK )
            N = read_fiberTRK( fpTractogram, fiber, pyValues->n_scalar, pyValues->n_properies );
        else
            N = read_fiberTCK( fpTractogram, fiber , pyValues->ptrToVOXMM );

        fiberForwardModel( fiber, N, pyValues->nReplicas, pyValues->ptrBlurRho, pyValues->ptrBlurAngle, 
                           pyValues->ptrBlurWeights, pyValues->ptrBlurApplyTo[f], pyValues->ptrHashTable, idx );

        kept = 0;

        if ( FiberSegments[idx].size() > 0 )
        {
            if ( FiberLen[idx] > minFiberLen && FiberLen[idx] < maxFiberLen )
            {
                int partialFib;

                // add segments to files
                for (it=FiberSegments[idx].begin(); it!=FiberSegments[idx].end(); it++)
                {
                    // NB: plese note inverted ordering for 'v'
                    v = it->first.x + dim.x * ( it->first.y + dim.y * it->first.z );
                    o = it->first.o;
                    fwrite( &pyValues->partialFibers[idx],      4, 1, pDict_IC_f );
                    fwrite( &v,              4, 1, pDict_IC_v );
                    fwrite( &o,              2, 1, pDict_IC_o );
                    fwrite( &(it->second),   4, 1, pDict_IC_len );
                    pyValues->ptrTDI[ it->first.z + dim.z * ( it->first.y + dim.y * it->first.x ) ] += it->second;
                    inVoxKey.set( it->first.x, it->first.y, it->first.z );
                    FiberNorm[inVoxKey] += it->second;
                }

                for (fiberNorm=0, itNorm=FiberNorm.begin(); itNorm!=FiberNorm.end(); itNorm++)
                    fiberNorm += pow(itNorm->second,2);
                
                fiberNorm = sqrt(fiberNorm);
                FiberNorm.clear();
                fwrite( &fiberNorm,   1, 4, pDict_TRK_norm );   // actual length considered in optimization
                fwrite( &FiberLen[idx],    1, 4, pDict_TRK_len );    // length of the streamline
                fwrite( &FiberLenTot[idx], 1, 4, pDict_TRK_lenTot ); // length of the streamline (considering the blur)
                pyValues->partialICSegments[idx] += FiberSegments[idx].size();
                
                partialFibers++; // qui serve una sol

                kept = 1;
            }
        }
        fwrite( &kept, 1, 1, pDict_TRK_kept );
    }
    PROGRESS.close();

    // fclose( vals->fp ); questo lo dovra' fare il main pero'
    fclose( pDict_TRK_norm );
    fclose( pDict_IC_f );
    fclose( pDict_IC_v );
    fclose( pDict_IC_o );
    fclose( pDict_IC_len );
    fclose( pDict_TRK_len );
    fclose( pDict_TRK_lenTot );
    fclose( pDict_TRK_kept );

    printf("     [ %d streamlines kept, %d segments in total, in thread %d ]\n", pyValues->partialFibers, pyValues->partialICSegments, idx );


    /*=========================*/
    /*     EC compartments     */
    /*=========================*/
    unsigned int partialECSegments = 0, partialECVoxels = 0; 

    printf( "\n   \033[0;32m* Exporting EC compartments:\033[0m\n" );

    filename = OUTPUT_path + "/dictionary_EC_v_" + to_string(idx) + ".dict";
    FILE* pDict_EC_v   = fopen( filename.c_str(),   "wb" );
    
    filename = OUTPUT_path + "/dictionary_EC_o_" + to_string(idx) + ".dict";       
    FILE* pDict_EC_o   = fopen( filename.c_str(),   "wb" );

    if ( pyValues->ptrPEAKS != NULL )
    {

        int Np = pyValues->Np;

        Vector<double> dir;
        double         longitude, colatitude;
        segKey         ec_seg;
        int            ix, iy, iz, id, atLeastOne;
        float          peakMax;
        float          norms[ Np ];
        float          *ptr;
        int            ox, oy;

        PROGRESS.reset( dim.z );
        for(iz=0; iz<dim.z ;iz++)
        {
            PROGRESS.inc();
            for(iy=0; iy<dim.y ;iy++)
            for(ix=0; ix<dim.x ;ix++)
            {
                // check if in mask previously computed from IC segments
                if ( pyValues->ptrTDI[ iz + dim.z * ( iy + dim.y * ix ) ] == 0 ) continue;

                peakMax = -1;
                for(id=0; id<Np ;id++)
                {
                    ptr = pyValues->ptrPEAKS + 3*(id + Np * ( iz + dim.z * ( iy + dim.y * ix ) ));
                    dir.x = ptr[0];
                    dir.y = ptr[1];
                    dir.z = ptr[2];
                    norms[id] = dir.norm();
                    if ( norms[id] > peakMax )
                        peakMax = norms[id];
                }

                if ( peakMax > 0 )
                {
                    ec_seg.x  = ix;
                    ec_seg.y  = iy;
                    ec_seg.z  = iz;
                    atLeastOne = 0;
                    for(id=0; id<Np ;id++)
                    {
                        if ( norms[id]==0 || norms[id] < pyValues->vf_THR*peakMax ) continue; // peak too small, don't consider it

                        // get the orientation of the current peak
                        ptr = pyValues->ptrPEAKS + 3*(id + Np * ( iz + dim.z * ( iy + dim.y * ix ) ));

                        // multiply by the affine matrix
                        dir.x = ptr[0] * pyValues->ptrPeaksAffine[0] + ptr[1] * pyValues->ptrPeaksAffine[1] + ptr[2] * pyValues->ptrPeaksAffine[2];
                        dir.y = ptr[0] * pyValues->ptrPeaksAffine[3] + ptr[1] * pyValues->ptrPeaksAffine[4] + ptr[2] * pyValues->ptrPeaksAffine[5];
                        dir.z = ptr[0] * pyValues->ptrPeaksAffine[6] + ptr[1] * pyValues->ptrPeaksAffine[7] + ptr[2] * pyValues->ptrPeaksAffine[8];

                        // flip axes if requested
                        dir.x *= pyValues->ECix;
                        dir.y *= pyValues->ECiy;
                        dir.z *= pyValues->ECiz;
                        if ( dir.y < 0 )
                        {
                            // ensure to be in the right hemisphere (the one where kernels were pre-computed)
                            dir.x = -dir.x;
                            dir.y = -dir.y;
                            dir.z = -dir.z;
                        }
                        colatitude = atan2( sqrt(dir.x*dir.x + dir.y*dir.y), dir.z );
                        longitude  = atan2( dir.y, dir.x );
                        ox = (int)round(colatitude/M_PI*180.0);
                        oy = (int)round(longitude/M_PI*180.0);

                        v = ec_seg.x + dim.x * ( ec_seg.y + dim.y * ec_seg.z );
                        o = pyValues->ptrHashTable[ox*181 + oy];
                        fwrite( &v, 4, 1, pDict_EC_v );
                        fwrite( &o, 2, 1, pDict_EC_o );
                        partialECSegments++;
                        atLeastOne = 1;
                    }
                    if ( atLeastOne>0 )
                        partialECVoxels++;
                }
            }
        }
        PROGRESS.close();
    }

    fclose( pDict_EC_v );
    fclose( pDict_EC_o );

    printf("     [ %d voxels, %d segments ]\n", partialECVoxels, partialECSegments );

    pyValues->partialECSegments[idx] = partialECSegments;
    pyValues->partialECVoxels[idx] = partialECVoxels;
    pyValues->partialICSegments[idx] = partialICSegments;
    pyValues->partialFibers[idx] = partialFibers;

    pthread_exit(0);

};



// =========================
// Function called by CYTHON
// =========================
int trk2dictionary(
    char* str_filename, int data_offset, int Nx, int Ny, int Nz, float Px, float Py, float Pz, int n_count, int n_scalars, int n_properties,
    float fiber_shiftX, float fiber_shiftY, float fiber_shiftZ, float min_seg_len, float min_fiber_len, float max_fiber_len,
    float* ptrPEAKS, int Np, float vf_THR, int ECix, int ECiy, int ECiz,
    float* _ptrMASK, float* ptrTDI, char* path_out, int c, double* ptrPeaksAffine,
    int nReplicas, double* ptrBlurRho, double* ptrBlurAngle, double* ptrBlurWeights, bool* ptrBlurApplyTo,
    float* ptrToVOXMM, unsigned short ndirs, short* ptrHashTable
)
{

    int spt = n_count/n_threads;    //streamline per thread

    if ( n_count%n_threads == 0 ) {
        spt = n_count/n_threads;
    } 
    else spt = ceil( n_count/n_threads );


    int isTRK;
    char *ext = strrchr(str_filename, '.');
    if (strcmp(ext,".trk")==0) //for .trk file -> strcmp, string compare, meaning if ext = ".trk" then ...
        isTRK = 1;
    else if (strcmp(ext,".tck")==0)// for .tck file
        isTRK = 0;
    else
        return 0;  


    printf( "\n   \033[0;32m* Exporting IC compartments:\033[0m\n" );


    // Calcolo offsets per ogni thread
    vector<double> offsets_struct;


    // Inizializzo struttura
    Parameters values = {
        str_filename,
        path_out,
        { 0, 0, 0, 0 }, //offsets
        0,  //ThreadID, settato dal thread
        n_scalars,
        n_properties,
        spt,            // streamline per thread da usare al posto di n_count
        ptrPEAKS,
        Np,
        vf_THR,
        ECix,
        ECiy,
        ECiz,
        ptrTDI,
        ptrPeaksAffine,
        nReplicas, 
        ptrBlurRho, 
        ptrBlurAngle, 
        ptrBlurWeights, 
        ptrBlurApplyTo,
        ptrToVOXMM, 
        ndirs,
        ptrHashTable

    };

    // set global variables
    // Passed form cython, retrived from hdr
    dim.Set( Nx, Ny, Nz );
    pixdim.Set( Px, Py, Pz );
    fiberShiftXmm = fiber_shiftX * pixdim.x; // shift in mm for the coordinates
    fiberShiftYmm = fiber_shiftY * pixdim.y;
    fiberShiftZmm = fiber_shiftZ * pixdim.z;
    ptrMASK       = _ptrMASK;
    doIntersect   = c > 0;
    minSegLen     = min_seg_len;
    minFiberLen   = min_fiber_len;
    maxFiberLen   = max_fiber_len;

    // P.resize( nReplicas );  (?)


    for(int j=1; j <= n_threads; j++){
        values.threadID[j] = j;
        pthread_create( &tin[j], NULL, inizializza, (void*)&values );
    }

    // Aspetto la terminazione
    for( int i=1; i<=n_threads ;i++ ){
            pthread_join( tin[i],NULL );
    }

    // chiudo il file in lettura

};



/********************************************************************************************************************/
/*                                                 fiberForwardModel                                                */
/********************************************************************************************************************/
void fiberForwardModel( float fiber[3][MAX_FIB_LEN], unsigned int pts, int nReplicas, double* ptrBlurRho, double* ptrBlurAngle, double* ptrBlurWeights, bool doApplyBlur, short* ptrHashTable, int idx )
{
    static Vector<double> S1, S2, S1m, S2m, P_old, P_int, q, n, nr, qxn, qxqxn;
    static Vector<double> vox, vmin, vmax, dir1, dir2;
    static double         len, t, alpha, w, R, dot;
    static int            i, j, k;

    FiberLen[idx] = 0.0;
    FiberLenTot[idx] = 0.0;
    FiberSegments[idx].clear();
    
    if ( pts <= 2 )
        return;

    // create duplicate points on circles
    S1.x = fiber[0][0]+fiberShiftXmm;
    S1.y = fiber[1][0]+fiberShiftYmm;
    S1.z = fiber[2][0]+fiberShiftZmm;
    dir2.x = (fiber[0][1]+fiberShiftXmm) - S1.x;
    dir2.y = (fiber[1][1]+fiberShiftYmm) - S1.y;
    dir2.z = (fiber[2][1]+fiberShiftZmm) - S1.z;
    dir2.Normalize();
    n.x = dir2.y-dir2.z;
    n.y = dir2.z-dir2.x;
    n.z = dir2.x-dir2.y;
    n.Normalize();

    // duplicate first point and move to corresponding grid locations
    for(k=0; k<nReplicas ;k++)
    {
        if ( !doApplyBlur && k>0 )
            continue;
        R = ptrBlurRho[k];
        alpha = ptrBlurAngle[k];

        // quaternion (q.x, q.y, q.z, w) for rotation
        w = sin(alpha/2.0);
        q.x = dir2.x * w;
        q.y = dir2.y * w;
        q.z = dir2.z * w;
        w = cos(alpha/2.0);

        // rotate the segment's normal
        qxn.x = 2.0 * ( q.y * n.z - q.z * n.y );
        qxn.y = 2.0 * ( q.z * n.x - q.x * n.z );
        qxn.z = 2.0 * ( q.x * n.y - q.y * n.x );
        qxqxn.x = q.y * qxn.z - q.z * qxn.y;
        qxqxn.y = q.z * qxn.x - q.x * qxn.z;
        qxqxn.z = q.x * qxn.y - q.y * qxn.x;
        nr.x = n.x + w * qxn.x + qxqxn.x;
        nr.y = n.y + w * qxn.y + qxqxn.y;
        nr.z = n.z + w * qxn.z + qxqxn.z;
        nr.Normalize();

        // move first point to corresponding grid location
        S2.x = S1.x + R*nr.x;
        S2.y = S1.y + R*nr.y;
        S2.z = S1.z + R*nr.z;
        P[k] = S2;
    }

    // move all remaining points
    for(i=1; i<pts ;i++)
    {
        /* get the intersection plane */
        // S2 = point on plane
        S2.x = fiber[0][i]+fiberShiftXmm;
        S2.y = fiber[1][i]+fiberShiftYmm;
        S2.z = fiber[2][i]+fiberShiftZmm;

        // n = normal to plane
        dir1.x = S2.x - (fiber[0][i-1]+fiberShiftXmm);
        dir1.y = S2.y - (fiber[1][i-1]+fiberShiftYmm);
        dir1.z = S2.z - (fiber[2][i-1]+fiberShiftZmm);
        dir1.Normalize();
        if ( i == pts-1 )
        {
            dir2.x = dir1.x;
            dir2.y = dir1.y;
            dir2.z = dir1.z;
        } else {
            dir2.x = (fiber[0][i+1]+fiberShiftXmm) - S2.x;
            dir2.y = (fiber[1][i+1]+fiberShiftYmm) - S2.y;
            dir2.z = (fiber[2][i+1]+fiberShiftZmm) - S2.z;
            dir2.Normalize();
        }
        n.x = 0.5*(dir1.x+dir2.x);
        n.y = 0.5*(dir1.y+dir2.y);
        n.z = 0.5*(dir1.z+dir2.z);

        // normalize to avoid computations later on
        dot = dir1.x*n.x + dir1.y*n.y + dir1.z*n.z;
        n.x /= dot;
        n.y /= dot;
        n.z /= dot;

        /* translate points */
        for(k=0; k<nReplicas ;k++)
        {
            if ( !doApplyBlur && k>0 )
                continue;

            if ( ptrBlurWeights[k] < 1e-3 )
                continue;

            P_old.x = P[k].x;
            P_old.y = P[k].y;
            P_old.z = P[k].z;
            len = (S2.x-P_old.x)*n.x + (S2.y-P_old.y)*n.y + (S2.z-P_old.z)*n.z;
            if ( len>0 )
            {
                P[k].x += dir1.x * len;
                P[k].y += dir1.y * len;
                P[k].z += dir1.z * len;

                /* save segment */
                if ( doIntersect==false )
                    segmentForwardModel( P_old, P[k], k, ptrBlurWeights[k], ptrHashTable, idx );
                else
                {
                    S1m.x = P_old.x;
                    S1m.y = P_old.y;
                    S1m.z = P_old.z;
                    S2m.x = P[k].x;
                    S2m.y = P[k].y;
                    S2m.z = P[k].z;
                    while( 1 )
                    {
                        len = sqrt( pow(S2m.x-S1m.x,2) + pow(S2m.y-S1m.y,2) + pow(S2m.z-S1m.z,2) ); // in mm
                        if ( len <= /*inStruct*/minSegLen )
                            break;

                        if ( floor(S1m.x/pixdim.x)==floor(S2m.x/pixdim.x) &&
                            floor(S1m.y/pixdim.y)==floor(S2m.y/pixdim.y) &&
                            floor(S1m.z/pixdim.z)==floor(S2m.z/pixdim.z)
                            )
                        {
                            // same voxel, no need to compute intersections
                            segmentForwardModel( S1m, S2m, k, ptrBlurWeights[k], ptrHashTable, idx );
                            break;
                        }

                        // compute AABB of the first point (in mm)
                        vmin.x = floor( (S1m.x + 1e-6*dir1.x)/pixdim.x ) * pixdim.x;
                        vmin.y = floor( (S1m.y + 1e-6*dir1.y)/pixdim.y ) * pixdim.y;
                        vmin.z = floor( (S1m.z + 1e-6*dir1.z)/pixdim.z ) * pixdim.z;
                        vmax.x = vmin.x + pixdim.x;
                        vmax.y = vmin.y + pixdim.y;
                        vmax.z = vmin.z + pixdim.z;

                        if ( rayBoxIntersection( S1m, dir1, vmin, vmax, t ) && t>0 && t<len )
                        {
                            // add the portion S1P, and then reiterate
                            P_int.x = S1m.x + t*dir1.x;
                            P_int.y = S1m.y + t*dir1.y;
                            P_int.z = S1m.z + t*dir1.z;
                            segmentForwardModel( S1m, P_int, k, ptrBlurWeights[k], ptrHashTable, idx );
                            S1m.x = P_int.x;
                            S1m.y = P_int.y;
                            S1m.z = P_int.z;
                        }
                        else
                        {
                            // add the segment S1S2 and stop iterating
                            segmentForwardModel( S1m, S2m, k, ptrBlurWeights[k], ptrHashTable, idx );
                            break;
                        }
                    }
                }
            }
        }
    }
}





/********************************************************************************************************************/
/*                                                segmentForwardModel                                               */
/********************************************************************************************************************/
void segmentForwardModel( const Vector<double>& P1, const Vector<double>& P2, int k, double w, short* ptrHashTable, int idx )
{
    static Vector<int>    vox;
    static Vector<double> dir, dirTrue;
    static double         longitude, colatitude, len;
    static segKey         key;
    static int            ox, oy;

    // direction of the segment
    dir.y = P2.y-P1.y;
    if ( dir.y >= 0 )
    {
        dir.x = P2.x-P1.x;
        dir.z = P2.z-P1.z;
    }
    else
    {
        dir.x = P1.x-P2.x;
        dir.y = P1.y-P2.y;
        dir.z = P1.z-P2.z;
    }

    // length of the segment
    len = dir.norm();
    if ( w*len <= minSegLen )
        return;
    dir.Normalize();

    // voxel of the segment is the centroid
    vox.x = floor( 0.5 * (P1.x + P2.x) / pixdim.x );
    vox.y = floor( 0.5 * (P1.y + P2.y) / pixdim.y );
    vox.z = floor( 0.5 * (P1.z + P2.z) / pixdim.z );
    if ( vox.x>=dim.x || vox.x<0 || vox.y>=dim.y || vox.y<0 || vox.z>=dim.z || vox.z<0 )
        return;
    if ( ptrMASK && ptrMASK[ vox.z + dim.z * ( vox.y + dim.y * vox.x ) ]==0 )
        return;

    // add the segment to the data structure
    longitude  = atan2(dir.y, dir.x);
    colatitude = atan2( sqrt(dir.x*dir.x + dir.y*dir.y), dir.z );
    ox = (int)round(colatitude/M_PI*180.0); // theta // i1
    oy = (int)round(longitude/M_PI*180.0);  // phi   // i2
    key.set( vox.x, vox.y, vox.z, (unsigned short) ptrHashTable[ox*181 + oy] );
    FiberSegments[idx][key] += w*len;
    FiberLenTot[idx] += w*len;
    if ( k==0 ) // fiber length computed only from original segments
        FiberLen[idx] += len;
}


/********************************************************************************************************************/
/*                                                rayBoxIntersection                                                */
/********************************************************************************************************************/
bool rayBoxIntersection( Vector<double>& origin, Vector<double>& direction, Vector<double>& vmin, Vector<double>& vmax, double & t)
{
    static double tmin, tmax, tymin, tymax, tzmin, tzmax;
    static Vector<double> invrd;

    // inverse direction to catch float problems
    invrd.x = 1.0 / direction.x;
    invrd.y = 1.0 / direction.y;
    invrd.z = 1.0 / direction.z;

    if (invrd.x >= 0)
    {
      tmin = (vmin.x - origin.x) * invrd.x;
      tmax = (vmax.x - origin.x) * invrd.x;
    }
    else
    {
      tmin = (vmax.x - origin.x) * invrd.x;
      tmax = (vmin.x - origin.x) * invrd.x;
    }

    if (invrd.y >= 0)
    {
      tymin = (vmin.y - origin.y) * invrd.y;
      tymax = (vmax.y - origin.y) * invrd.y;
    }
    else
    {
      tymin = (vmax.y - origin.y) * invrd.y;
      tymax = (vmin.y - origin.y) * invrd.y;
    }

    if ( (tmin > tymax) || (tymin > tmax) ) return false;
    if ( tymin > tmin) tmin = tymin;
    if ( tymax < tmax) tmax = tymax;

    if (invrd.z >= 0)
    {
      tzmin = (vmin.z - origin.z) * invrd.z;
      tzmax = (vmax.z - origin.z) * invrd.z;
    }else
    {
      tzmin = (vmax.z - origin.z) * invrd.z;
      tzmax = (vmin.z - origin.z) * invrd.z;
    }

    if ( (tmin > tzmax) || (tzmin > tmax) ) return false;
    if ( tzmin > tmin) tmin = tzmin;
    if ( tzmax < tmax) tmax = tzmax;

    // check if values are valid
    t = tmin;
    if (t <= 0) t = tmax;

    return true;
}


// Read a fiber from file .trk
unsigned int read_fiberTRK( FILE* fp, float fiber[3][MAX_FIB_LEN], int ns, int np )
{
    int N; //Number of points in the track
    fread((char*)&N, 1, 4, fp);

    if ( N >= MAX_FIB_LEN || N <= 0 )
        return 0;

    float P[3];

    for(int i=0; i<N; i++)
    {
        fread((char*)P, 1, 12, fp); // P e' l'allocazione di memoria in cui vengono salvati i valori, 1 il num di bytes, 12 i bytes da leggere  ( dal momento che i float valgono 4 bytes) e fp il file da dove leggerli
        fiber[0][i] = P[0];
        fiber[1][i] = P[1];
        fiber[2][i] = P[2];
        fseek(fp,4*ns,SEEK_CUR);
    }
    fseek(fp,4*np,SEEK_CUR);

    return N;
}


// Read a fiber from file .tck
unsigned int read_fiberTCK( FILE* fp, float fiber[3][MAX_FIB_LEN], float* ptrToVOXMM )
{
    int i = 0;
    float P[3];
    fread((char*)P, 1, 12, fp);
    while( !(isnan(P[0])) && !(isnan(P[1])) &&  !(isnan(P[2])) )
    {
        fiber[0][i] = P[0] * ptrToVOXMM[0] + P[1] * ptrToVOXMM[1] + P[2] * ptrToVOXMM[2]  + ptrToVOXMM[3];
        fiber[1][i] = P[0] * ptrToVOXMM[4] + P[1] * ptrToVOXMM[5] + P[2] * ptrToVOXMM[6]  + ptrToVOXMM[7];
        fiber[2][i] = P[0] * ptrToVOXMM[8] + P[1] * ptrToVOXMM[9] + P[2] * ptrToVOXMM[10] + ptrToVOXMM[11];
        i++;
        fread((char*)P, 1, 12, fp);
    }

    return i;
}





