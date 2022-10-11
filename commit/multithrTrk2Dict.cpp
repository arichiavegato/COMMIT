#include <thread>
#include <mutex>

// Dichiarata globalmente
std::mutex mutexCout; // mutex per la lettura della streamline


void Trk2DictMT( /* variabili passate da cython */ ){

    // Dichiarazione di P?

    for(int f=0; f<n_count ;f++)
    {
        mutexCout.lock();

        if ( isTRK ) // if isTRK is true then
            N = read_fiberTRK( fpTractogram, fiber, n_scalars, n_properties );
        else
            N = read_fiberTCK( fpTractogram, fiber , ptrToVOXMM );
        
        mutexCout.unlock();
            
        fiberForwardModel( fiber, N, nReplicas, ptrBlurRho, ptrBlurAngle, ptrBlurWeights, ptrBlurApplyTo[f], ptrHashTable, /*P?*/ );




}






