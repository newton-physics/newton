#include <iostream>
//#include <DefKitAdv.h>

//#include "..\VMProtectSDK.h"
//#include <mkl.h>

extern "C" {
    void SPBSV(const char* uplo, const int* n, const int* kd, const int* nrhs, float* ab, const int* ldab, float* b, const int* ldb, int* info) {
        std::cerr << "SPBSV called but MKL is disabled." << std::endl;
        *info = -1;
    }
}


#include <Eigen\Core>
#include <Eigen\src\Core\BandMatrix.h>
#ifdef  VMPROTECT
#include "..\VMProtectSDK.h"
#endif
#include "PositionBasedElasticRods.h"
#include "MathFunctions.h"


#define _USE_MATH_DEFINES

#include "math.h"

//#include <complex>
//#define lapack_complex_float std::complex<float>
//#define lapack_complex_double std::complex<double>
//#include <lapack.h>

using namespace PBD;

const Real eps = static_cast<Real>(1e-6);

const int permutation[3][3] = {
	0, 2, 1,
	1, 0, 2,
	2, 1, 0
};

// ----------------------------------------------------------------------------------------------
bool PositionBasedCosseratRods::solve_StretchShearConstraint(
// ... rest of the file
