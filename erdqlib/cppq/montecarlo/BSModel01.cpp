#include "BSModel01.h"
#include "ModelCommon.h"
#include <cmath>


void BSModel::GenerateSamplePath(double T, int m, SamplePath& S)
{
    const double dt = T/m;

    double St = GetP0();
    for (int k = 0; k < m; k++)
    {
        S[k] = St * exp((GetR() - 0.5 * sqrt(sigma)) * dt + sigma * sqrt(dt) * Gauss());
        St = S[k];
    }
}
