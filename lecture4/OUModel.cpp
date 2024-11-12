//
// Created by sdr2002 on 12/11/24.
//

#include "OUModel.h"
#include "ModelCommon.h"
#include <cmath>

void OUModel::GenerateSamplePath(double T, int m, SamplePath& S)
{
    /* Random sample the security price by OU process as GROWTH rate of the price
     * The dynamics must be negative in order to pricing the zero coupon bond
     */
    const double dt = T/m;
    const double rho = 1 - exp(-drift * dt);
    const double s0effect = exp(s0/drift * rho); // s_0 * integrate[exp(-drift*t), t, 0, T] = exp( s_0 * rho/drift )
    const double sINFeffect = exp(GetSinf() * (dt - rho/drift)); // s_inf * integrate[1 - exp(-drift*t), 0, T] = exp( s_inf * (T - rho/drift) )

    double St = GetP0();
    for (int k = 0; k < m; k++)
    {
        /* volatility * integrate[ exp(-drift * t) * integrate[ exp(drift*s) * dW(s), s, 0, t], t, 0, T ]
         * Variance = ( volatility/drift )^2 * (T - 0.5/drift(rho^2 + 2*rho)) */
        const double var = pow(sigma/drift, 2.0)  * (dt - 0.5 * (pow(rho,2.0) + 2*rho)/drift);
        const double noise = exp(sqrt(var) * Gauss());
        S[k] = St * s0effect * sINFeffect * noise;
        St = S[k];
    }
}