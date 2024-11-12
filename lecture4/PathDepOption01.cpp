#include "PathDepOption01.h"
#include <cmath>

double PathDepOption::PriceByMC(BSModel Model, long N)
{
    double H = 0.0;
    SamplePath S(m);
    for (long i = 0; i < N; i++)
    {
        Model.GenerateSamplePath(T, m, S);
        H += Payoff(S);
    }

    const double DiscountFactor = exp(-Model.r * T);
    return DiscountFactor * H / N;
}

double ArthmAsianCall::Payoff(SamplePath& S)
{
    double Ave = 0.0;
    for (int k = 0; k < m; k++) Ave += S[k];
    Ave /= m;
    return max(Ave - K, 0.0);
}

double EuropeanCall::Payoff(SamplePath& S)
{
    double Sf = S.back();
    return max(Sf - K, 0.0);
}

double EuropeanPut::Payoff(SamplePath& S)
{
    double Sf = S.back();
    return max(K - Sf, 0.0);
}