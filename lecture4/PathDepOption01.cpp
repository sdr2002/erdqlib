#include "PathDepOption01.h"
#include <cmath>

double PathDepOption::PriceByMC(Model& Model, long N)
{
    double H = 0.0;
    SamplePath S(m);
    for (long i = 0; i < N; i++)
    {
        Model.GenerateSamplePath(T, m, S);
        const double Hi = Payoff(S);
        H += Hi;
    }
    const double ExpectedH = H / N;

    const double DiscountFactor = exp(-Model.GetR() * T);
    return DiscountFactor * ExpectedH;
}

double PathDepOption::PriceByMC(Model& Model, long N, vector<double>& Sterminals)
{
    double H = 0.0;
    SamplePath S(m);
    for (long i = 0; i < N; i++)
    {
        Model.GenerateSamplePath(T, m, S);
        Sterminals.push_back(S.back());
        const double Hi = Payoff(S);
        H += Hi;
    }
    const double ExpectedH = H / N;

    const double DiscountFactor = exp(-Model.GetR() * T);
    return DiscountFactor * ExpectedH;
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
    return max(Sf - K, 0.0);
}