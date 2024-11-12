#include <iostream>
#include "PathDepOption01.h"

using namespace std;

int evaluateWithBlackScholesDynamics()
{
    double S0 = 100.0, r = 0.03, sigma = 0.2;
    BSModel Model(S0, r, sigma);

    double T = 1.0 / 12.0, K = 100.0; // Expiry is 1 month.
    int m = 30;                       // Daily observations for one month!

    long N = 30000;

    ArthmAsianCall AriAsiCallOption(T, K, m);
    cout << "Arithmetic Asian Call Price = " << AriAsiCallOption.PriceByMC(Model, N) << endl;

    // lecture 4. Exercise 1
    EuropeanCall EurCallOption(T, K, m);
    cout << "European Call Price = " << EurCallOption.PriceByMC(Model, N) << endl;
    EuropeanCall EurPutOption(T, K, m);
    cout << "European Put Price = " << EurPutOption.PriceByMC(Model, N) << endl;

    return 0;
}


int main() {
    evaluateWithBlackScholesDynamics();
}