//
// Created by sdr2002 on 12/11/24.
//

#ifndef OUMODEL_H
#define OUMODEL_H

#include <vector>
#include <cstdlib>
#include <ctime>
#include "BSModel01.h"

using namespace std;

typedef vector<double> SamplePath;

class OUModel: public virtual Model
{
    public:
        double s0; // initial short-rate: economically equivalent to OverNight repo rate of the pricing date
        double drift;
        double sigma;

        OUModel(double P0_, double r_, double s0_, double drift_, double sigma_)
            : Model{P0_, r_}, s0(s0_), drift(drift_), sigma(sigma_) { srand(time(NULL)); }

        void GenerateSamplePath(double T, int m, SamplePath& S) override;
        double GetSinf() { return GetR(); }; // TODO validate if the asymptotic short-rate shall be the risk-free rate

        string toString() { return "OUModel"; };
};

#endif //OUMODEL_H
