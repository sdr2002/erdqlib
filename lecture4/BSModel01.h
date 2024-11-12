#ifndef BSModel01_h
#define BSModel01_h

#include <vector>
#include <cstdlib>
#include <ctime>
#include "Model.h"

using namespace std;

class BSModel: public virtual Model
{
    public:
        double sigma;

        BSModel(double S0_, double r_, double sigma_)
            : Model{S0_, r_}, sigma(sigma_) { srand(time(NULL)); }

        void GenerateSamplePath(double T, int m, SamplePath& S) override;

        string toString() { return "BSModel"; };
};

#endif
