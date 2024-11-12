//
// Created by sdr2002 on 12/11/24.
//

#ifndef MODEL_H
#define MODEL_H
#include <stdexcept>
#include <vector>

using namespace std;

typedef std::vector<double> SamplePath;

class Model
{
    double P0;  // initial price
    double r;   // risk-free rate
    public:
        Model(double P0_, double r_): P0(P0_), r(r_) {};
        double GetR() { return r; }
        double GetP0() { return P0; }

        virtual void GenerateSamplePath(double T, int m, SamplePath& S) {
            throw logic_error("Not implemented");
        };

        virtual string toString() { throw logic_error("Not implemented"); };
};

#endif //MODEL_H
