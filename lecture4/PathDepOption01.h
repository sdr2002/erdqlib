#ifndef PathDepOption01_h
#define PathDepOption01_h

#include "BSModel01.h"

class PathDepOption {
public:
    double T;
    int m;

    PathDepOption(double T_, int m_) : T(T_), m(m_) {};

    double PriceByMC(Model& Model, long N);
    double PriceByMC(Model& Model, long N, vector<double>& Sterminals);

    virtual double Payoff(SamplePath& S) = 0;
};

class ArthmAsianCall : public PathDepOption {
public:
    double K;

    ArthmAsianCall(double T_, double K_, int m_) : PathDepOption(T_, m_), K(K_) {}

    double Payoff(SamplePath &S) override;
};

class EuropeanCall : public PathDepOption {
public:
    double K;

    EuropeanCall(double T_, double K_, int m_) : PathDepOption(T_, m_), K(K_) {}

    double Payoff(SamplePath &S) override;
};

class EuropeanPut : public PathDepOption {
public:
    double K;

    EuropeanPut(double T_, double K_, int m_) : PathDepOption(T_, m_), K(K_) {}

    double Payoff(SamplePath &S) override;
};

#endif
