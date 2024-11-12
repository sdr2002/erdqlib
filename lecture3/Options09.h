//
// Created by sdr2002 on 29/10/24.
//

#ifndef OPTIONS08_H
#define OPTIONS08_H
#include <format>
#include "BinLattice02.h"
#include "../lecture2/BinModel02.h"

// TODO introduce Vega and Theta calculator

class Option {
    int N;
    public:
        double PerturbationRatio = 0.1;
        void SetN(int N_){N=N_;}
        int GetN(){return N;}
        virtual double Payoff(double z)=0;
};

class EurOption: public virtual Option {
    public:
        double PriceByCRR(BinModel Model);
        double DeltaByCRR(BinModel Model) {
            BinModel Modelpp = BinModel(Model);
            Modelpp.SetS0(Model.GetS0()*(1+PerturbationRatio));

            BinModel Modelmm = BinModel(Model);
            Modelmm.SetS0(Model.GetS0()*(1-PerturbationRatio));

            return (PriceByCRR(Modelpp) - PriceByCRR(Modelmm)) / (Model.GetS0()*2*PerturbationRatio);
        };
        double GammaByCRR(BinModel Model) {
            BinModel Modelpp = BinModel(Model);
            Modelpp.SetS0(Model.GetS0()*(1+PerturbationRatio));

            BinModel Modelmm = BinModel(Model);
            Modelmm.SetS0(Model.GetS0()*(1-PerturbationRatio));

            return (DeltaByCRR(Modelpp) - DeltaByCRR(Modelmm)) / (Model.GetS0()*2*PerturbationRatio);
        };
};

class AmOption: public virtual Option {
    public:
        double PriceBySnell(
            BinModel Model,
            BinLattice<double>& PriceTree,
            BinLattice<bool>& StoppingTree
        );
        double PriceBySnell(BinModel Model) {
            BinLattice<double> PriceTree("PriceTree");
            BinLattice<bool> StoppingTree("StoppingTree");
            return PriceBySnell(Model, PriceTree, StoppingTree);
        }
        double DeltaBySnell(BinModel Model) {
            BinModel Modelpp = BinModel(Model);
            Modelpp.SetS0(Model.GetS0()*(1+PerturbationRatio));

            BinModel Modelmm = BinModel(Model);
            Modelmm.SetS0(Model.GetS0()*(1-PerturbationRatio));
            return (PriceBySnell(Modelpp) - PriceBySnell(Modelmm)) / (Model.GetS0()*2*PerturbationRatio);
        }
        double GammaBySnell(BinModel Model) {
            BinModel Modelpp = BinModel(Model);
            Modelpp.SetS0(Model.GetS0()*(1+PerturbationRatio));

            BinModel Modelmm = BinModel(Model);
            Modelmm.SetS0(Model.GetS0()*(1-PerturbationRatio));
            return (DeltaBySnell(Modelpp) - DeltaBySnell(Modelmm)) / (Model.GetS0()*2*PerturbationRatio);
        }
};

class Call: public EurOption, public AmOption {
    double K;
    public:
        friend std::string to_string(const Call& x) { return "Call"; }
        void SetK(double K_){K=K_;}
        double GetK(){return K;}
        int GetInputGridParameters();
        int GetDefaultGridParameters();
        double Payoff(double z);
};

class Put: public EurOption, public AmOption {
    double K;
    public:
        friend std::string to_string(const Put& x) { return "PUT"; }
        void SetK(double K_){K=K_;}
        double GetK(){return K;}
        int GetInputGridParameters();
        int GetDefaultGridParameters();
        double Payoff(double z);
};

class KnockOutCall: public Call {
    double K;
    double Barrier;
    public:
        friend std::string to_string(const KnockOutCall& x) { return "KnockOutCall"; }
        void SetK(double K_){K=K_;}
        void SetBarrier(double barrier){Barrier=barrier;}
        double GetBarrier(){return this->Barrier;}
        int GetInputGridParameters();
        int GetDefaultGridParameters();
        double Payoff(double z);
};

class CallSpread: public virtual Option {
    Call CallLong;
    Call CallShort;

    public:
        friend std::string to_string(const CallSpread& x) { return "CallSpread"; }
        void SetK(double Kl_, double Ks_) {
            CallLong.SetK(Kl_);
            CallShort.SetK(Ks_);
        };
        double GetK() {
            double Ks[] = {CallLong.GetK(), CallShort.GetK()};
            return *Ks;
        }
        string GetKasString() {
            string buffer;
            format_to(
                std::back_inserter(buffer),
                "Kl={0:.1f}, Ks={1:.1f}",
                CallLong.GetK(), CallShort.GetK()
            );
            return buffer;
        }
        void SetN(double N) {
            CallLong.SetN(N);
            CallShort.SetN(N);
        };
        int GetDefaultGridParameters();
        double Payoff(double z) {
            return CallLong.Payoff(z) - CallShort.Payoff(z);
        };

        double PriceByCRR(BinModel Model) {
            return CallLong.PriceByCRR(Model) - CallShort.PriceByCRR(Model);
        };
        double PriceBySnell(BinModel Model) {
            return CallLong.PriceBySnell(Model) - CallShort.PriceBySnell(Model);
        }
        double DeltaByCRR(BinModel Model) {
            BinModel Modelpp = BinModel(Model);
            Modelpp.SetS0(Model.GetS0()*(1+PerturbationRatio));

            BinModel Modelmm = BinModel(Model);
            Modelmm.SetS0(Model.GetS0()*(1-PerturbationRatio));
            return (PriceByCRR(Modelpp) - PriceByCRR(Modelmm)) / (Model.GetS0()*2*PerturbationRatio);
        }
        double GammaByCRR(BinModel Model) {
            BinModel Modelpp = BinModel(Model);
            Modelpp.SetS0(Model.GetS0()*(1+PerturbationRatio));

            BinModel Modelmm = BinModel(Model);
            Modelmm.SetS0(Model.GetS0()*(1-PerturbationRatio));
            return (DeltaByCRR(Modelpp) - DeltaByCRR(Modelmm)) / (Model.GetS0()*2*PerturbationRatio);
        }
        double DeltaBySnell(BinModel Model) {
            BinModel Modelpp = BinModel(Model);
            Modelpp.SetS0(Model.GetS0()*(1+PerturbationRatio));

            BinModel Modelmm = BinModel(Model);
            Modelmm.SetS0(Model.GetS0()*(1-PerturbationRatio));
            return (PriceBySnell(Modelpp) - PriceBySnell(Modelmm)) / (Model.GetS0()*2*PerturbationRatio);
        }
};

#endif //OPTIONS08_H
