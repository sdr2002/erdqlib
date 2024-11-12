#include "BinLattice02.h"
#include "../lecture2/BinModel02.h"
#include "Options09.h"
#include <iostream>
#include <string>
using namespace std;


/* Run PVs over Strikes for Eur/Ame Call/KnockoutCall using BinomialModel dynamics*/
int run_ex3_2_part1()
{
    BinModel Model;

    // if (Model.GetInputDynamicsParameters() == 1) return 1;
    if (Model.GetDefaultDynamicsParameters() == 1) return 1;

    Call CallOption;
    KnockOutCall KnockOutCallOption;

    CallOption.GetDefaultGridParameters();
    KnockOutCallOption.GetDefaultGridParameters();

    vector<double> StrikeRange;
    vector<double> EuropeanPVs;
    vector<double> AmericanPVs;
    vector<double> KoEuropeanPVs;
    vector<double> KoAmericanPVs;
    for (double k = 55.0; k <= 135.0; k += 5.0) {
        CallOption.SetK(k);
        KnockOutCallOption.SetK(k);
        StrikeRange.push_back(k);

        EuropeanPVs.push_back(CallOption.PriceByCRR(Model));
        AmericanPVs.push_back(CallOption.PriceBySnell(Model));

        KoEuropeanPVs.push_back(KnockOutCallOption.PriceByCRR(Model));
        KoAmericanPVs.push_back(KnockOutCallOption.PriceBySnell(Model)); //
        // BinLattice<double> PriceTree("PriceTree");
        // BinLattice<bool> StoppingTree("StoppingTree");
        // KoAmericanPVs.push_back(KnockOutCallOption.PriceBySnell(Model), PriceTree, StoppingTree);

    }

    cout << setw(10) << "K"
         << setw(20) << "PV_Eur"
         << setw(20) << "PV_Ame"
         << setw(20) << "PV_KoEur(Barrier=" << KnockOutCallOption.GetBarrier() << ")"
         << setw(20) << "PV_KoAme(Barrier=" << KnockOutCallOption.GetBarrier() << ")"
         << endl;
    cout << string(90, '-') << endl;
    for (size_t i=0; i< StrikeRange.size(); i++) {
        cout << setw(10) << StrikeRange[i]
             << setw(20) << EuropeanPVs[i]
             << setw(20) << AmericanPVs[i]
             << setw(20) << KoEuropeanPVs[i]
             << setw(20) << KoAmericanPVs[i]
             << endl;
    }

    // const double eurPutPrice = Option.PriceByCRR(Model);
    // cout << "European " << to_string(Option) << " price: " << eurPutPrice << endl << endl;
    //
    // BinLattice<double> PriceTree("PriceTree");
    // BinLattice<bool> StoppingTree("StoppingTree");
    // const double amPutPrice = Option.PriceBySnell(Model, PriceTree, StoppingTree);
    // cout << "American " << to_string(Option) << " price: " << amPutPrice << endl << endl;
    // PriceTree.Display();
    // StoppingTree.Display();

    return 0;
}


/* Run PVs over Strikes for Eur/Ame Call Spread using BinomialModel dynamics*/
int run_ex3_2_part2() {
    BinModel Model;

    // if (Model.GetInputDynamicsParameters() == 1) return 1;
    if (Model.GetDefaultDynamicsParameters() == 1) return 1;

    Call CallOption;
    CallOption.GetDefaultGridParameters();

    CallSpread CallSpreadOption;
    CallSpreadOption.GetDefaultGridParameters();

    vector<double> StrikeRange;
    vector<double> EuropeanPVs;
    vector<double> AmericanPVs;
    vector<double> StrikeLongRange;
    vector<double> StrikeShortRange;
    vector<double> SpreadEuropeanPVs;
    vector<double> SpreadAmericanPVs;
    for (double k = 25.0; k <= 175.0; k += 5.0) {
        CallOption.SetK(k);
        StrikeRange.push_back(k);

        const double ks = k + 30;
        CallSpreadOption.SetK(k, ks);
        StrikeLongRange.push_back(k);
        StrikeShortRange.push_back(ks);

        EuropeanPVs.push_back(CallOption.PriceByCRR(Model));
        AmericanPVs.push_back(CallOption.PriceBySnell(Model));

        SpreadEuropeanPVs.push_back(CallSpreadOption.PriceByCRR(Model));
        SpreadAmericanPVs.push_back(CallSpreadOption.PriceBySnell(Model));
    }

    cout << setw(10) << "K"
         << setw(20) << "PV_Eur"
         << setw(20) << "PV_Ame"
         << setw(10) << "|"
         << setw(10) << "Kl"
         << setw(10) << "Ks"
         << setw(20) << "PV_EurSpread"
         << setw(20) << "PV_AmeSpread"
         << endl;
    cout << string(130, '-') << endl;
    for (size_t i=0; i< StrikeRange.size(); i++) {
        cout << setw(10) << StrikeRange[i]
             << setw(20) << EuropeanPVs[i]
             << setw(20) << AmericanPVs[i]
             << setw(10) << "|"
             << setw(10) << StrikeLongRange[i]
             << setw(10) << StrikeShortRange[i]
             << setw(20) << SpreadEuropeanPVs[i]
             << setw(20) << SpreadAmericanPVs[i]
             << endl;
    }

    return 0;
}

void render_over_S0range(
    BinModel& Model,
    Call& CallOption, KnockOutCall& KnockOutCallOption, CallSpread& CallSpreadOption
) {
    // Render over the S0range
    // TODO diagnose the ugliness of PV and Delta valuation in S0 range calculation compared to K range
    vector<double> S0Range;
    vector<double> EuropeanPVs;
    vector<double> EuropeanDeltas;
    vector<double> AmericanPVs;
    vector<double> KoEuropeanPVs;
    vector<double> KoEuropeanDeltas;
    vector<double> KoAmericanPVs;
    vector<double> KoAmericanDeltas;
    vector<double> SpreadEuropeanPVs;
    vector<double> SpreadEuropeanDeltas;
    vector<double> SpreadAmericanPVs;
    vector<double> SpreadAmericanDeltas;
    for (double s0 = 50.0; s0 <= 250.0; s0 += 5.0) {
        Model.SetS0(s0);
        S0Range.push_back(s0);

        EuropeanPVs.push_back(CallOption.PriceByCRR(Model));
        EuropeanDeltas.push_back(CallOption.DeltaByCRR(Model));
        AmericanPVs.push_back(CallOption.PriceBySnell(Model));

        KoEuropeanPVs.push_back(KnockOutCallOption.PriceByCRR(Model));
        KoEuropeanDeltas.push_back(KnockOutCallOption.DeltaByCRR(Model));
        KoAmericanPVs.push_back(KnockOutCallOption.PriceBySnell(Model));
        KoAmericanDeltas.push_back(KnockOutCallOption.DeltaBySnell(Model));

        SpreadEuropeanPVs.push_back(CallSpreadOption.PriceByCRR(Model));
        SpreadEuropeanDeltas.push_back(CallSpreadOption.DeltaByCRR(Model));
        SpreadAmericanPVs.push_back(CallSpreadOption.PriceBySnell(Model));
        SpreadAmericanDeltas.push_back(CallSpreadOption.DeltaBySnell(Model));
    }

    cout << setw(10) << "S0"
         << setw(20) << "PV_Eur"
         << setw(20) << "Delta_Eur"
         << setw(20) << "PV_Ame"
         << setw(10) << "|"
         << setw(20) << "PV_KoEur"
         << setw(20) << "Delta_KoEur"
         << setw(20) << "PV_KoAme"
         << setw(20) << "Delta_KoAme"
         << setw(10) << "|"
         << setw(20) << "PV_EurSpread"
         << setw(20) << "Delta_EurSpread"
         << setw(20) << "PV_AmeSpread"
         << setw(20) << "Delta_AmeSpread"
         << endl;
    cout << string(250, '-') << endl;
    for (size_t i=0; i< S0Range.size(); i++) {
        cout << setw(10) << S0Range[i]
             << setw(20) << EuropeanPVs[i]
             << setw(20) << EuropeanDeltas[i]
             << setw(20) << AmericanPVs[i]
             << setw(10) << "|"
             << setw(20) << KoEuropeanPVs[i]
             << setw(20) << KoEuropeanDeltas[i]
             << setw(20) << KoAmericanPVs[i]
             << setw(20) << KoAmericanDeltas[i]
             << setw(10) << "|"
             << setw(20) << SpreadEuropeanPVs[i]
             << setw(20) << SpreadEuropeanDeltas[i]
             << setw(20) << SpreadAmericanPVs[i]
             << setw(20) << SpreadAmericanDeltas[i]
             << endl;
    }

    cout << endl << endl;
}

void render_over_Krange(
    BinModel& Model,
    Call& CallOption, KnockOutCall& KnockOutCallOption, CallSpread& CallSpreadOption
) {
    // Render over the Krange
    // TODO add vega after introducing U,D <-> Volt, then theta
    vector<double> KRange;
    vector<double> EuropeanPVs;
    vector<double> EuropeanDeltas;
    vector<double> EuropeanGammas;
    vector<double> EuropeanThetas;
    vector<double> AmericanPVs;
    vector<double> KoEuropeanPVs;
    vector<double> KoEuropeanDeltas;
    vector<double> KoEuropeanGammas;
    vector<double> KoAmericanPVs;
    vector<double> KoAmericanDeltas;
    vector<double> KoAmericanGammas;
    vector<double> SpreadEuropeanPVs;
    vector<double> SpreadEuropeanDeltas;
    vector<double> SpreadEuropeanGammas;
    vector<double> SpreadAmericanPVs;
    vector<double> SpreadAmericanDeltas;
    for (double K = 50.0; K <= 200.0; K += 5.0) {
        CallOption.SetK(K);
        KnockOutCallOption.SetK(K);
        CallSpreadOption.SetK(K, K+35);
        KRange.push_back(K);

        EuropeanPVs.push_back(CallOption.PriceByCRR(Model));
        EuropeanDeltas.push_back(CallOption.DeltaByCRR(Model));
        EuropeanGammas.push_back(CallOption.GammaByCRR(Model));
        EuropeanThetas.push_back(CallOption.ThetaByCRR(Model));
        AmericanPVs.push_back(CallOption.PriceBySnell(Model));

        KoEuropeanPVs.push_back(KnockOutCallOption.PriceByCRR(Model));
        KoEuropeanDeltas.push_back(KnockOutCallOption.DeltaByCRR(Model));
        KoEuropeanGammas.push_back(KnockOutCallOption.GammaByCRR(Model));
        KoAmericanPVs.push_back(KnockOutCallOption.PriceBySnell(Model));
        KoAmericanDeltas.push_back(KnockOutCallOption.DeltaBySnell(Model));
        KoAmericanGammas.push_back(KnockOutCallOption.GammaBySnell(Model));

        SpreadEuropeanPVs.push_back(CallSpreadOption.PriceByCRR(Model));
        SpreadEuropeanDeltas.push_back(CallSpreadOption.DeltaByCRR(Model));
        SpreadEuropeanGammas.push_back(CallSpreadOption.GammaByCRR(Model));
        SpreadAmericanPVs.push_back(CallSpreadOption.PriceBySnell(Model));
        SpreadAmericanDeltas.push_back(CallSpreadOption.DeltaBySnell(Model));
    }

    cout << setw(10) << "K"
         << setw(15) << "PV_Eur"
         << setw(15) << "Delta_Eur"
         << setw(15) << "Gamma_Eur"
         << setw(15) << "Theta_Eur"
         << setw(15) << "PV_Ame"
         << setw(10) << "|"
         << setw(15) << "PV_KoEur"
         << setw(15) << "Delta_KoEur"
         << setw(15) << "Gamma_KoEur"
         << setw(15) << "PV_KoAme"
         << setw(15) << "Delta_KoAme"
         << setw(15) << "Gamma_KoAme"
         << setw(10) << "|"
         << setw(15) << "PV_EurSprd"
         << setw(15) << "Delta_EurSprd"
         << setw(15) << "Gamma_EurSprd"
         << setw(15) << "PV_AmeSprd"
         << setw(15) << "Delta_AmeSprd"
         << endl;
    cout << string(275, '-') << endl;
    for (size_t i=0; i< KRange.size(); i++) {
        cout << setw(10) << KRange[i]
             << setw(15) << EuropeanPVs[i]
             << setw(15) << EuropeanDeltas[i]
             << setw(15) << EuropeanGammas[i]
             << setw(15) << EuropeanThetas[i]
             << setw(15) << AmericanPVs[i]
             << setw(10) << "|"
             << setw(15) << KoEuropeanPVs[i]
             << setw(15) << KoEuropeanDeltas[i]
             << setw(15) << KoEuropeanGammas[i]
             << setw(15) << KoAmericanPVs[i]
             << setw(15) << KoAmericanDeltas[i]
             << setw(15) << KoAmericanGammas[i]
             << setw(10) << "|"
             << setw(15) << SpreadEuropeanPVs[i]
             << setw(15) << SpreadEuropeanDeltas[i]
             << setw(15) << SpreadEuropeanGammas[i]
             << setw(15) << SpreadAmericanPVs[i]
             << setw(15) << SpreadAmericanDeltas[i]
             << endl;
    }

    cout << endl << endl;
}

void reset_objects(
    BinModel& Model,
    Call& CallOption, KnockOutCall& KnockOutCallOption, CallSpread& CallSpreadOption
) {
    Model.GetDefaultDynamicsParameters();
    CallOption.GetDefaultGridParameters();
    KnockOutCallOption.GetDefaultGridParameters();
    CallSpreadOption.GetDefaultGridParameters();
}

/* Run PVs over a range of initial stock prices (S0) for Eur/Ame Call Spread using BinomialModel dynamics*/
int run_ex3_2_part3() {
    BinModel Model;

    Call CallOption;
    KnockOutCall KnockOutCallOption;
    CallSpread CallSpreadOption;

    reset_objects(Model, CallOption, KnockOutCallOption, CallSpreadOption);
    render_over_Krange(Model, CallOption, KnockOutCallOption, CallSpreadOption);

    // reset_objects(Model, CallOption, KnockOutCallOption, CallSpreadOption);
    // render_over_S0range(Model, CallOption, KnockOutCallOption, CallSpreadOption);

    return 0;
}


int main() {
    // run_ex3_2_part1();
    // run_ex3_2_part2();
    run_ex3_2_part3();
}