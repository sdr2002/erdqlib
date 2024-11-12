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

/* Run PVs over a range of initial stock prices (S0) for Eur/Ame Call Spread using BinomialModel dynamics*/
int run_ex3_2_part3() {
    BinModel Model;

    // if (Model.GetInputDynamicsParameters() == 1) return 1;
    if (Model.GetDefaultDynamicsParameters() == 1) return 1;

    Call CallOption;
    CallOption.GetDefaultGridParameters();

    KnockOutCall KnockOutCallOption;
    KnockOutCallOption.GetDefaultGridParameters();

    CallSpread CallSpreadOption;
    CallSpreadOption.GetDefaultGridParameters();

    vector<double> S0Range;
    vector<double> EuropeanPVs;
    vector<double> AmericanPVs;
    vector<double> KoEuropeanPVs;
    vector<double> KoAmericanPVs;
    vector<double> SpreadEuropeanPVs;
    vector<double> SpreadAmericanPVs;
    for (double s0 = 50.0; s0 <= 200.0; s0 += 5.0) {
        Model.SetS0(s0);
        S0Range.push_back(s0);

        EuropeanPVs.push_back(CallOption.PriceByCRR(Model));
        AmericanPVs.push_back(CallOption.PriceBySnell(Model));

        KoEuropeanPVs.push_back(KnockOutCallOption.PriceByCRR(Model));
        KoAmericanPVs.push_back(KnockOutCallOption.PriceBySnell(Model));

        SpreadEuropeanPVs.push_back(CallSpreadOption.PriceByCRR(Model));
        SpreadAmericanPVs.push_back(CallSpreadOption.PriceBySnell(Model));
    }

    cout << setw(10) << "S0"
         << setw(20) << "PV_Eur"
         << setw(20) << "PV_Ame"
         << setw(10) << "|"
         << setw(20) << "PV_KoEur"
         << setw(20) << "PV_KoAme"
         << setw(10) << "|"
         << setw(20) << "PV_EurSpread"
         << setw(20) << "PV_AmeSpread"
         << endl;
    cout << string(130, '-') << endl;
    for (size_t i=0; i< S0Range.size(); i++) {
        cout << setw(10) << S0Range[i]
             << setw(20) << EuropeanPVs[i]
             << setw(20) << AmericanPVs[i]
             << setw(10) << "|"
             << setw(20) << KoEuropeanPVs[i]
             << setw(20) << KoAmericanPVs[i]
             << setw(10) << "|"
             << setw(20) << SpreadEuropeanPVs[i]
             << setw(20) << SpreadAmericanPVs[i]
             << endl;
    }

    return 0;
}


int main() {
    // run_ex3_2_part1();
    // run_ex3_2_part2();
    run_ex3_2_part3();
}