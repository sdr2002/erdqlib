#include "BinLattice02.h"
#include "../lecture2/BinModel02.h"
#include "Options09.h"
#include <iostream>
#include <string>
using namespace std;


int main()
{
    BinModel Model;

    // if (Model.GetInputGridParameters() == 1) return 1;
    if (Model.GetDefaultGridParameters() == 1) return 1;

    KnockOutCall KnockOutCallOption;
    Call CallOption;

    // Option.GetInputGridParameters();
    KnockOutCallOption.GetDefaultGridParameters();
    CallOption.GetDefaultGridParameters();

    vector<double> StrikeRange;
    vector<double> KoEuropeanPVs;
    vector<double> KoAmericanPVs;
    vector<double> EuropeanPVs;
    vector<double> AmericanPVs;
    for (double k = 55.0; k <= 135.0; k += 5.0) {
        KnockOutCallOption.SetK(k);
        CallOption.SetK(k);
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
