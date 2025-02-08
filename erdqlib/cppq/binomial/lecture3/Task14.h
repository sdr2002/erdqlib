#ifndef TASK14_H
#define TASK14_H

#include "../lecture2/BinModel02.h"
#include "BinLattice02.h"
#include "Options09.h"
#include <iostream>
#include <string>
using namespace std;

int display_putoption_tree(BinModel& Model, Put& Option)
{
    if (Model.GetDefaultDynamicsParameters() == 1) return 1;

    // Option.GetInputGridParameters();
    Option.GetDefaultGridParameters();

    const double eurPutPrice = Option.PriceByCRR(Model);
    cout << "European " << to_string(Option) << " price: " << eurPutPrice << endl << endl;

    BinLattice<double> PriceTree("PriceTree");
    BinLattice<bool> StoppingTree("StoppingTree");
    const double amPutPrice = Option.PriceBySnell(Model, PriceTree, StoppingTree);
    cout << "American " << to_string(Option) << " price: " << amPutPrice << endl << endl;

    PriceTree.Display();
    StoppingTree.Display();

    return 0;
}

#endif