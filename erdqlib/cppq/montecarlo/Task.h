#ifndef TASK_H
#define TASK_H

#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <cmath>

#include "OUModel.h"
#include "PathDepOption01.h"

using namespace std;

/**
 * @brief Evaluates option pricing using the Black-Scholes model dynamics.
 *
 * This function initializes the Black-Scholes model with given parameters,
 * sets up European and Arithmetic Asian call options, and calculates their
 * prices using Monte Carlo simulations.
 *
 * @return int Returns 0 upon successful execution.
 */
int evaluateWithBlackScholesDynamics()
{
    cout << endl << "Running BSModel dynamics..." << endl;
    double S0 = 100.0, r = 0.03, sigma = 0.2;
    BSModel bsModel(S0, r, sigma);

    double T = 1.0 / 12.0, K = 100.0; // Expiry is 1 month.
    int m = 30;                       // Daily observations for one month!

    long N = 30000;

    ArthmAsianCall AriAsiCallOption(T, K, m);
    cout << "Arithmetic Asian Call Price = " << AriAsiCallOption.PriceByMC(bsModel, N) << endl;

    // lecture 4. Exercise 1
    EuropeanCall EurCallOption(T, K, m);
    cout << "European Call Price = " << EurCallOption.PriceByMC(bsModel, N) << endl;
    EuropeanCall EurPutOption(T, K, m);
    cout << "European Put Price = " << EurPutOption.PriceByMC(bsModel, N) << endl;

    return 0;
}

/**
 * @brief Evaluates option pricing using the Ornstein-Uhlenbeck model dynamics.
 *
 * This function initializes the Ornstein-Uhlenbeck model with given parameters,
 * sets up European and Arithmetic Asian call options, and calculates their
 * prices using Monte Carlo simulations.
 *
 * @return int Returns 0 upon successful execution.
 */
int evaluateWithOrnsteinUhlenbeckDynamics()
{
    cout << endl << "Running OUModel dynamics..." << endl;
    double P0 = 100.0, r=0.03, s0=0.045, drift = 4*std::log(2.0), sigma = 0.2;
    OUModel ouModel(P0, r, s0, drift, sigma);

    double T = 1.0 / 12.0, K = 100.0; // Expiry is 1 month.
    int m = 30;                       // Daily observations for one month!

    long N = 30000;

    ArthmAsianCall AriAsiCallOption(T, K, m);
    cout << "Arithmetic Asian Call Price = " << AriAsiCallOption.PriceByMC(ouModel, N) << endl;

    // lecture 4. Exercise 1
    EuropeanCall EurCallOption(T, K, m);
    cout << "European Call Price = " << EurCallOption.PriceByMC(ouModel, N) << endl;
    EuropeanCall EurPutOption(T, K, m);
    cout << "European Put Price = " << EurPutOption.PriceByMC(ouModel, N) << endl;

    return 0;
}

/**
 * @brief Renders option prices over a range of strike prices.
 *
 * This function iterates over a specified range of strike prices (K) and
 * calculates the present values of European call options using the provided
 * model dynamics. It outputs the results to the console and saves terminal
 * stock prices to a CSV file.
 *
 * @param modelDynamics Reference to the model dynamics (e.g., BSModel or OUModel).
 */
void render_over_KRange(Model& modelDynamics) {
    // double T = 1.0 / 12.0, K = 100.0; // Expiry is 1 month.
    // int m = 30;                       // Daily observations for one month!

    double T = 3.0 / 12.0, K = 100.0; // Expiry is 1 month.
    int m = 13;                       // Daily observations for one month!

    long N = 30000;

    vector<double> StrikeRange;
    vector<double> EuropeanPVs;

    vector<double> Sterminals;
    bool TerminalStockPriceRecorded = false;
    for (double K=97.5; K<=102.5; K+=0.1) {
        StrikeRange.push_back(K);

        EuropeanCall EurCallOption(T, K, m);
        if (!TerminalStockPriceRecorded) {
            EuropeanPVs.push_back(EurCallOption.PriceByMC(modelDynamics, N, Sterminals));
            TerminalStockPriceRecorded = true;
        }
        else {
            EuropeanPVs.push_back(EurCallOption.PriceByMC(modelDynamics, N));
        }
    }

    cout << setw(10) << "K"
         << setw(20) << "PV_Eur"
         << endl;
    for (size_t i=0; i<StrikeRange.size(); i++) {
        cout << setw(10) << StrikeRange[i]
             << setw(20) << EuropeanPVs[i]
             << endl;
    }

    string fname = "Sterminals_" + modelDynamics.toString() +".csv";
    ofstream SterminalsFile("/home/sdr2002/dev/cpp/erdqlib/erdqlib/cppq/out/" + fname);
    ostream_iterator<double> out_it (SterminalsFile,"\n");
    copy(Sterminals.begin(), Sterminals.end(), out_it);
    cout << "Sterminals saved to " + fname << endl;
}

#endif //OUMODEL_H
