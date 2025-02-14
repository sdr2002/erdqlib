#ifndef BinModel2_h
#define BinModel2_h

// TODO introduce U,D <-> Volatility and R <-> exp(r*dt) conversion
// TODO introduce per-annum concept to U,D,R as well as for Volatility

class BinModel
{
private:
    double S0; // Initial stock price
    double U;  // Up factor
    double D;  // Down factor
    double R;  // Risk-free rate

    // Shared method to check parameters
    int checkData();

public:
    // Constructor
    BinModel();

    // Method to compute the risk-neutral probability
    double RiskNeutProb();

    // Method to compute stock price at node (n, i)
    double S(int n, int i);

    // Method to get input data from the user
    int GetInputDynamicsParameters();

    // Method to set default values
    int GetDefaultDynamicsParameters();

    // Getters and Setters
    double GetS0();
    void SetS0(double S0new);
    double GetR();
};

#endif
