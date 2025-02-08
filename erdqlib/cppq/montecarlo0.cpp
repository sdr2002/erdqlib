#include "montecarlo/OUModel.h"
#include "montecarlo/Task.h"
/**
* @brief Entry point of the application.
 *
 * This function initializes the Ornstein-Uhlenbeck and Black-Scholes models,
 * and invokes the `render_over_KRange` function to calculate and display
 * option prices over a range of strike prices.
 *
 * @return int Returns 0 upon successful execution.
 */
int main() {
    // evaluateWithBlackScholesDynamics();
    // evaluateWithOrnsteinUhlenbeckDynamics();

    OUModel ouModel(100.0, 0.03, 0.045, 12*log(2), 0.2);
    render_over_KRange(ouModel);

    BSModel bsModel(100.0, 0.03, 0.2);
    render_over_KRange(bsModel);
}