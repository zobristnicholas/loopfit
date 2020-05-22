#include<string>
#include<complex>

// double detuning(const double f, const bool decreasing, const double pr[], const double pd[]);

std::complex<double> resonance(const double f, const bool decreasing, const double pr[], const double pd[]);

std::complex<double> baseline(const double f, const double fm, const double pb[]);

std::complex<double> model(const double f, const double fm, const bool decreasing, const double pr[], const double pd[],
                           const double pb[], const double pi[], const double po[]);

std::complex<double> calibrate(const double f, double& i, double& q, const double fm, const double pb[],
                               const double pi[], const double po[]);

std::string fit(const double f[], const double i[], const double q[], const unsigned int data_size, const double fm,
                const bool decreasing, const bool baseline, const bool nonlinear, const bool imbalance,
                const bool offset, const bool numerical, double pr[], double pd[], double pb[], double pi[],
                double po[]);
