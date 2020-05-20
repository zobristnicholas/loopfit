#include<complex>
#include<string>

std::complex<double> resonance(double f, bool decreasing, const double pr[], const double pd[]);

std::complex<double> baseline(double f, double fm, const double pb[]);

std::complex<double> model(double f, double fm, bool decreasing, const double pr[], const double pd[],
const double pb[], const double pi[], const double po[]);

std::complex<double> calibrate(double f, double& i, double& q, double fm, double pb[], double pi[], double po[]);

std::string fit(double f[], double i[], double q[], unsigned int data_size, double fm, bool decreasing, bool baseline,
bool nonlinear, bool imbalance, bool offset, double pr[], double pd[], double pb[], double pi[], double po[]);
