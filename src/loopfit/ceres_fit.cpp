#include <math.h>
#include <complex.h>
#include <algorithm>    // std::min_element, std::max_element
#include "ceres_fit.hpp"
#include "eqn_cubic.hpp"
#include "ceres/ceres.h"
using ceres::NumericDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace std;

// indices of parameters in their parameter blocks (for easy recognition)
const unsigned int gain0 = 0; // pb
const unsigned int gain1 = 1;
const unsigned int gain2 = 2;
const unsigned int phase0 = 3;
const unsigned int phase1 = 4;
const unsigned int qi = 0;  // pr
const unsigned int qc = 1;
const unsigned int f0 = 2;
const unsigned int xa = 3;
const unsigned int a_sqrt = 0;  // pd
const unsigned int alpha = 0; // pi
const unsigned int gamma_ = 1;
const unsigned int i_offset = 0; // po
const unsigned int q_offset = 1;

// other useful constants
const complex<double> J(0.0, 1.0);
const double PI = 3.141592653589793238463;



// define the generator detuning function
double detuning(double f, bool decreasing, const double pr[], const double pd[]) {
    double x0 = (f - pr[f0]) / pr[f0];
    if (pd[a_sqrt] == 0.0) {
        return x0;
    };
    double q0 =  1.0 / (1.0 / pr[qi] + 1.0 / pr[qc]);
    double y0 = q0 * x0;
    double a[4] = {-y0 - pow(pd[a_sqrt], 2.0), 1.0, -4.0 * y0, 4.0};
    double x[3];
    int n_roots = eqn_cubic(a, x);
    if (decreasing) {
        return *max_element(x, x + n_roots) / q0;
    }
    else {
        return *min_element(x, x + n_roots) / q0;
    };
};


// define the resonance function
complex<double> resonance(double f, bool decreasing, const double pr[], const double pd[]) {
    double x = detuning(f, decreasing, pr, pd);
    double qi_qc = pr[qi] * pr[qc];
    return (pr[qc] + 2.0 * J * qi_qc * (x + pr[xa])) / (pr[qi] + pr[qc] + 2.0 * J * qi_qc * x);
};


// define a function to approximate the baseline of the resonance
complex<double> baseline(double f, double fm, const double pb[]) {
    double xm = (f - fm) / fm;
    return (pb[gain0] + pb[gain1] * xm + pb[gain2] * pow(xm, 2.0)) * exp(J * (pb[phase0] + pb[phase1] * xm));
};


// define the mixer nonlinearity
complex<double> mixer(complex<double> z, const double pi[], const double po[]) {
    if (pi[alpha] != 1.0 || pi[gamma_] != 0.0) {
        z = (z.real() * pi[alpha] + J * (z.real() * sin(pi[gamma_]) + z.imag() * cos(pi[gamma_])));
    };
    if (po[i_offset] != 0.0 || po[q_offset] != 0.0) {
        z += po[i_offset] + J * po[q_offset];
    };
    return z;
};


// define the objective function from the baseline and resonance
complex<double> model(double f, double fm, bool decreasing, const double pr[], const double pd[], const double pb[],
const double pi[], const double po[]) {
    return mixer(baseline(f, fm, pb) * resonance(f, decreasing, pr, pd), pi, po);
};


// define a calibration function for external use
complex<double> calibrate(double f, double& i, double& q, double fm, double pb[], double pi[], double po[]) {
    complex<double> z = i + J * q;
    if (po[i_offset] != 0.0 || po[q_offset] != 0.0) {  // remove offset
        z -= (po[i_offset] + J * po[q_offset]);
    };
    if (pi[alpha] != 1.0 || pi[gamma_] != 0.0) {  // remove imbalance
        z = (z.real() / pi[alpha] + J * (-z.real() * tan(pi[gamma_]) / pi[alpha] + z.imag() / cos(pi[gamma_])));
    };
    z /= baseline(f, fm, pb);
    i = z.real();
    q = z.imag();
};


// construct the residual
struct Residual {
    Residual(const double f, const double i, const double q, const double fm, const bool decreasing):
    f_(f), i_(i), q_(q), fm_(fm), decreasing_(decreasing) {}
    bool operator()(const double* const pr, const double* const pd, const double* const pb, const double* const pi,
    const double* const po, double* residual) const {
        complex<double> z = model(f_, fm_, decreasing_, pr, pd, pb, pi, po);
        residual[0] = i_ - z.real();
        residual[1] = q_ - z.imag();
        return true;
    };
 private:
    const double f_;
    const double i_;
    const double q_;
    const double fm_;
    const bool decreasing_;
};


// set up and perform the fit
string fit(double f[], double i[], double q[], unsigned int data_size, double fm, bool decreasing, bool baseline,
bool nonlinear, bool imbalance, bool offset, double pr[], double pd[], double pb[], double pi[], double po[]) {
    // set up residual
    Problem problem;
    for (int ii = 0; ii < data_size; ++ii) {
        problem.AddResidualBlock(
            new NumericDiffCostFunction<Residual, ceres::FORWARD, 2, 4, 1, 5, 2, 2>(
                new Residual(f[ii], i[ii], q[ii], fm, decreasing)
            ),
            NULL, pr, pd, pb, pi, po
        );
    };
    // set up parameter bounds (parameter, index in parameter, value)
    problem.SetParameterLowerBound(pr, qi, 1.0);
    problem.SetParameterLowerBound(pr, qc, 1.0);
    problem.SetParameterLowerBound(pr, f0, *min_element(f, f + data_size));
    problem.SetParameterUpperBound(pr, f0, *max_element(f, f + data_size));
    problem.SetParameterLowerBound(pb, gain0, 0.0);
    problem.SetParameterLowerBound(pi, gamma_, pi[gamma_] - PI / 2.0);
    problem.SetParameterUpperBound(pi, gamma_, pi[gamma_] + PI / 2.0);
    // fix some parameters to be constants
    if (!baseline) {  // don't fit baseline
        problem.SetParameterBlockConstant(pb);
    };
    if (!nonlinear) {  // don't fit nonlinear parameters
        problem.SetParameterBlockConstant(pd);
    };
    if (!imbalance) {  // don't fit mixer imbalance parameters
        problem.SetParameterBlockConstant(pi);
    }
    if (!offset) {  // don't fit mixer offset parameters
        problem.SetParameterBlockConstant(po);
    };
    // set up solver
    Solver::Options options;
    options.max_num_iterations = 50;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    Solver::Summary summary;
    // solve and return output
    Solve(options, &problem, &summary);
    string output;
    output = summary.FullReport() + string("\n");
    return output;
};
