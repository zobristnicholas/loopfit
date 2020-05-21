#include <math.h>
#include <complex.h>
#include <algorithm>    // std::min_element, std::max_element
#include "ceres_fit.hpp"
#include "eqn_cubic.hpp"
#include "ceres/ceres.h"
using ceres::NumericDiffCostFunction;
using ceres::SizedCostFunction;
using ceres::DENSE_QR;
using ceres::CENTRAL;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace std;

// indices of parameters in their parameter blocks (for easy recognition)
const unsigned int qi = 0;  // pr
const unsigned int qc = 1;
const unsigned int f0 = 2;
const unsigned int xa = 3;
const unsigned int a_sqrt = 0;  // pd
const unsigned int gain0 = 0; // pb
const unsigned int gain1 = 1;
const unsigned int gain2 = 2;
const unsigned int phase0 = 3;
const unsigned int phase1 = 4;
const unsigned int alpha = 0; // pi
const unsigned int gamma_ = 1;
const unsigned int i_offset = 0; // po
const unsigned int q_offset = 1;

// other useful constants
const complex<double> J (0.0, 1.0);
const double PI = 3.141592653589793238463;



// define the generator detuning function (defined by L.J. Swenson et al.: https://doi.org/10.1063/1.4794808)
double detuning(const double f, const bool decreasing, const double pr[], const double pd[]) {
    const double x0 = (f - pr[f0]) / pr[f0];
    if (pd[a_sqrt] == 0.0) {
        return x0;
    };
    const double q0 =  1.0 / (1.0 / pr[qi] + 1.0 / pr[qc]);
    const double y0 = q0 * x0;
    const double a[4] = {-y0 - pow(pd[a_sqrt], 2.0), 1.0, -4.0 * y0, 4.0};
    double x[3];
    int n_roots = eqn_cubic(a, x);
    if (decreasing) {
        return *max_element(x, x + n_roots) / q0;
    }
    else {
        return *min_element(x, x + n_roots) / q0;
    };
};


// define the S21 resonance function
complex<double> resonance(const double f, const bool decreasing, const double pr[], const double pd[]) {
    const double x = detuning(f, decreasing, pr, pd);
    const double qi_qc = pr[qi] * pr[qc];
    return (pr[qc] + 2.0 * J * qi_qc * (x + pr[xa])) / (pr[qi] + pr[qc] + 2.0 * J * qi_qc * x);
};


// define a function to approximate the baseline of the resonance
complex<double> baseline(const double f, const double fm, const double pb[]) {
    const double xm = (f - fm) / fm;
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
complex<double> model(const double f, const double fm, const bool decreasing, const double pr[], const double pd[],
                      const double pb[], const double pi[], const double po[]) {
    return mixer(baseline(f, fm, pb) * resonance(f, decreasing, pr, pd), pi, po);
};


// define a calibration function for external use
complex<double> calibrate(const double f, double& i, double& q, const double fm, const double pb[], const double pi[],
                          const double po[]) {
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


// construct the residual for a numerical jacobian calculation
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


// construct the cost function which evaluates the residual and analytic jacobian at each iteration
class CostFunction : public SizedCostFunction <2, 4, 1, 5, 2, 2> {
    public:
        CostFunction(const double f, const double i, const double q, const double fm, const bool decreasing):
            f_(f), i_(i), q_(q), fm_(fm), decreasing_(decreasing) {}
        virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
            // break out parameters
            const double* pr = parameters[0];
            const double* pd = parameters[1];
            const double* pb = parameters[2];
            const double* pi = parameters[3];
            const double* po = parameters[4];
            // break model up for efficiency with Jacobian calculation
            const complex<double> bl = baseline(f_, fm_, pb);
            const complex<double> res = resonance(f_, decreasing_, pr, pd);
            const complex<double> bl_res = bl * res;
            // equivalent of model(f_, fm_, decreasing_, pr, pd, pb, pi, po)
            const complex<double> z = mixer(bl_res, pi, po);
            // compute residual
            residuals[0] = z.real() - i_;  // real part
            residuals[1] = z.imag() - q_;  // imaginary part
            // compute jacobian
            if (jacobians != NULL) {
                // expressions needed for most blocks
                const double x = detuning(f_, decreasing_, pr, pd);
                const complex<double> a_i_sin_g = pi[alpha] + J * sin(pi[gamma_]);
                const complex<double> t1 = 2.0 * bl * pr[qc] * pr[qi] * a_i_sin_g;
                const complex<double> qc2_xa_i = (2.0 * pr[qc] * pr[xa] + J);
                const complex<double> t2 = -pr[qi] * t1 * qc2_xa_i;
                const complex<double> denom = 2.0 * pr[qc] * pr[qi] * x - J * (pr[qc] + pr[qi]);
                const complex<double> denom_sq = pow(denom, 2.0);
                // compute detuning derivatives
                double dxdqi;
                double dxdqc;
                double dxdf0;
                double dxda_sqrt;
                if (pd[a_sqrt] != 0.0) {
                    const double q0 = 1.0 / (1.0 / pr[qi] + 1.0 / pr[qc]);
                    const double a = pow(pd[a_sqrt], 2.0);
                    const double t3 = (1.0 + 4.0 * pow(q0 * x, 2.0));
                    const double t4 = pow(t3, 2.0);
                    const double t5 = -a * (12.0 * pow(q0 * x, 2.0) + 1.0);
                    const double denom2 = 8.0 * a * q0 * x + t4;
                    dxdqi = t5 / pow(pr[qi], 2.0) / denom2;
                    dxdqc = t5 / pow(pr[qc], 2.0) / denom2;
                    dxdf0 = -f_ * t4 / pow(pr[f0], 2.0) / denom2;
                    dxda_sqrt = 2.0 * pd[a_sqrt] * t3 / q0 / denom2;
                }
                else {
                    dxdqi = 0.0;
                    dxdqc = 0.0;
                    dxdf0 = -f_ / pow(pr[f0], 2.0);
                    dxda_sqrt = 0.0;
                };
                if (jacobians[0] != NULL) {  // pr
                    complex<double> j_qi = (-J * bl * pr[qc] * qc2_xa_i * a_i_sin_g + t2 * dxdqi) / denom_sq;
                    // [parameter block][residual index * parameter block size + parameter index]
                    jacobians[0][0 * 4 + qi] = j_qi.real();
                    jacobians[0][1 * 4 + qi] = j_qi.imag();
                    const complex<double> j_qc = (-bl * pr[qi] * J * a_i_sin_g * (2.0 * pr[qi] * (x + pr[xa]) - J)  +
                                                  t2 * dxdqc) / denom_sq;
                    jacobians[0][0 * 4 + qc] = j_qc.real();
                    jacobians[0][1 * 4 + qc] = j_qc.imag();
                    const complex<double> j_f0 = t2 / denom_sq * dxdf0;
                    jacobians[0][0 * 4 + f0] = j_f0.real();
                    jacobians[0][1 * 4 + f0] = j_f0.imag();
                    const complex<double> j_xa = t1 / denom;
                    jacobians[0][0 * 4 + xa] = j_xa.real();
                    jacobians[0][1 * 4 + xa] = j_xa.imag();
                };
                if (jacobians[1] != NULL) {  // pd
                    const complex<double> j_a_sqrt = t2 * dxda_sqrt / denom_sq;
                    jacobians[1][0 * 1 + a_sqrt] = j_a_sqrt.real();
                    jacobians[1][1 * 1 + a_sqrt] = j_a_sqrt.imag();
                };
                if (jacobians[2] != NULL) {  // pb
                    const complex<double> xm = (f_ - fm_) / fm_;
                    const complex<double> j_gain0 = res * a_i_sin_g * exp(J * (pb[phase0] + pb[phase1] * xm));
                    jacobians[2][0 * 5 + gain0] = j_gain0.real();
                    jacobians[2][1 * 5 + gain0] = j_gain0.imag();
                    const complex<double> j_gain1 = xm * j_gain0;
                    jacobians[2][0 * 5 + gain1] = j_gain1.real();
                    jacobians[2][1 * 5 + gain1] = j_gain1.imag();
                    const complex<double> j_gain2 = xm * j_gain1;
                    jacobians[2][0 * 5 + gain2] = j_gain2.real();
                    jacobians[2][1 * 5 + gain2] = j_gain2.imag();
                    const complex<double> j_phase0 = J * j_gain0 * (pb[gain0] + pb[gain1] * xm +
                                                                    pb[gain2] * pow(xm, 2.0));
                    jacobians[2][0 * 5 + phase0] = j_phase0.real();
                    jacobians[2][1 * 5 + phase0] = j_phase0.imag();
                    const complex<double> j_phase1 = xm * j_phase0;
                    jacobians[2][0 * 5 + phase1] = j_phase1.real();
                    jacobians[2][1 * 5 + phase1] = j_phase1.imag();
                };
                if (jacobians[3] != NULL) {  // pi
                    const complex<double> bl_res_exp = bl_res * exp(J * pi[gamma_]);
                    const complex<double> j_alpha = (conj(bl_res) + bl_res) / 2.0;
                    jacobians[3][0 * 2 + alpha] = j_alpha.real();
                    jacobians[3][1 * 2 + alpha] = j_alpha.imag();
                    const complex<double> j_gamma = J * bl_res_exp.real();
                    jacobians[3][0 * 2 + gamma_] = j_gamma.real();
                    jacobians[3][1 * 2 + gamma_] = j_gamma.imag();
                };
                if (jacobians[4] != NULL) {  // po
                    jacobians[4][0 * 2 + i_offset] = 1.0;
                    jacobians[4][1 * 2 + i_offset] = 0.0;
                    jacobians[4][0 * 2 + q_offset] = 0.0;
                    jacobians[4][1 * 2 + q_offset] = 1.0;
                };
                return true;
            };
        };
    private:
        const double f_, i_, q_, fm_;
        const bool decreasing_;
};


// set up and perform the fit
string fit(const double f[], const double i[], const double q[], const unsigned int data_size, const double fm,
           const bool decreasing, const bool baseline, const bool nonlinear, const bool imbalance, const bool offset,
           const bool numerical, double pr[], double pd[], double pb[], double pi[],
           double po[]) {
    // set up the residual
    Problem problem;
    for (int ii = 0; ii < data_size; ++ii) {
        if (numerical) {
            problem.AddResidualBlock(
                new NumericDiffCostFunction<Residual, CENTRAL, 2, 4, 1, 5, 2, 2>(
                    new Residual(f[ii], i[ii], q[ii], fm, decreasing)
            ),
            NULL, pr, pd, pb, pi, po
        );
        }
        else {
            problem.AddResidualBlock(new CostFunction(f[ii], i[ii], q[ii], fm, decreasing), NULL, pr, pd, pb, pi, po);
        };
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
    options.max_num_iterations = 100;
    options.linear_solver_type = DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    Solver::Summary summary;
    // solve and return output
    Solve(options, &problem, &summary);
    const string output = summary.FullReport() + string("\n");
    return output;
};
