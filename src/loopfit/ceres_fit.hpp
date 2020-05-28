#include <string>
#include <cmath>
#include <complex>
#include <algorithm>    // std::min_element, std::max_element
#include "eqn_cubic.hpp"
#include "ceres/ceres.h"

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
const unsigned int beta = 1;
const unsigned int gamma_ = 0; // po
const unsigned int delta = 1;

// other useful constants
const std::complex<double> J (0.0, 1.0);
const double PI = 3.141592653589793238463;


// define the generator detuning function (defined by L.J. Swenson et al.: https://doi.org/10.1063/1.4794808)
template <class T>
double detuning(const T f, const bool decreasing, const double pr[], const double pd[]) {
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
        return *std::max_element(x, x + n_roots) / q0;
    }
    else {
        return *std::min_element(x, x + n_roots) / q0;
    };
};


// define the S21 resonance function
template <class T>
std::complex<double> resonance(const T f, const bool decreasing, const double pr[], const double pd[]) {
    const double x = detuning(f, decreasing, pr, pd);
    const double qi_qc = pr[qi] * pr[qc];
    return (pr[qc] + 2.0 * J * qi_qc * (x + pr[xa])) / (pr[qi] + pr[qc] + 2.0 * J * qi_qc * x);
};


// define a function to approximate the baseline of the resonance
template <class T>
std::complex<double> baseline(const T f, const double fm, const double pb[]) {
    const double xm = (f - fm) / fm;
    return (pb[gain0] + pb[gain1] * xm + pb[gain2] * pow(xm, 2.0)) * std::exp(J * (pb[phase0] + pb[phase1] * xm));
};


// define the mixer nonlinearity
template <class T>
std::complex<double> mixer(const std::complex<T> z, const double pi[], const double po[]) {
    std::complex<double> result;
    if (pi[alpha] != 1.0 || pi[beta] != 0.0) {
        result = z.real() + J * pi[alpha] * (z.real() * std::sin(pi[beta]) + z.imag() * std::cos(pi[beta]));
    }
    else {
        result = z;
    };
    if (po[gamma_] != 0.0 || po[delta] != 0.0) {
        result += po[gamma_] + J * po[delta];
    };
    return result;
};


// define the objective function from the baseline and resonance
template <class T>
std::complex<double> model(const T f, const double fm, const bool decreasing, const double pr[], const double pd[],
                           const double pb[], const double pi[], const double po[]) {
    return mixer(baseline(f, fm, pb) * resonance(f, decreasing, pr, pd), pi, po);
};


// define a calibration function for external use
template <class T, class U>
void calibrate(const T f, U& i, U& q, const double fm, const double pb[], const double pi[], const double po[]) {
    std::complex<double> z = i + J * q;
    if (po[gamma_] != 0.0 || po[delta] != 0.0) {  // remove offset
        z -= (po[gamma_] + J * po[delta]);
    };
    if (pi[alpha] != 1.0 || pi[beta] != 0.0) {  // remove imbalance
        z = (z.real() + J * (-z.real() * std::tan(pi[beta]) + z.imag() / std::cos(pi[beta]) / pi[alpha]));
    };
    z /= baseline(f, fm, pb);
    i = z.real();
    q = z.imag();
};


// construct the residual for a numerical jacobian calculation
template<class T, class U>
struct Residual {
    Residual(const T f, const U i, const U q, const double fm, const bool decreasing):
        f_(f), i_(i), q_(q), fm_(fm), decreasing_(decreasing) {}
    bool operator()(const double* const pr, const double* const pd, const double* const pb, const double* const pi,
    const double* const po, double* residual) const {
        std::complex<double> z = model(f_, fm_, decreasing_, pr, pd, pb, pi, po);
        residual[0] = i_ - z.real();
        residual[1] = q_ - z.imag();
        return true;
    };
 private:
    const T f_;
    const U i_;
    const U q_;
    const double fm_;
    const bool decreasing_;
};


// construct the cost function which evaluates the residual and analytic Jacobian at each iteration
template<class T, class U>
class CostFunction : public ceres::SizedCostFunction <2, 4, 1, 5, 2, 2> {
    public:
        CostFunction(const T f, const U i, const U q, const double fm, const bool decreasing):
            f_(f), i_(i), q_(q), fm_(fm), decreasing_(decreasing) {}
        virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
            // break out parameters
            const double* pr = parameters[0];
            const double* pd = parameters[1];
            const double* pb = parameters[2];
            const double* pi = parameters[3];
            const double* po = parameters[4];
            // break model up for efficiency with Jacobian calculation
            const std::complex<double> bl = baseline(f_, fm_, pb);
            const std::complex<double> res = resonance(f_, decreasing_, pr, pd);
            const std::complex<double> bl_res = bl * res;
            // equivalent of model(f_, fm_, decreasing_, pr, pd, pb, pi, po)
            const std::complex<double> z = mixer(bl_res, pi, po);
            // compute residual
            residuals[0] = z.real() - i_;  // real part
            residuals[1] = z.imag() - q_;  // imaginary part
            // compute Jacobian
            if (jacobians != NULL) {
                // expressions needed for most blocks
                const double x = detuning(f_, decreasing_, pr, pd);
                const std::complex<double> t1 = 2.0 * bl * pr[qc] * pr[qi];
                const std::complex<double> qc2_xa_i = (2.0 * pr[qc] * pr[xa] + J);
                const std::complex<double> t2 = -pr[qi] * t1 * qc2_xa_i;
                const std::complex<double> denom = 2.0 * pr[qc] * pr[qi] * x - J * (pr[qc] + pr[qi]);
                const std::complex<double> denom_sq = pow(denom, 2.0);
                const double dpo[2] = {0.0, 0.0};
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
                    const std::complex<double> dzdqi = (-J * bl * pr[qc] * qc2_xa_i + t2 * dxdqi) / denom_sq;
                    const std::complex<double> j_qi = mixer(dzdqi, pi, dpo);
                    // [parameter block][residual index * parameter block size + parameter index]
                    jacobians[0][0 * 4 + qi] = j_qi.real();
                    jacobians[0][1 * 4 + qi] = j_qi.imag();
                    const std::complex<double> dzdqc = (-J * bl * pr[qi] * (2.0 * pr[qi] * (x + pr[xa]) - J) +
                                                        t2 * dxdqc) / denom_sq;
                    const std::complex<double> j_qc = mixer(dzdqc, pi, dpo);
                    jacobians[0][0 * 4 + qc] = j_qc.real();
                    jacobians[0][1 * 4 + qc] = j_qc.imag();
                    const std::complex<double> dzdf0 = t2 / denom_sq * dxdf0;
                    const std::complex<double> j_f0 = mixer(dzdf0, pi, dpo);
                    jacobians[0][0 * 4 + f0] = j_f0.real();
                    jacobians[0][1 * 4 + f0] = j_f0.imag();
                    const std::complex<double> dzdxa = t1 / denom;
                    const std::complex<double> j_xa = mixer(dzdxa, pi, dpo);
                    jacobians[0][0 * 4 + xa] = j_xa.real();
                    jacobians[0][1 * 4 + xa] = j_xa.imag();
                };
                if (jacobians[1] != NULL) {  // pd
                    const std::complex<double> dzdasqrt = t2 * dxda_sqrt / denom_sq;
                    const std::complex<double> j_a_sqrt = mixer(dzdasqrt, pi, dpo);
                    jacobians[1][0 * 1 + a_sqrt] = j_a_sqrt.real();
                    jacobians[1][1 * 1 + a_sqrt] = j_a_sqrt.imag();
                };
                if (jacobians[2] != NULL) {  // pb
                    const std::complex<double> xm = (f_ - fm_) / fm_;
                    const std::complex<double> dzdgain0 = res * std::exp(J * (pb[phase0] + pb[phase1] * xm));
                    const std::complex<double> j_gain0 = mixer(dzdgain0, pi, dpo);
                    jacobians[2][0 * 5 + gain0] = j_gain0.real();
                    jacobians[2][1 * 5 + gain0] = j_gain0.imag();
                    const std::complex<double> j_gain1 = xm * j_gain0;
                    jacobians[2][0 * 5 + gain1] = j_gain1.real();
                    jacobians[2][1 * 5 + gain1] = j_gain1.imag();
                    const std::complex<double> j_gain2 = xm * j_gain1;
                    jacobians[2][0 * 5 + gain2] = j_gain2.real();
                    jacobians[2][1 * 5 + gain2] = j_gain2.imag();
                    const std::complex<double> dzdphase0 = J * dzdgain0 * (pb[gain0] + pb[gain1] * xm +
                                                                           pb[gain2] * pow(xm, 2.0));
                    const std::complex<double> j_phase0 = mixer(dzdphase0, pi, dpo);
                    jacobians[2][0 * 5 + phase0] = j_phase0.real();
                    jacobians[2][1 * 5 + phase0] = j_phase0.imag();
                    const std::complex<double> j_phase1 = xm * j_phase0;
                    jacobians[2][0 * 5 + phase1] = j_phase1.real();
                    jacobians[2][1 * 5 + phase1] = j_phase1.imag();
                };
                if (jacobians[3] != NULL) {  // pi
                    const std::complex<double> bl_res_exp = bl_res * std::exp(J * pi[beta]);
                    jacobians[3][0 * 2 + alpha] = 0.0;
                    jacobians[3][1 * 2 + alpha] = bl_res_exp.imag();
                    jacobians[3][0 * 2 + beta] = 0.0;
                    jacobians[3][1 * 2 + beta] = pi[alpha] * bl_res_exp.real();
                };
                if (jacobians[4] != NULL) {  // po
                    jacobians[4][0 * 2 + gamma_] = 1.0;
                    jacobians[4][1 * 2 + gamma_] = 0.0;
                    jacobians[4][0 * 2 + delta] = 0.0;
                    jacobians[4][1 * 2 + delta] = 1.0;
                };
                return true;
            };
        };
    private:
        const T f_;
        const U i_, q_;
        const double fm_;
        const bool decreasing_;
};


template<class T, class U>
std::string fit(const T f[], const U i[], const U q[], const unsigned int data_size, const double fm,
                const bool decreasing, const bool baseline, const bool nonlinear, const bool imbalance,
                const bool offset, const bool numerical, const int max_iterations, int& threads, int& varied,
                bool& success, double pr[], double pd[], double pb[], double pi[], double po[]) {
    // set up the residual
    ceres::Problem problem;
    for (int ii = 0; ii < data_size; ++ii) {
        if (numerical) {
            problem.AddResidualBlock(
                new ceres::NumericDiffCostFunction<Residual<T, U>, ceres::CENTRAL, 2, 4, 1, 5, 2, 2>(
                    new Residual<T, U>(f[ii], i[ii], q[ii], fm, decreasing)
            ),
            NULL, pr, pd, pb, pi, po
        );
        }
        else {
            problem.AddResidualBlock(new CostFunction<T, U>(f[ii], i[ii], q[ii], fm, decreasing),
                                     NULL, pr, pd, pb, pi, po);
        };
    };
    // set up parameter bounds (parameter, index in parameter, value)
    problem.SetParameterLowerBound(pr, qi, 1.0);
    problem.SetParameterLowerBound(pr, qc, 1.0);
    problem.SetParameterLowerBound(pr, f0, *std::min_element(f, f + data_size));
    problem.SetParameterUpperBound(pr, f0, *std::max_element(f, f + data_size));
    problem.SetParameterLowerBound(pb, gain0, 0.0);
    problem.SetParameterLowerBound(pi, alpha, 0.0);
    problem.SetParameterLowerBound(pi, beta, pi[beta] - PI / 2.0);
    problem.SetParameterUpperBound(pi, beta, pi[beta] + PI / 2.0);
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
    ceres::Solver::Options options;
    options.num_threads = threads;
    options.max_num_iterations = max_iterations;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    // solve and return output
    ceres::Solve(options, &problem, &summary);
    threads = summary.num_threads_used;
    varied = summary.num_effective_parameters_reduced;
    success = (summary.termination_type == ceres::CONVERGENCE || summary.termination_type == ceres::USER_SUCCESS);
    const std::string output = summary.FullReport() + std::string("\n");
    return output;
};

