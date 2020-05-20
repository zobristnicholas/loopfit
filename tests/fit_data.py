import loopfit
import numpy as np
from matplotlib import pyplot as plt


f = np.linspace(4, 4 + 0.002, 1000)
true_params = {'qi': 100000, 'qc': 30000, 'f0': np.median(f), 'xa': 5e-6, 'a': 0.4, 'alpha': 1.2, 'gamma': 0.2,
               'gain0': 3.0, 'gain1': 1000.0, 'gain2': 50000.0, 'phase0': 2.0, 'phase1': -1000.0}

loop = loopfit.model(f, **true_params)
loop += np.random.normal(0, 0.02, loop.shape) + 1j * np.random.normal(0, 0.02, loop.shape)

nonlinear = True
init_params = loopfit.guess(f, loop.real, loop.imag, nonlinear=nonlinear, alpha=1.1, gamma=0.14)
result = loopfit.fit(f, loop.real, loop.imag, baseline=True, nonlinear=nonlinear, imbalance=True, **init_params)

print("True:", true_params)
print("Init:", init_params)
print("Fit:", result)

plt.figure()
plt.title("Data Coordinates")
plt.plot(loop.real, loop.imag)

model = loopfit.model(f, **true_params)
plt.plot(model.real, model.imag, label='true')

model_init = loopfit.model(f, **init_params)
plt.plot(model_init.real, model_init.imag, label='init')

model_fit = loopfit.model(f, **result)
plt.plot(model_fit.real, model_fit.imag, label='fit')

plt.axis('equal')
plt.legend()
plt.show(block=False)

plt.figure()
plt.title("Fit-Calibrated Coordinates")

loop_calibrated = loopfit.calibrate(f, z=loop, **result)
plt.plot(loop_calibrated.real, loop_calibrated.imag)

model_calibrated = loopfit.calibrate(f, z=model, **result)
plt.plot(model_calibrated.real, model_calibrated.imag, label='true')

model_init_calibrated = loopfit.calibrate(f, z=model_init, **result)
plt.plot(model_init_calibrated.real, model_init_calibrated.imag, label='init')

model_fit_calibrated_i,  model_fit_calibrated_q = loopfit.calibrate(f, i=model_fit.real, q=model_fit.imag, **result)
plt.plot(model_fit_calibrated_i, model_fit_calibrated_q, label='fit')

plt.axis('equal')
plt.legend()
plt.show()
