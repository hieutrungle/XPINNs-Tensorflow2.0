# Unofficial Implementation of Extended physics-informed neural networks (XPINNs) : A generalized space-time domain decomposition based deep learning framework for nonlinear partial differential equations

Recommended software versions: TensorFlow 2.8, Python 3.9


**Setup**

```
$ https://github.com/hieutrungle/XPINNs-Tensorflow2.0
$ cd XPINNs-Tensorflow2.0
$ pip install -r requirements.txt
```

**Run Deomo**
```
python xpinn_tf2.py
```

References: For Domain Decomposition based PINN framework

1. A.D.Jagtap, G.E.Karniadakis, Extended Physics-Informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear Partial Differential Equations, Commun. Comput. Phys., Vol.28, No.5, 2002-2041, 2020. (https://doi.org/10.4208/cicp.OA-2020-0164)


**Contribution**

* Your issues and PRs are always welcome.

**Author**

* Hieu Le
* Author Email : hieu.tg.le@gmail.com

**License**

* [GPL-3.0 License](https://github.com/hieutrungle/XPINNs-Tensorflow2.0/blob/main/LICENSE)


<p align="center">
Results of XPINNS after training on Poisson Equation f = x + y.
</p>

<div align="center">
<img src="https://github.com/hieutrungle/XPINNs-Tensorflow2.0/blob/main/xpinn_tf2_figures/exact_solution.png" alt="exact solution" height="650" width="700"/>
<div>
<br />          
<div align="center">
<img src="https://github.com/hieutrungle/XPINNs-Tensorflow2.0/blob/main/xpinn_tf2_figures/predicted_solution.png" alt="predicted solution" height="650" width="700"/>
<div>
<br />          
<div align="center">
<img src="https://github.com/hieutrungle/XPINNs-Tensorflow2.0/blob/main/xpinn_tf2_figures/point_wise_error.png" alt="point-wise error" height="650" width="700"/>
<div>

