# EOSC595: Directed Study  
## Induced Polarization Inversion for Seafloor Hydrothermal Deposit Rock Sample  
**Masayuki Motoori**

## Overview

Seafloor hydrothermal deposits are polymetallic massive sulfide ore deposits formed by the precipitation of metal components from hot water ejected from the seafloor (JOGMEC, 2020). These deposits are commonly located at ocean depths between 700 and 2000 meters. The ore bodies may span hundreds of meters horizontally and tens of meters vertically, and are often exposed on the seafloor surface (Morozumi et al., 2020).

Lab-based petrophysical studies suggest that **resistivity** and **chargeability** are key physical properties that distinguish hydrothermal deposits. These deposits tend to be more conductive and more chargeable than seawater and the surrounding host rock (Nakayama et al., 2012). However, relatively few studies have investigated spectral IP parameters such as the **time constant** and **exponent \( C \)**.

This project is a synthetic modeling and inversion study of induced polarization (IP), applying techniques from the CPSC340: Machine Learning course. It focuses on implementing IP inversion models in PyTorch to explore both spectral and time domain approaches.

## Objectives

This repository presents:

- Differentiable modeling for IP inversion using PyTorch and automatic differentiation.
- Efficient computation of the Jacobian and gradient of the objective function.
- Implementation of both Pelton and Cole-Cole models, which account for:
  - Resistivity
  - Chargeability
  - Time constant
  - Exponent \( C \)
- A proposed Linearly Weighted Debye model that does not rely on exponent \( C \).
- An objective function that includes L1 regularization for feature selection.
- Simulation and inversion of spectral and time-domain IP using FFT-based convolution within an autodiff framework.

## Repository

Code and materials are available at:  
[https://github.com/MasayukiMotoori/EOSC595](https://github.com/MasayukiMotoori/EOSC595)

## References

- Haber, E. (2014). *Computational Methods in Geophysical Electromagnetics*. Society for Industrial and Applied Mathematics. https://doi.org/10.1137/1.9781611973808

- JOGMEC. (n.d.). *Conducts World’s First Successful Excavation of Cobalt-Rich Seabed in the Deep Ocean*. Retrieved April 3, 2025, from https://www.jogmec.go.jp/english/news/release/news_01_000033.html

- Morozumi, H., Watanabe, K., Sakurai, H., Hino, H., Kado, Y., Motoori, M., & Tendo, H. (2020). *Additional information for characteristics of seafloor hydrothermal deposits investigated by JOGMEC*. Shigen-Chishitsu, 70(2), 113–119. https://doi.org/10.11456/shigenchishitsu.70.113

- Nakayama, K., Yamashita, Y., Yasui, M., Yamazaki, A., & Saito, A. (2012). *Electric and magnetic properties of the sea-floor hydrothermal mineral ore deposits for the marine EM explorations*. Proceedings of the SEGJ Conference, 126, 162–165.

- Tarasov, A., & Titov, K. (2013). *On the use of the Cole–Cole equations in spectral induced polarization*. Geophysical Journal International, 195(1), 352–356. https://doi.org/10.1093/gji/ggt251
