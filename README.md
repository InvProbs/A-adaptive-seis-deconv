# Adaptive LU at inference
This repository provides the implementation of our novel adaptive loop unrolling method designed to tackle the seismic deconvolution problem with forward model mismatch. The paper will be availabe here: 

## Overview
In seismic deconvolution, inaccuracies in the source wavelet can degrade the performance of loop unrolling (LU) methods trained on correct wavelets. Retraining models for each scenario is computationally expensive and impractical in many real-world applications. To address this issue, our method introduces inference-time adaptation, which corrects forward model mismatches without requiring retraining.

## Features
* Implements UNet and LU training.
* Provide code for inference time adaptive LU.

## Installation
clone the repository
~~~
git clone https://github.com/InvProbs/adaptive_seismic_deconv
~~~

Install the required dependencies
~~~
pip install -r requirements.txt
~~~

Training and evaluation code can be found under /solvers/
