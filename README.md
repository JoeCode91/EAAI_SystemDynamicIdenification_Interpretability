This repository corresponds to the simulations developed for the Journal Article 
"Retrieving Interpretability to Support Vector Machine Regression Models in Dynamic 
Systems Identification" Each file responds to a case proposed in the manuscript

ABSTRACT
Black-box models are commonly used to identify dynamic systems, however, due to its 
nature, it is difficult to understand what the model actually does to the input data. 
Therefore, interpretability of black-box models is recently gaining attention in the 
community. In this paper, an algorithm to decompose the output of a nonlinear model 
is proposed. The algorithm uses a support vector machine (SVM) regression, and it 
decomposes its output into an additive model using nonlinear oblique subspace 
projections (NObSP). NObSP was originally developed for static models using Least-
Squares SVM (LS-SVM) regression, and it aims to improve the model interpretability 
by decomposing its output into an additive model. Each component of the decomposition 
represents the partial (non)linear contribution of each input variable on the output. 
In this paper, the contribution is two-fold, on the one hand, an extension for NObSP 
for dynamic systems is presented. On the other hand, an out-of-sample extension to 
reduce its computational complexity is proposed. The arithmetic complexity for NObSP 
is changed from O(N^3), being N the number of observations, to O (Nd^2), where d is 
the number of support vectors. Several simulations were performed considering a Wiener 
and a Hammerstein structure of the system to identify. It is shown that NObSP is able 
to find the nonlinear contribution of each input variable on the output, and its 
computational complexity can be reduced. Additionally, the simulations indicate that 
NObSP is able to retrieve the nonlinear contribution of each input variable on the 
output. Furthermore, we show that NObSP is robust to changes on the input data, 
correlated inputs and low SNR.
