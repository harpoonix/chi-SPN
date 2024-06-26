# $\chi$-SPN

Code Repository for the paper $\chi$-SPN: Characteristic Interventional Sum-Product Networks for Causal Inference in Hybrid Domains.  
Submitted to UAI 2024.

## Abstract 

Causal inference in hybrid domains, characterized by a mixture of discrete and continuous variables, presents a formidable challenge. We take a step towards this direction and propose Characteristic Interventional Sum-Product Network ($\chi$-SPN) that is capable of estimating interventional distributions in presence of random variables drawn from mixed distributions. $\chi$-SPN uses characteristic functions in the leaves of an interventional SPN (iSPN) thereby providing a unified view for discrete and continuous random variables through the Fourier–Stieltjes transform of the probability measures. A neural network is used to estimate the parameters of the learned iSPN using the intervened data. Our experiments on 3 synthetic heterogeneous datasets suggest that $\chi$-SPN can effectively capture the interventional distributions for both discrete and continuous variables while being expressive and causally adequate. We also show that $\chi$-SPN generalize to multiple interventions while being trained only on a single intervention data.