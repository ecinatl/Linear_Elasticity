# Linear_Elasticity

Author: Emma Cinatl, Clemson University, 2018

Based on the deal.II step-8 tutorial program

This code accompanies the thesis "Finite Element Discretizations for Linear Elasticity" by Emma Cinatl.  It relies on the deal.II open-source finite element library, and is based on the deal.II step-8 tutorial program (https://www.dealii.org/).

The pure displacement form of the linear elasticity equations is solved in reduced_integration_timoshenko.cc and reduced_integration_manufactured_soln.cc, and the displacement-pressure form of the linear elasticity equations is solved in  disp_pressure_manufactured_soln.cc and disp_pressure_timoshenko.cc.

Several adjustable parameters are in each .cc file.  They are
(1) int fe_degree: Reflects the underlying finite element used (e.g. Q1 or Q2)
(2) int r: Quadrature used on the \int_{\Omega} \lambda div u div v term of the pure displacement linear elasticity equations
(3) int cycles: Number of refinements
(4) double nu, double mu_scalar, double lamb: Material parameters

Adaptive refinement is also available in each file (use the refine_grid function instead of refine_global).

