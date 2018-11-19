/* ---------------------------------------------------------------------
*
* Copyright (C) 2000 - 2016 by the deal.II authors
*
* This file is part of the deal.II library.
*
* The deal.II library is free software; you can use it, redistribute
* it, and/or modify it under the terms of the GNU Lesser General
* Public License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
* The full text of the license can be found in the file LICENSE at
* the top level of the deal.II distribution.
*
* ---------------------------------------------------------------------

*
* Author: Wolfgang Bangerth, University of Heidelberg, 2000
*/



#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/base/symmetric_tensor.h>

#include <fstream>
#include <iostream>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/base/timer.h>

double lamb =100.; //global variable
int red_quad = 0;    //global variable
bool iterative = false; //global variable
double nu = 0.25;
double mu_scalar = lamb/(2*nu)-lamb;

bool mixed = true;

namespace Step8
{
using namespace dealii;


template <int dim>
class ElasticProblem
{
public:
  ElasticProblem (int fe_degree);
  ~ElasticProblem ();
  void run (int fe_degree);

private:
  void setup_system ();
  void assemble_system (int fe_degree);
  double solve ();
  void refine_grid ();
  void output_results (const unsigned int cycle,int fe_degree, double solver_effort) const;

  Triangulation<dim>   triangulation;
  DoFHandler<dim>      dof_handler;

  FESystem<dim>        fe;

  ConstraintMatrix     hanging_node_constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double>       solution;
  Vector<double>       system_rhs;
};



template <int dim>
void right_hand_side (const std::vector<Point<dim> > &points,
                    std::vector<Tensor<1, dim> >   &values)
{
  Assert (values.size() == points.size(),
          ExcDimensionMismatch (values.size(), points.size()));
  Assert (dim >= 2, ExcNotImplemented());

  double lambda =lamb;


  for (unsigned int point_n = 0; point_n < points.size(); ++point_n)  //i should figure out the right way to do this
  {
      values[point_n][0]=0.;
      values[point_n][1]=0.;
   }

}


template <int dim>
class ComputeStress : public DataPostprocessorTensor<dim>
{
public:
  ComputeStress ()
    :
    DataPostprocessorTensor<dim> ("stress",
                                  update_gradients)
  {}
  virtual void
  evaluate_vector_field (const DataPostprocessorInputs::Vector<dim> &input_data,
                         std::vector<Vector<double> >               &computed_quantities) const
  {
    // ensure that there really are as many output slots
    // as there are points at which DataOut provides the
    // gradients:
    AssertDimension (input_data.solution_gradients.size(),
                     computed_quantities.size());
    for (unsigned int p=0; p<input_data.solution_gradients.size(); ++p)
      {
        // ensure that each output slot has exactly 'dim*dim'
        // components (as should be expected, given that we
        // want to create tensor-valued outputs), and copy the
        // gradients of the solution at the evaluation points
        // into the output slots:
        AssertDimension (computed_quantities[p].size(),
                         (Tensor<2,dim>::n_independent_components));

        Tensor<2,dim> gradient;
        for (unsigned int d=0; d<dim; ++d)
            gradient[d] = input_data.solution_gradients[p][d]; //is there an issue here?
        Tensor<2,dim> stress;
        double divergence = trace(gradient);
        for (unsigned int d=0; d<dim; ++d)
            stress[d][d]=lamb*divergence;
        stress += mu_scalar*gradient + mu_scalar*transpose(gradient);

        for (unsigned int d=0; d<dim; ++d)
          for (unsigned int e=0; e<dim; ++e)
            computed_quantities[p][Tensor<2,dim>::component_to_unrolled_index(TableIndices<2>(d,e))]
              = stress[d][e];
        // sigma(u) = lambda*div(u)*I + mu (grad(u) + grad(u)^T)

      }
  }
};

template<int dim>
FESystem<dim> make_fe(int fe_degree, bool mixed, int type)
{
  if (!mixed)
    return FESystem<dim>(FE_Q<dim>(fe_degree), dim);

 // Assert(fe_degree==2, ExcNotImplemented());

  if (mixed && red_quad==1)
    return FESystem<dim>(FE_Q<dim>(fe_degree), dim, FE_DGP<dim>(0), 1);

  if (mixed && red_quad==2)
    return FESystem<dim>(FE_Q<dim>(fe_degree), dim, FE_Q<dim>(1), 1);

  if (mixed && red_quad==3)
    return FESystem<dim>(FE_Q<dim>(fe_degree), dim, FE_DGP<dim>(1), 1);

  Assert(false, ExcNotImplemented());
}


template <int dim>
ElasticProblem<dim>::ElasticProblem (int fe_degree)
  :
    dof_handler (triangulation),
    fe (make_fe<dim>(fe_degree, mixed, red_quad))
{
  std::cout << "FE: " << fe.get_name() << std::endl;

}




template <int dim>
ElasticProblem<dim>::~ElasticProblem ()
{
  dof_handler.clear ();
}



template <int dim>
void ElasticProblem<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);
  hanging_node_constraints.clear ();
  DoFTools::make_hanging_node_constraints (dof_handler,
                                           hanging_node_constraints);

  FEValuesExtractors::Vector velocities(0);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           ConstantFunction<dim>(0.0, fe.n_components()),
                                           hanging_node_constraints,
                                           fe.component_mask(velocities));
  hanging_node_constraints.close ();

  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  hanging_node_constraints,
                                  /*keep_constrained_dofs = */ true);
  sparsity_pattern.copy_from (dsp);

  system_matrix.reinit (sparsity_pattern);

  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
}



template <int dim>
void ElasticProblem<dim>::assemble_system (int fe_degree)
{
  QGauss<dim>  quadrature_formula(2*fe_degree+2); //for god's sake change this back
  QGauss<dim-1> face_quadrature_formula(2*fe_degree+2); //for god's sake change this back  //for the boundary integral

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,  //for the boundary integral
                                    update_values         | update_quadrature_points  |
                                    update_normal_vectors | update_JxW_values);
  const unsigned int   n_q_points    = quadrature_formula.size();

  const unsigned int n_face_q_points = face_quadrature_formula.size();  //for the boundary integral


//BEGIN reduced quadrature setup

//for MidPoint
  QMidpoint<dim>  quad_1_formula;
  FEValues<dim> fe_1_val (fe, quad_1_formula,
                          update_values   | update_gradients |
                          update_quadrature_points | update_JxW_values);
  const unsigned int   n_1_pt    = quad_1_formula.size();


//for QTrapez
  QTrapez<dim> q_trap_formula;
  FEValues<dim> q_trap_vals (fe, q_trap_formula,
                             update_values   | update_gradients |
                              update_quadrature_points | update_JxW_values);
  const unsigned int   n_trap_pt = q_trap_formula.size();


//for my own 3-pt quadrature rule
  const std::vector<Point<dim>>& quad_3_pts = {Point<dim>(0.25,0.25),Point<dim>(0.75,0.25),Point<dim>(0.5,0.75)}; //Are these points defined correctly?
  const std::vector<double> & weights = {1./3 ,1./3,1./3};
  Quadrature<dim> quad_3_formula (quad_3_pts,weights);
  FEValues<dim> quad_3_vals (fe, quad_3_formula,
                             update_values   | update_gradients |
                              update_quadrature_points | update_JxW_values);
  const unsigned int  n_3_pts = quad_3_formula.size();

//END reduced quadrature setup


  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  std::vector<double>     lambda_values (n_q_points);
  std::vector<double>     mu_values (n_q_points);
  ConstantFunction<dim> lambda(lamb), mu(mu_scalar);
  std::vector<Tensor<1,dim> > rhs_values (n_q_points,
                                          Tensor<1,dim>());
  const FEValuesExtractors::Vector v(0);
  const FEValuesExtractors::Scalar p(dim);
  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
          endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
      cell_matrix = 0;
      cell_rhs = 0;
      fe_values.reinit (cell);


//BEGIN reduced quadrature reinitialization
      if(red_quad==0){
        //nothing, we've already done it
      }
      else if(red_quad==1){
          fe_1_val.reinit(cell);
      }
      else if(red_quad==2){
          q_trap_vals.reinit(cell);
      }
      else if(red_quad==3){
          quad_3_vals.reinit(cell);
      }
      else{
          std::cout<<"What have you done?!?! --Sincerely, the initialization chunk\n";
      }

//END reduced quadrature reinitialization

      lambda.value_list (fe_values.get_quadrature_points(), lambda_values);
      mu.value_list     (fe_values.get_quadrature_points(), mu_values);
      right_hand_side(fe_values.get_quadrature_points(),
                      rhs_values);



      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {

          for (unsigned int j=0; j<dofs_per_cell; ++j)
          {
              for (unsigned int q_point=0; q_point<n_q_points;
                   ++q_point)
              {
                  cell_matrix(i,j)
                          +=
                          (
                              (fe_values[v].symmetric_gradient(i,q_point) *  //Should the 2 be here? Look up symmetric gradient?
                               fe_values[v].symmetric_gradient(j,q_point)
                               *2.*mu_values[q_point])

                              )*
                          fe_values.JxW(q_point);
              }

              if (mixed)
                {
                  for (unsigned int q_point=0; q_point<n_q_points;
                       ++q_point)
                  cell_matrix(i,j)
                          +=
                              (
                          -fe_values[v].divergence(j,q_point) * fe_values[p].value(i,q_point)
                          -fe_values[v].divergence(i,q_point) * fe_values[p].value(j,q_point)
                          -1./ lambda_values[q_point] *
                          fe_values[p].value(i,q_point) * fe_values[p].value(j,q_point)
                              )*
                          fe_values.JxW(q_point);
                }
              else if(red_quad==0){ //(the rest of) the original step-8 quadrature
                  for (unsigned int q_point=0; q_point<n_q_points;
                       ++q_point){

                      cell_matrix(i,j)
                              +=
                              (
                                  (fe_values[v].divergence(i,q_point) *
                                   fe_values[v].divergence(j,q_point)
                                   *lambda_values[q_point])
                                  )*
                              fe_values.JxW(q_point);

                  }
              }
              else if(red_quad==1){
                  for (unsigned int q_point=0; q_point<n_1_pt;
                       ++q_point){

                      cell_matrix(i,j)
                              +=
                              (
                                  (fe_1_val[v].divergence(i,q_point) *
                                   fe_1_val[v].divergence(j,q_point)
                                   *lambda_values[q_point])
                                  )*
                              fe_1_val.JxW(q_point);


                  }
              }

              else if(red_quad==2){
                  for (unsigned int q_point=0; q_point<n_trap_pt;
                       ++q_point){

                      cell_matrix(i,j)
                              +=
                              (
                                  (q_trap_vals[v].divergence(i,q_point) *
                                   q_trap_vals[v].divergence(j,q_point)
                                   *lambda_values[q_point])
                                  )*
                              q_trap_vals.JxW(q_point);



                  }
              }
              else if(red_quad==3){
                  for (unsigned int q_point=0; q_point<n_3_pts;
                       ++q_point){

                      cell_matrix(i,j)
                              +=
                              (
                                  (quad_3_vals[v].divergence(i,q_point) *
                                   quad_3_vals[v].divergence(j,q_point)
                                   *lambda_values[q_point])
                                  )*
                              quad_3_vals.JxW(q_point);


                  }
              }
              else{
                   std::cout<<"What have you done?!?!" <<std::endl;
                   std::cout <<"red_quad="<<red_quad<<std::endl;
              }

      }

    }
//BEGIN boundary integrals
  for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number){
    if (cell->face(face_number)->at_boundary() &&  (cell->face(face_number)->boundary_id() == 1))
      {
        fe_face_values.reinit (cell, face_number);
        for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
          {
            const Tensor<1,dim> neumann_value = Tensor<1, dim> (Point<dim>(0.0,-1.0));
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              cell_rhs(i) += (neumann_value *
                              fe_face_values[v].value(i,q_point) *
                              fe_face_values.JxW(q_point));
          }
      }
  }
//END boundary integrals

  // RHS
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
          const unsigned int component_i = fe.system_to_component_index(i).first;
          if (component_i>=dim)
            continue;
          for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
              cell_rhs(i) += fe_values.shape_value(i,q_point) *
                      rhs_values[q_point][component_i] *
                      fe_values.JxW(q_point);
      }
      cell->get_dof_indices (local_dof_indices);
      hanging_node_constraints.distribute_local_to_global(cell_matrix,
                                                          cell_rhs,
                                                          local_dof_indices,
                                                          system_matrix,
                                                          system_rhs);
//      for (unsigned int i=0; i<dofs_per_cell; ++i)
//      {
//          for (unsigned int j=0; j<dofs_per_cell; ++j)
//              system_matrix.add (local_dof_indices[i],
//                                 local_dof_indices[j],
//                                 cell_matrix(i,j));
//          system_rhs(local_dof_indices[i]) += cell_rhs(i);
//      }

  }


//  hanging_node_constraints.condense (system_matrix);
//  hanging_node_constraints.condense (system_rhs);

//  std::map<types::global_dof_index,double> boundary_values;
//  VectorTools::interpolate_boundary_values (dof_handler,
//                                            0,
//                                            ConstantFunction<dim>(std::vector<double>{0.0,0.0}), //should this be a Tensor<1, dim> instead???
//                                            boundary_values);

//  MatrixTools::apply_boundary_values (boundary_values,
//                                      system_matrix,
//                                      solution,
//                                      system_rhs);
}


template <int dim>
double ElasticProblem<dim>::solve ()
{

  if(iterative){

      SolverControl           solver_control (10000, 1e-8 * system_rhs.l2_norm()); //make the err tol relative (step-32)
      SolverCG<>              cg (solver_control);
      PreconditionSSOR<> preconditioner;
      preconditioner.initialize(system_matrix, 1.2);


      cg.solve (system_matrix, solution, system_rhs,
                preconditioner);
      hanging_node_constraints.distribute(solution);
      return static_cast<double>(solver_control.last_step());
  }
  else{
      Timer timer;
      timer.start ();
      SparseDirectUMFPACK  sd;
      sd.initialize(system_matrix);
      sd.vmult (solution, system_rhs);
      timer.stop ();
      deallog << "done ("
              << timer()
              << "s)"
              << std::endl;
      hanging_node_constraints.distribute(solution);
      return timer();

  }
  //hanging_node_constraints.distribute(solution);

}



template <int dim>
void ElasticProblem<dim>::refine_grid ()
{
  Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

  KellyErrorEstimator<dim>::estimate (dof_handler,
                                      QGauss<dim-1>(3),
                                      typename FunctionMap<dim>::type(),
                                      solution,
                                      estimated_error_per_cell);

  GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                   estimated_error_per_cell,
                                                   0.3, 0.03);

  triangulation.execute_coarsening_and_refinement ();
}




template <int dim>
void ElasticProblem<dim>::output_results (const unsigned int cycle, int fe_degree,double solver_effort) const
{
  std::vector<std::string> solution_names (dim, "displacement");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
          data_component_interpretation
          (dim, DataComponentInterpretation::component_is_part_of_vector);
  if (mixed)
    {
      solution_names.push_back("pressure");
      data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    }

  ComputeStress<dim> compute_stress;
  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, solution_names,
                            DataOut<dim>::type_dof_data,
                            data_component_interpretation);
  data_out.add_data_vector(solution, compute_stress);
  data_out.build_patches ();

//output the displacement in the bottom right corner (timo thinks this is where the max displacement will be)

  std::cout << "h= " << triangulation.begin_active()->diameter();

  Vector<double> val(fe.n_components());
  VectorTools::point_value (dof_handler, solution,
                                           Point<2>(8.,0.),val);
  std::cout << "  y-displacement_at_(8.,0.): "
            << val(1);  //more digits "std cout double more digits"

  std::ostringstream filename;
  filename << "solution-"
           << Utilities::int_to_string (cycle, 2)
           << "-"
           << red_quad
           << "-"
           << lamb
           << "-"
           << nu
           << "-"
           << mu_scalar
           << ".vtk";
  std::ofstream output (filename.str().c_str());
  data_out.write_vtk (output);

  if(iterative==false){
      std::cout<< "  time(sec)= " <<solver_effort<<std::endl;
  }
  else {  //iterative==true
      std::cout<< "  CG its= " <<solver_effort<<std::endl;

  }


}

template <int dim>
void ElasticProblem<dim>::run (int fe_degree)
{
  for (unsigned int cycle=0; cycle<7; ++cycle)
  {
      std::cout << "Cycle " << cycle << ':' << std::endl;

      if (cycle == 0)
      {

          GridGenerator::subdivided_hyper_rectangle (triangulation, std::vector<unsigned int> {4,1}, Point<dim>(0.0,-1.0),Point<dim>(8.0,1.0),
                                                     true);
         // triangulation.refine_global (2);

      }
      else
         // refine_grid();
          triangulation.refine_global(1);

      std::cout << "   Number of active cells:       "
                << triangulation.n_active_cells()
                << std::endl;
      setup_system ();

      std::cout << "   Number of degrees of freedom: "
                << dof_handler.n_dofs()
                << std::endl;

      assemble_system (fe_degree);
      double solver_effort = solve ();
      output_results (cycle,fe_degree, solver_effort);
  }
}

}

int old_main ()
{
  int fe_degree =1;

  try
  {
      Step8::ElasticProblem<2> elastic_problem_2d(fe_degree);
      elastic_problem_2d.run (fe_degree);
  }
  catch (std::exception &exc)
  {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
  }
  catch (...)
  {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
  }

  return 0;
}


int main()
{

  for(unsigned int it = 0; it < 1; it++){

      if (it == 0){ //direct solver
          iterative = false;

          for(int r =3; r<4; r++){
	    // 0 full, 1 midpoint, 2 trapez, 3 triangle
              red_quad=r;

              double lambdas[1]={100.0};


              for (unsigned int i=0; i < 1;i++)
              {
                   lamb = lambdas[i];
                   //mu_scalar = 100; //lamb/(2*nu)-lamb;
                   //nu = 0.5 * lamb / (lamb+mu_scalar);

                   std::cout<<std::endl<<" direct solver, red_quad="<<red_quad
                           <<", lambda="<<lamb
                              <<", nu="<<nu
                                <<", mu_scalar="<<mu_scalar
                          <<std::endl;
                   old_main();
              }


          }

      }
      else{ //iterative solver
          iterative = true;

          for(int r =0; r<4; r++){

              red_quad = r;

              //double lambdas[6]={, 10.0, 50.0, 100.0, 400.0, 500.0};
              //for (unsigned int i=0; i < 6;i++){

                //  lamb = lambdas[i];
                 // lamb = 400.;
                  std::cout<<std::endl<<" iterative solver, red_quad="<<red_quad<<", lambda="<<lamb<<std::endl;
                  old_main();
              //}
          }
      }
  }
  return 0;

}


