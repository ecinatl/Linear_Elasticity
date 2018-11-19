/* ---------------------------------------------------------------------
*
* Copyright (C) 2018
*
* Author: Emma Cinatl, Clemson University, 2018
* Based on the deal.II step-8 tutorial program
*
* ---------------------------------------------------------------------
*
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
#include <deal.II/base/symmetric_tensor.h>

#include <fstream>
#include <iostream>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_dgp.h>



double lamb =1.; //global variable
int red_quad = 0;    //global variable
bool iterative = false; //global variable
double mu_scalar = 1.;
double nu = 0.5*lamb/(lamb+mu_scalar);
int sol_type =1;

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



    for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
    {
        double x=points[point_n][0];
        double y=points[point_n][1];
        double pi = numbers::PI;

        if(sol_type==1 || sol_type==2){  //if sol_type == 1 or 2
            values[point_n][0]=0.01*mu_scalar*pi*pi*sin(pi*x);
            values[point_n][1]=-0.01*mu_scalar*pi*pi*pi*y*cos(pi*x);
        }
        else {       //if sol_type == 0 or ==3
            values[point_n][0] = 0.01*((lambda+2*mu_scalar)*pi*pi*y*sin(pi*x)
                                       +(lambda+mu_scalar)*pi*0.25*sin(pi*0.25*y));

            values[point_n][1] = 0.01*((lambda+2*mu_scalar)*pi*(pi/16.)*x*cos(pi*0.25*y)
                                       -(lambda+mu_scalar)*pi*cos(pi*x));

        }

    }
}

template <int dim>
class SolutionValues : public Function<dim>
{
public:
    SolutionValues () : Function<dim>(dim+1) {}

    virtual void vector_value (const Point<dim>   &p,
                               Vector<double> &values) const;
    virtual void vector_gradient (const Point<dim>  &p,
                                  std::vector<Tensor<1,dim>> &gradients) const;

};

template <int dim>
void SolutionValues<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const{

    //Assert(values.size()==dim, ExcMessage("oh no!"));

    if(sol_type == 1|| sol_type ==2){ //if sol_type == 1 or 2
        values[0]=0.01*sin(numbers::PI*p[0]);
        values[1]=-0.01*numbers::PI*p[1]*cos(numbers::PI*p[0]);

    }
    else{        //if sol_type == 0 or ==3
        values[0]=0.01*p[1]*sin(numbers::PI*p[0]);
        values[1]=0.01*p[0]*cos(numbers::PI*p[1]/4);

    }

}

template <int dim>
void SolutionValues<dim>::vector_gradient(const Point<dim> &p,
                                          std::vector<Tensor<1,dim>> &gradients) const{

    //Assert(gradients.size()==dim, ExcMessage("oh no!"));
    Assert(dim==2, ExcMessage("SolutionValues vector_gradient not implemented in 3D"));

    if(sol_type==1 || sol_type == 2){
        gradients[0][0]=0.01*numbers::PI*cos(numbers::PI*p[0]);
        gradients[0][1]=0;
        gradients[1][0]=0.01*numbers::PI*numbers::PI*p[1]*sin(numbers::PI*p[0]);
        gradients[1][1]=-0.01*numbers::PI*cos(numbers::PI*p[0]);

    }
    else{  //sol_type==0 || sol_type == 3
        gradients[0][0]=0.01*numbers::PI*p[1]*cos(numbers::PI*p(0));
        gradients[0][1]=0.01*sin(numbers::PI*p(0));
        gradients[1][0]=0.01*cos(p(1)*numbers::PI/4);
        gradients[1][1]=-0.01*(numbers::PI/4)*p(0)*sin(p(1)*numbers::PI/4);

    }

}

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
{}




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
                                             SolutionValues<dim>(),
                                              //ConstantFunction<dim>(0.0, fe.n_components()),
                                             hanging_node_constraints,
                                             fe.component_mask(velocities));

    if(sol_type==2 || sol_type==3){
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  2,
                                                  SolutionValues<dim>(),
                                                  hanging_node_constraints,
                                                  fe.component_mask(velocities));

        VectorTools::interpolate_boundary_values (dof_handler,
                                                  3,
                                                  SolutionValues<dim>(),
                                                  hanging_node_constraints,
                                                  fe.component_mask(velocities));
    }



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


    QGauss<dim>  quadrature_formula(2*fe_degree+2);

    
    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values   | update_gradients |
                             update_quadrature_points | update_JxW_values);

//BEGIN for the boundary integral
    QGauss<dim-1> face_quadrature_formula(2*fe_degree+2); //for the boundary integral
    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,  //for the boundary integral
                                      update_values         | update_quadrature_points  |
                                      update_normal_vectors | update_JxW_values);
    const unsigned int n_face_q_points = face_quadrature_formula.size();  //for the boundary integral


//END for the boundary integral


//BEGIN reduced quadrature set  up

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

        //BEGIN outputting the quadrature points of QTrapez

    //    for(int i = 0; i < q_trap_formula.get_points().size(); i++){
    //        std::cout<<q_trap_formula.get_points()[i]<<std::endl;;

    //    }
        //END outputting the quadrature points of QTrapez

//BEGIN: How to make the reduced quadrature better
//    FEValues<dim> *fe_div;
//    if (red_quad==0)
//        fe_div = &fe_values;
//    else if (red_quad==1)
//        fe_div = &q_trap_vals;
//END: How to make the reduced quadrature better


//for my own 3-pt quadrature rule
    const std::vector<Point<dim>>& quad_3_pts = {Point<dim>(0.25,0.25),Point<dim>(0.75,0.25),Point<dim>(0.5,0.75)}; //Are these points defined correctly?
    const std::vector<double> & weights = {1./3,1./3,1./3};
    Quadrature<dim> quad_3_formula (quad_3_pts,weights);

    FEValues<dim> quad_3_vals (fe, quad_3_formula,
                               update_values   | update_gradients |
                                update_quadrature_points | update_JxW_values);
    const unsigned int  n_3_pts = quad_3_formula.size();

//END reduced quadrature setup


    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();
    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    std::vector<double>     lambda_values (n_q_points);
    std::vector<double>     mu_values (n_q_points);
    ConstantFunction<dim> lambda(lamb), mu(mu_scalar);
    std::vector<Tensor<1,dim> > rhs_values (n_q_points,
                                            Tensor<1,dim>());
    const FEValuesExtractors::Vector v(0); //"at component
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
        const SolutionValues<dim> exact_solution; //similar to step-7 but not to the declarations above it...



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
                                (fe_values[v].symmetric_gradient(i,q_point) * //[v]: v is an extractor
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

        if(sol_type==2 || sol_type==3){
        //BEGIN boundary integrals

          for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
          {
            if (cell->face(face_number)->at_boundary() &&  (cell->face(face_number)->boundary_id() == 1))
              {
                fe_face_values.reinit (cell, face_number);


                for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                  {

                    //BEGIN calcuating stress the "right" way
                    Tensor<2,dim> stress;

                    std::vector<Tensor<1,dim>> gradients(dim);
                    exact_solution.vector_gradient(fe_face_values.quadrature_point(q_point),gradients);
                    double divergence = gradients[0][0]+gradients[1][1]; //this won't work in 3d

                    //the below lines will also not work in 3d
                    std::vector<Tensor<1,dim>> grad_transpose(dim);


                    grad_transpose[0][0]=gradients[0][0];
                    grad_transpose[0][1]=gradients[1][0];
                    grad_transpose[1][0]=gradients[0][1];
                    grad_transpose[1][1]=gradients[1][1];

                    for (unsigned int d=0; d<dim; ++d){
                        stress[d][d]=lamb*divergence;
                        stress[d][0] += mu_scalar*gradients[d][0] + mu_scalar*grad_transpose[d][0];
                        stress[d][1] += mu_scalar*gradients[d][1] + mu_scalar*grad_transpose[d][1];
                    }

                   // sigma(u) = lambda*div(u)*I + mu (grad(u) + grad(u)^T)




                    const Tensor<1,dim>
                 neumann_value
                      = stress * fe_face_values.normal_vector(q_point);
                    //END calculating stress the "right" way




                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                      cell_rhs(i) += (neumann_value *
                                      fe_face_values[v].value(i,q_point) *
                                      fe_face_values.JxW(q_point));
                  }
              }
          }
        }
        //END boundary integrals

        // RHS



        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            const unsigned int
                    component_i = fe.system_to_component_index(i).first;
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
    }
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
        //std::cout << "   CG converged in " << solver_control.last_step() << " iterations." << std::endl;
        hanging_node_constraints.distribute (solution);
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
        //std::cout<<"solve time="<<timer()<<" sec\n";
        hanging_node_constraints.distribute (solution);
        return timer();

    }


}



template <int dim>
void ElasticProblem<dim>::refine_grid ()
{
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate (dof_handler,
                                        QGauss<dim-1>(2),
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
    std::vector<std::string> solution_names (dim, "velocity");
    solution_names.push_back("pressure");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation
            (dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, solution_names,
                              DataOut<dim>::type_dof_data,
                              data_component_interpretation);
    data_out.build_patches ();
    std::ostringstream filename;
    filename << "solution-"
             << Utilities::int_to_string (cycle, 2)
             << "-"
             << iterative
             << "-"
             << red_quad
             << "-"
             << lamb
             << ".vtk";
    std::ofstream output (filename.str().c_str());
    data_out.write_vtk (output);


    Vector<float> difference_per_cell (triangulation.n_active_cells());
    VectorTools::integrate_difference (dof_handler,
                                       solution,
                                       SolutionValues<dim>(),
                                       difference_per_cell,
                                       QGauss<dim>(2*fe_degree+2), //pretty sure input here just needs to be >= to input used in actual problem
                                       VectorTools::L2_norm);
    const double L2_error = difference_per_cell.l2_norm();

    VectorTools::integrate_difference (dof_handler,
                                       solution,
                                       SolutionValues<dim>(),
                                       difference_per_cell,
                                       QGauss<dim>(2*fe_degree+2),
                                       VectorTools::H1_norm);
    const double H1_error = difference_per_cell.l2_norm();

    std::cout << "  h= " << triangulation.begin_active()->diameter()
              << "  L2= " << L2_error
              << "  H1= " << H1_error;
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
    for (unsigned int cycle=0; cycle<3; ++cycle) //refining
    {
        std::cout << "Cycle " << cycle << ':' << std::endl;

        bool colorize = false;
        if (sol_type == 2 || sol_type == 3)
            colorize = true;


        if (cycle == 0)
        {
            GridGenerator::hyper_cube (triangulation, -1, 1, colorize); //this should maybe be a const bool?
            triangulation.refine_global (5);


            //BEGIN check if it's really colorized
//            typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(),
//                    endc = triangulation.end();
//            for (; cell!=endc; ++cell)
//            {
//               for (unsigned int f=0;f<GeometryInfo<dim>::faces_per_cell; ++f)
//                   if (cell->face(f)->at_boundary())
//                   {
//                       std::cout << cell->face(f)->boundary_id()<<std::endl;
//                   }
//            }
            //END check if it's really colorized



        }
        else
            triangulation.refine_global (1);

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

int old_main (int fe_degree)
{
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
    if(sol_type == 1)
        std::cout<<"Divergence-free manufactured solution with all boundaries Dirichlet\n";
    else if (sol_type == 2)
        std::cout<<"Divergence-free manufactured solution with 3 boundaries Dirichlet, 4th traction force\n";
    else if (sol_type==3)
        std::cout<<"Original manufactured solution with 3 boundaries Dirichlet, 4th traction force\n";
    else
        std::cout<<"Original manufactured solution\n";

    int fe_degree =2;

    for(unsigned int it = 0; it < 1; it++){

        if (it == 0){ //direct solver
            iterative = false;

            for(int r =1; r<2; r++){

               red_quad=r;

                double lambdas[3]={1., 500.0, 50000.0};

                for (unsigned int i=0; i < 3;i++)
                {
                    lamb = lambdas[i];
                    double nu = 0.5*lamb/(lamb+mu_scalar);
                    std::cout<<std::endl<<"fe_degree="<<fe_degree<<", red_quad="<<red_quad<<", lambda="<<lamb<<", nu=" <<nu<<std::endl;
                    old_main(fe_degree);
                 }


            }

        }
//        else{ //iterative solver
//            iterative = true;

//            for(int r =0; r<4; r++){

//                red_quad = r;

//               // double lambdas[6]={0.5, 10.0, 50.0, 100.0, 300.0, 500.0};

//                //for (unsigned int i=0; i < 6;i++)
//              //  {
//                   // lamb = lambdas[i];
//                    //lamb = 100.;
//                    std::cout<<std::endl<<" iterative solver, red_quad="<<red_quad<<", lambda="<<lamb<<std::endl;
//                    old_main();
//                //}
//            }
//        }
    }
    return 0;

}


