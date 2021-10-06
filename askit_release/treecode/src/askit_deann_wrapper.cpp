

#include <mpi.h>

#include "askit_alg.hpp"
#include "gaussian_kernel.hpp"
#include "laplace_kernel.hpp"
#include "kernel_inputs.hpp"
#include "askit_utils.hpp"

#include <CmdLine.h>
#include <ompUtils.h>
#include <ctime>
#include <omp.h>
#include <float.h>
#include <parallelIO.h>
#include <iostream>

using namespace Torch;
using namespace askit;


int main (int argc, char* argv[])
{

  MPI::Init(argc, argv);

  int rank, size;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  srand((unsigned)time(NULL)*rank);

  // Set up and read in the parameters
  CmdLine cmd;
  const char *phelp = "Executable for KDE as used in DEANN experiments and a Gaussian kernel. IMPORTANT: number of MPI ranks must be a power of two!";
  cmd.addInfo(phelp);


  /////////////////////////// Input Data /////////////////////////////////
  char* training_filename;
  cmd.addSCmdOption("-training_data", &training_filename, "REQUIRED",
    "File containing data set.");  

  char* test_filename;
  cmd.addSCmdOption("-test_data", &test_filename, "REQUIRED", "File containing test point coordinates.");

  char* training_knn_filename;
  cmd.addSCmdOption("-training_knn_file", &training_knn_filename, "REQUIRED",
    "File containing the neighbor info for the training points.");

  char* test_knn_file;
  cmd.addSCmdOption("-test_knn_file", &test_knn_file, "none", 
    "Filename containing the KNN info for the test data.");


  bool is_binary_file = true;

  long training_N;
  cmd.addLCmdOption("-training_N", &training_N, 0, "Total number of data points in input file.");

  long test_N;
  cmd.addLCmdOption("-test_N", &test_N, 0, 
    "Number of test points in test_data_file.");

  int d;
  cmd.addICmdOption("-d", &d, 0, "Dimension of data points.");

  ////////////////////// Main ASKIT parameters //////////////////////////

  int num_neighbors_in;
  cmd.addICmdOption("-k", &num_neighbors_in, 10, "Number of nearest neighbors to use (Default: 10).");

  double id_tol;
  cmd.addRCmdOption("-id_tol", &id_tol, 0.1, "(\\tau) Error tolerance in estimated singular values in skeletonization. This is the cutoff for the diagonal entries of R. (default: 0.1)");

  double bandwidth;
  cmd.addRCmdOption("-h", &bandwidth, 1.0, "Bandwidth of the Gaussian kernel (Default 1.0).");

  int max_points_per_node;
  cmd.addICmdOption("-max_points", &max_points_per_node, 512, "Max points per node (Default: 512).");

  int oversampling_factor;
  cmd.addICmdOption("-oversampling", &oversampling_factor, 5, "Oversampling factor (Default: 5).");

  // output
  //char* output_file;
  //cmd.addSCmdOption("-output", &output_file, "output.txt", "File to store computed densities.");

  ////////////////////////////////////////////////////////////////////////
  
  cmd.read(argc, argv);
 
  string training_file_string(training_filename);
  string charge_filename("ones");
  string test_file_string(test_filename);

  // Read the data and charges, set up the global IDs 
  fksData* train_data = ReadDistData(training_file_string, charge_filename, training_N, d, is_binary_file);

  // Inputs to the kernel function
  KernelInputs kernel_inputs;
  kernel_inputs.bandwidth = bandwidth;
  kernel_inputs.do_variable_bandwidth = false;

  // Set up the inputs for the kernel function

  // These are generally good defaults
  int num_skel_targets = 2;
  int id_rank = 2048;
  int max_tree_level = 30;
  int min_skeleton_level = 2;
  int num_uniform_required = 0;
  
  AskitInputs askit_inputs(num_skel_targets, id_rank, max_points_per_node, max_tree_level,
    num_neighbors_in, min_skeleton_level, oversampling_factor, num_uniform_required, training_knn_filename, is_binary_file);

  // Set up flags
  askit_inputs.compress_self = false;
  askit_inputs.skeletonize_self = false;

  askit_inputs.use_simplified_adaptive_id = true;
  askit_inputs.use_adaptive_id = false;
  askit_inputs.id_tol = id_tol;
  askit_inputs.do_absolute_id_cutoff = true;
  askit_inputs.do_scale_near_adaptive = false;

  askit_inputs.traverse_to_self_only = false;
  askit_inputs.do_absolute_num_targets = false;
  askit_inputs.dont_store_proj = false;

  // FMM flags
  askit_inputs.do_fmm = true;
  askit_inputs.merge_aggressive = false;
  
  // Save potentials output
  askit_inputs.save_training_potentials = false;
  askit_inputs.neighbors_to_pass_up = 4;
  askit_inputs.do_adaptive_level_restriction = true;
  askit_inputs.do_split_k = true;
  // 0 results in the default (k/2) setting unless k = 1
  askit_inputs.pruning_num_neighbors = 0;


  double build_start = omp_get_wtime();
  AskitAlg<GaussianKernel> alg(train_data, kernel_inputs, askit_inputs);
  double build_time = omp_get_wtime() - build_start;

  std::cout << "BUILD TIME: " << build_time << std::endl;

  ifstream test_fstream(test_filename);
  
  vector<double> potentials;
  double normalization;
  
  bool do_test_set = test_fstream.good();


  cout << "Computing KDE on " << test_N << " test points.\n";

  fksData* test_data = ReadDistData(test_file_string, charge_filename, test_N, d, is_binary_file);

  alg.AddTestPoints(test_N, test_data, test_knn_file, is_binary_file);

  potentials = alg.ComputeAllTestPotentials();

  vector<double> yest(potentials.size());
  alg.tree->shuffle_back(alg.N_test, potentials.data(),
                          alg.tree->inProcTestData->gids.data(),
                          alg.N_test, yest.data(), MPI_COMM_WORLD);

  double oosqrtn = 1.0/sqrt(training_N);

//  for (auto& d: potentials) {
//    cout << d << endl;
//  }
//
//  // Output timings
//  cout << "\n\n\tTIMINGS\n\n";
//  std::cout << "Tree construction and NN exchange: " << alg.tree_build_time << " s.\n";
//  std::cout << "LET construction, communication, and target list construction: " << alg.let_traversal_time << " s.\n";
//  std::cout << "Skeleton construction: " << alg.skeletonization_time << " s.\n";
//  std::cout << "Interaction List Blocking: " << alg.list_blocking_time << " s.\n";
//  std::cout << "Evaluation: " << alg.evaluation_time << " s.\n";
//  std::cout << "Test Evaluation: " << alg.test_evaluation_time << " s.\n";
//  std::cout << "Direct Comp. (on evaluation points): " << alg.exact_comp_time << " s.\n";
//  std::cout << "Test Interaction List Blocking: " << alg.test_list_blocking_time << " s.\n";
//  std::cout << "Update Charges time (average): " << alg.update_charges_time << " s.\n";

double time = alg.test_evaluation_time / test_N; // just report on the average because of the batch mode
int samples = alg.num_kernel_evaluations / test_N;

for (int i = 0; i < test_N; i++) {
  cout << "RESULT id=" << i << " est=" << potentials[i] * oosqrtn << " samples=" << samples <<  " time=" << time << endl;
}
  

  return 0;
  
}


