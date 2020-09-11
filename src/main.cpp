//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
/////////////////////////////////// AthenaXXX Main Program ///////////////////////////////
//! \file main.cpp
//  \brief AthenaXXX main program
//
// Based on the Athena (Cambridge version) and Athena++ MHD codes. Athena was originally
// written in 2002-2005 by Jim Stone, Tom Gardiner, and Peter Teuben, with many important
// contributions by many other developers after that, i.e. 2005-2014.
//
// Athena++ was started in Jan 2014.  The core design was finished during 4-7/2014 at the
// KITP by Jim Stone.  GR was implemented by Chris White and AMR by Kengo Tomida during
// 2014-2016.  Contributions from many others have continued to the present.
//
// AthenaXXX is an outgrowth of the Athena-Parthenon collaboration, and is a completely
// new implementation based on the Kokkos performance-portability library (an external
// module for this version of the code, required to run on GPUs)
//========================================================================================

#include <cstdlib>
#include <iostream>   // cout, endl
#include <string>     // string

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "utils/utils.hpp"
#include "parameter_input.hpp"
//#include "bvals/bvals.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "outputs/outputs.hpp"
#include "driver/driver.hpp"

//----------------------------------------------------------------------------------------
//! \fn int main(int argc, char *argv[])
//  \brief AthenaK main program

int main(int argc, char *argv[])
{
  std::string input_file, restart_file, run_dir;
  int  res_flag = 0;  // set to 1 if -r        argument is on cmdline 
  int narg_flag = 0;  // set to 1 if -n        argument is on cmdline
  int marg_flag = 0;  // set to 1 if -m        argument is on cmdline
  int iarg_flag = 0;  // set to 1 if -i <file> argument is on cmdline
  int wtlim = 0;

  //--- Step 1. --------------------------------------------------------------------------
  // Initialize environment

#if MPI_PARALLEL_ENABLED
#if OPENMP_PARALLEL_ENABLED
  int mpiprv;
  if (MPI_SUCCESS != MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpiprv)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI Initialization failed." << std::endl;
    return(0);
  }
  if (mpiprv != MPI_THREAD_MULTIPLE) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI_THREAD_MULTIPLE must be supported for hybrid parallelzation. "
              << MPI_THREAD_MULTIPLE << " : " << mpiprv
              << std::endl;
    MPI_Finalize();
    return(0);
  }
#else  // no OpenMP
  if (MPI_SUCCESS != MPI_Init(&argc, &argv)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI Initialization failed." << std::endl;
    return(0);
  }
#endif  // OPENMP_PARALLEL_ENABLED
  // Get process id (rank) in MPI_COMM_WORLD
  if (MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &(global_variable::my_rank))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI_Comm_rank failed." << std::endl;
    MPI_Finalize();
    return(0);
  }

  // Get total number of MPI processes (ranks)
  if (MPI_SUCCESS != MPI_Comm_size(MPI_COMM_WORLD, &global_variable::nranks)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI_Comm_size failed." << std::endl;
    MPI_Finalize();
    return(0);
  }
#else  // no MPI
  global_variable::my_rank = 0;
  global_variable::nranks  = 1;
#endif  // MPI_PARALLEL_ENABLED

  //--- Step 2. --------------------------------------------------------------------------
  // Check for command line options and respond.

  for (int i=1; i<argc; i++) {
    // If argv[i] is a 2 character string of the form "-?" then:
    if (*argv[i] == '-'  && *(argv[i]+1) != '\0' && *(argv[i]+2) == '\0') {

      // check that command line options that require arguments actually have them:
      char opt_letter = *(argv[i]+1);
      switch(opt_letter) {
        case 'c':
        case 'h':
        case 'm':
        case 'n':
          break;
        default:
          if ((i+1 >= argc) // flag is at the end of the command line options
              || (*argv[i+1] == '-') ) { // flag is followed by another flag
            if (global_variable::my_rank == 0) {
              std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                        << std::endl << "-" << opt_letter 
                        << " must be followed by a valid argument" << std::endl;
#if MPI_PARALLEL_ENABLED
              MPI_Finalize();
#endif
              return(0);
            }
          }
      }

      // set arguments, flags, or execute tasks specified by options
      switch(*(argv[i]+1)) {
        case 'i':                      // -i <input_file>
          input_file.assign(argv[++i]);
          iarg_flag = 1;
          break;
        case 'r':                      // -r <restart_file>
          res_flag = 1;
          restart_file.assign(argv[++i]);
          break;
        case 'd':                      // -d <run_directory>
          run_dir.assign(argv[++i]);
          break;
        case 'n':
          narg_flag = 1;
          break;
        case 'm':
          marg_flag = 1;
          break;
        case 't':                      // -t <hh:mm:ss>
          int wth, wtm, wts;
          std::sscanf(argv[++i], "%d:%d:%d", &wth, &wtm, &wts);
          wtlim = wth*3600 + wtm*60 + wts;
          break;
        case 'c':
          if (global_variable::my_rank == 0) ShowConfig();
#if MPI_PARALLEL_ENABLED
          MPI_Finalize();
#endif
          return(0);
          break;
        case 'h':
        default:
          if (global_variable::my_rank == 0) {
            std::cout << "Athena++ v" << ATHENA_VERSION_MAJOR << "." 
                                      << ATHENA_VERSION_MINOR << std::endl;
            std::cout << "Usage: " << argv[0] << " [options] [block/par=value ...]\n";
            std::cout << "Options:" << std::endl;
            std::cout << "  -i <file>       specify input file [athinput]\n";
            std::cout << "  -r <file>       restart with this file\n";
            std::cout << "  -d <directory>  specify run dir [current dir]\n";
            std::cout << "  -n              parse input file and quit\n";
            std::cout << "  -c              show configuration and quit\n";
            std::cout << "  -m <nproc>      output mesh structure and quit\n";
            std::cout << "  -t hh:mm:ss     wall time limit for final output\n";
            std::cout << "  -h              this help\n";
            ShowConfig();
          }
#if MPI_PARALLEL_ENABLED
          MPI_Finalize();
#endif
          return(0);
          break;
      }
    } // else if argv[i] not of form "-?" ignore it here (tested in ModifyFromCmdline)
  }

  // print error if input or restart file not given
  if (restart_file.empty() && input_file.empty()) {
    // no input file is given
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Either an input or restart file must be specified." << std::endl
              << "See " << argv[0] << " -h for options and usage." << std::endl;
#if MPI_PARALLEL_ENABLED
    MPI_Finalize();
#endif
    return(0);
  }

  //--- Step 3. --------------------------------------------------------------------------
  // Construct ParameterInput object and store smart pointer to it.
  // Constructor reads input_file and stores block/parameter names.
  // With MPI, the input is read by every rank in parallel using MPI-IO.

  ParameterInput par_input(input_file);
  par_input.ModifyFromCmdline(argc, argv);

  // Dump input parameters and quit if code was run with -n option.
  if (narg_flag) {
    if (global_variable::my_rank == 0) par_input.ParameterDump(std::cout);
//    if (res_flag == 1) restartfile.Close();
#ifdef MPI_PARALLEL
    MPI_Finalize();
#endif
    return(0);
  }

  //--- Step 4. --------------------------------------------------------------------------
  // Construct Mesh and MeshBlockTree and store smart pointer to Mesh.  Then initialize
  // Tree and construct MeshBlocks on this rank

  Mesh mesh0(&par_input);
  mesh0.BuildTree(&par_input);

  // output Mesh diagnostics
  if (global_variable::my_rank == 0) mesh0.OutputMeshStructure(marg_flag);

  //  If code was run with -m option, write mesh structure to file and quit.
  if (marg_flag) {
#ifdef MPI_PARALLEL
    MPI_Finalize();
#endif
    return(0);
  }

  //--- Step 5. --------------------------------------------------------------------------
  // Construct and initialize Physics modules.

  for (auto &pmb : mesh0.mblocks) {pmb.InitPhysicsModules(&par_input);}

  //--- Step 6. --------------------------------------------------------------------------
  // Set initial conditions by calling problem generator, or reading restart file

  auto pgen = std::make_unique<ProblemGenerator>(&par_input, &mesh0);

  //--- Step 7. --------------------------------------------------------------------------
  // Construct Outputs. Output of initial conditions is made in Driver (if needed)

  Outputs out_types(&par_input, &mesh0);

  //--- Step 8. --------------------------------------------------------------------------
  // Construct and Execute Driver

  auto pdrive = std::make_unique<Driver>(&par_input, &mesh0);

  ChangeRunDir(run_dir);
  pdrive->Initialize(&mesh0, &par_input,  &out_types);
  pdrive->Execute(&mesh0, &par_input,  &out_types);
  pdrive->Finalize(&mesh0, &par_input,  &out_types);

  //--- Step 9. -------------------------------------------------------------------------
  // clean up, and terminate

#if MPI_PARALLEL_ENABLED
  MPI_Finalize();
#endif
  return(0);
}
