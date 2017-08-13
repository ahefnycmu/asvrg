#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>

#include "CommandLineArgsReader.h"
#include "Platform.h"
#include "BatchOracle.h"
#include "LogisticRegressionOracle.h"

#include "SGDSolver.h"
#include "SVRGSolver.h"

template<class Solver>
void fillOptions(const CommandLineArgsReader &args, typename Solver::Options *options) {
  double step = atof(args.getParam("--step", "1e-4").c_str());
  double alpha = atof(args.getParam("--alpha", "-1").c_str());
  ParallelMode parallel_mode = ParallelMode::fromString(args.getParam("--pmode", "FREE_FOR_ALL").c_str());
  int max_epochs = atoi(args.getParam("--max_epochs", "1000").c_str()); //Use -1 for unlimited
  int num_nupdates_per_epoch = atoi(args.getParam("--nupd", "1").c_str());
    
  double target_objective = -std::numeric_limits<double>::infinity();
  std::string obj = args.getParam("--obj", "-inf");
  if(obj != "-inf") {
    target_objective = atof(obj.c_str());
  }

  //Set solver options
  options->max_num_epochs = 1000;
  options->step = step;
  options->alpha_step = alpha;
  options->parallel_mode = parallel_mode;
  options->target_objective = target_objective;
  options->max_num_epochs = max_epochs;
  options->num_nupdates_per_epoch = num_nupdates_per_epoch;
}

template<class Solver>
void train_lr(const CommandLineArgsReader &args) {
  typedef typename Solver::ParamVector ParamVector;
  typedef typename Solver::Options Options;
  typedef typename Solver::Solution Solution;

  double l2_reg = atof(args.getParam("--l2_reg", "0.0").c_str());
  bool normalize_examples = static_cast<bool>(
      args.getParam("--normalize_examples", "1").c_str());
  std::string training_file = args.getParam("--train_file", "");
  std::string test_file = args.getParam("--test_file", "");
  bool split_train_test = static_cast<bool>(
      atoi(args.getParam("--split_train_test", "0").c_str()));
  ASSERT(test_file == "" || !split_train_test, "");

  int batch_size = atoi(args.getParam("--batch", "0").c_str());
  //ASSERT(batch_size > 0, "Invalid batch size");
  
  int num_features, num_test_features;
  std::vector<SparseVec> examples;
  std::vector<double> labels;
  std::vector<SparseVec> test_examples;
  std::vector<SparseVec> *test_examples_ptr = 0;
  std::vector<double> test_labels;
  std::vector<double> *test_labels_ptr = 0;

  BinaryDataReader::readTrainingFile(
      training_file.c_str(), normalize_examples, examples, labels,
      num_features);

  if(test_file != "") {
    BinaryDataReader::readTrainingFile(
        test_file.c_str(), normalize_examples, test_examples, test_labels,
        num_test_features);
    ASSERT(num_features == num_test_features,
           "Incompatible train and test files");
    test_examples_ptr = &test_examples;
    test_labels_ptr = &test_labels;
  } else if(split_train_test) {
    std::default_random_engine r(0);
    std::uniform_int_distribution<int> u(1, 100);

    int end = examples.size()-1; int idx = 0;
    while(idx < end) {
      int p = u(r);
      if (p <= 20) {
	test_examples.push_back(examples[idx]);
	test_labels.push_back(labels[idx]);
	examples[idx] = examples[end];
	labels[idx] = labels[end];
	--end;
      } else {++idx;}
    }

    examples.erase(examples.begin()+end, examples.end());
    labels.erase(labels.begin()+end, labels.end());

    test_examples_ptr = &test_examples;
    test_labels_ptr = &test_labels;
  }

  LOG("# Train Examples: " << examples.size());
  LOG("# Test Examples:" << test_examples.size());
  
  Oracle<ParamVector, SparseVec> *oracle = new
      LogisticRegressionOracle<ParamVector>(
          &examples, &labels, num_features, l2_reg, test_examples_ptr,
          test_labels_ptr);

  if(batch_size > 0) {
    oracle = new BatchOracle<ParamVector>(oracle, true, batch_size);
  }
 
  Options options;
  fillOptions<Solver>(args, &options);
  
  options.print(std::cout);
  std::cout << "L2 Reg: " << l2_reg << std::endl;
  std::cout << "Threads: " << Platform::getNumLocalThreads() << std::endl;

  std::unique_ptr<Solver> solver(new Solver(options));
  Solution solution = solver->solve(oracle);
  
  std::cout << "Time: " << solution.timems << std::endl;
  std::cout << "Objective: " << solution.objective << std::endl;
  std::cout << "Trace:" << std::endl;

  std::cout << "epoch\ttime(ms)\tobj\tgrad_sq_norm\ttest_error" << std::endl;
  for(auto &t : solution.trace) {
    std::cout << t.other_info["epoch"] << "\t" << t.timems <<
        "\t" << t.objective << "\t" << t.grad_sq_norm;
    std::cout << "\t" << t.other_info["test_error"] << std::endl;
  }

  delete oracle;
}

int main(int argc, const char **argv) {
  // Set max double output precision
  std::cout.precision(std::numeric_limits<long double>::digits10 + 1);
  std::cerr.precision(std::numeric_limits<long double>::digits10 + 1);
  
  Platform::init();

  CommandLineArgsReader args;
  args.read(argc, argv);

  int num_threads = atoi(args.getParam("--num_threads", "1").c_str());  
  if(num_threads > 0) {Platform::setNumLocalThreads(num_threads);}
  
  std::string log_tag = args.getParam("--log_tag", "");
  SET_LOG_TAG(log_tag);
    
  LOG("Using " << Platform::getNumLocalThreads() << " threads");

  std::string solver = args.getParam("--solver", "svrg").c_str();

  std::cout << "Using " << solver << " Algorithm" << std::endl;
  
  if(solver == "sgd") {
    train_lr<SGDSolver>(args);
  } else if(solver == "svrg") {
    train_lr<SVRGSolver>(args);
  } else {
    ASSERT(false, "Invalid Sovler");
  }  
}


