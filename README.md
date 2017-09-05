# Asynchronous Parallel SVRG

This is a newer version of the code for:
Sashank J. Reddi, Ahmed Hefny, Suvrit Sra, Barnabas Poczos and Alexander Smola
"On Variance Reduction in Stochastic Gradient Descent and its Asynchronous Variants",
NIPS 2015 (Also: http://arxiv.org/abs/1506.06840)


To compile the code, simply run "make".
A static library will be produced in "lib/opt". Executables will be produced "bin/opt".
Use "make clean" to remove output files. Use "make rebuild" as a shortcut for "make clean all"


To compile without optimizations, run "make CONFIG=dbg".
A static library will be produced in "lib/dbg". Executables will be produced "bin/dbg".


# Executables:
1. bin/opt/svm2bin - Converts the data from LIBSVM format to binary format used by this
package. The program assumes binary labels (1/+1 for positive, 0/-1 for negative).
To run use:
```
bin/opt/svm2bin <input_svm_file> <output_binary_file>
```
NOTE: You might get "Insufficient buffer size" error message if you have very large
examples. That is because the program assumes that any single example fits into the I/O
buffer whose size is defined in DataReader.h. You can try increasing this value.

2. bon/opt/bin2svm - Converts a binary data file to LIBSVM format.

3. bin/opt/train_lr - Trains a logistic regression model
(it does not actually save the model, just print the objective and gradient square norm
across time).
To run use:
```
bin/opt/train_lr --train_file=<binray training file> <optional arguments>
```
Optional arguments include:

--num_threads=<integer> (default 1)

--solver=<sgd/svrg> (default sgd)

--max_epochs=<integer> (default 1000) Maximum number of epochs (-1 for infinity).

--nupd=<integer> (default 1) Number of updates for each epoch specified in multiples
  of number of examples. Use negative numbers to specify fractions (i.e. -k is interpreted   as 1/k).
  
--step=<float> (default 1e-4) Step size.

--alpha=<float> (default -1) When greater than 0, step at iteration t is given by
  step * sqrt(al  pha/(t+alpha)), otherwise a constant step size is used.

--l2_reg=<float> (default 0.0) L2 Regularization
  (set to 1.0 to use \lambda=1/n in the paper).

--test_file=<binary test file> (default "") Specifies test examples.

--split_train_test=<1/0> (default 0) If 0, the input training file is entirely
  used for training. If 1, 20% of the training examples are used for testing.
  Has no effect if a test file is provided.

--pmode=<mode> (default FREE_FOR_ALL) Specifies parallel execution mode which can be:
* LOCKED: A thread needs to hold a lock before updating parameters.
  The lock covers the entire paramter vector.
* LOCK_FREE: A thread can update the parameter vector without software locks using
  atomic additions (using compare and swap instruction).
* FREE_FOR_ALL: Same as LOCK_FREE but without using atomic additions. We have observed
  that for sparse data and small number of threads, this mode still converges but in more
  epochs compared to LOCK_FREE. However, it can still take less wall clock time since it
  avoids the overhead of using atomic additions.
  
# Note
This code was intened for demonstration so it does not save the parameters.
  
