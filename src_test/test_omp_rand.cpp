#include <random>
#include <iostream>
#include <omp.h>

#include "Solver.h"

struct S {
	S() {
		#pragma omp critical
		std::cerr << "init " << omp_get_thread_num() << std::endl;
	}

	int t;
};

int main(int argc, char **argv) {
#pragma omp parallel 
	{
	int n = 100;

	std::default_random_engine r;
	std::uniform_int_distribution<int> u(0, n-1);
	S s; s.t = 0;

#pragma omp for schedule(dynamic) 
	for(int i = 0; i < n; ++i) {
		int j = u(r);
		s.t++;

		//#pragma omp critical
		//std::cerr << j << std::endl;
	}

	#pragma omp critical
	std::cerr << s.t << std::endl;
}
}
