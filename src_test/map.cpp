#include <unordered_map>
#include <iostream>

using namespace std;

int main() {
  unordered_map<int, double> x;
  x.reserve(100);

  x[1] = 1.0;
  x[2] = 2.0;
  x[10] = 10.0;
  x[100] = 100.0;

  x[1000] += 1000.0;
  
  auto it = x.begin();

  for(; it != x.end(); ++it) {
    cerr << &it->first << endl;
  }

  cerr << x[1000] << endl;
  
  return 0;
}
