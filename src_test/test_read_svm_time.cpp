#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstring>

#include<unistd.h>
#include<fcntl.h>

#include "Platform.h"
#include "DataReader.h"

using namespace std;

const char *file_name = "/home/ahefny/data/url_combined";

int g_num_tokens = 0;
double g_sum = 0.0;

char *readDouble(char *buffer, size_t size, char delim, double *value) {
	int whole = 0; int frac = 0; int denom = 1;
	const char *end = buffer + size;

	while(buffer < end && *buffer != delim && *buffer != '.') {
		whole = whole * 10 + (*(buffer++) - '0');
	} 

	while(buffer < end && *buffer != delim) {
		//assert(*buffer != 'e');
		//assert(*buffer != 'E');
		frac = frac * 10 + (*(buffer++) - '0');
		denom *= 10;
	} 

	*value = whole + frac / (double) denom;

	if(buffer == end) {return nullptr;}
	else {return buffer+1;}
}

void processToken(char *token, size_t size, 
						 int *index, double *value) {
	token[size] = 0;
	char*end = token + size;
	++g_num_tokens;

	int idx = 0;
	
	while(*token != ':') idx = (idx)*10 + (*(token++) - '0');

	*index = idx;
	*value = atof(token+1);
	//readDouble(token+1, end-token-1, ' ', value);
	g_sum += *value;
}

void processLine(char *line, size_t size, int &num_lines) {
	line[size] = 0;
	char *end = line + size;

	char *label_end = (char*) memchr(line, ' ', size);
	*label_end = 0;
	int label = atof(line);
	++g_num_tokens;
	
	int index; double value;

	char *last_p = label_end + 1;
	char *p = last_p;
	while((p = (char*) memchr(p+1, ' ', end-p-1)) != 0) {
		// Process Token		
		processToken(last_p, p-last_p, &index, &value);		
		last_p = p+1;
	}

	processToken(last_p, end-last_p, &index, &value);		

	++num_lines;
}

void processFileUnix() {
	int fd = open(file_name, O_RDONLY);
	posix_fadvise(fd, 0, 0, 1);  // FDADVICE_SEQUENTIAL

	const size_t BUFFER_SIZE = 16 * 1024;
	const size_t STASH_SIZE = 10000;	
	static_assert(STASH_SIZE <= BUFFER_SIZE, "Invalid stash size");

	char storage[BUFFER_SIZE + STASH_SIZE];
	char * const stash = storage;
	char * const buffer = storage + STASH_SIZE;
	
	int l = 0;

	char *last_stash = buffer;
	bool is_next_token_label = true;

	int label;
	int index;
	double value;

	while(ssize_t b = read(fd, buffer, BUFFER_SIZE)) {
		char *p = buffer-1;
		char *end = buffer + b;
		char *last_p = last_stash;
		
		while((p = (char*) memchr(p+1, '\n', end-p-1)) != 0) {
			
			// process from last_p to p			
			processLine(last_p, p-last_p, l);
			//is_next_token_label = 
			//	processToken(last_p, p-last_p, is_next_token_label, 
			//				 &label, &index, &value);

			last_p = p+1;			
		}

		const size_t stash_size = end-last_p;
		last_stash = buffer - stash_size;
		ASSERT(stash_size <= STASH_SIZE, "Insufficient stash size");
		memcpy(last_stash, last_p, stash_size);
	}

	LOG(l);

	close(fd);
}

void processFile() {	
	ifstream in(file_name, ios::binary);
  //TODO: Check file

  double result = 0.0;

  string line, token;
  int linenum = 0;
  int numFeatures = -1;

  double label;

  while(getline(in, line, '\n')) {
    ++linenum;

	//result += 1.0;

    vector<string> tokens;
    istringstream linestream(line);

    while(getline(linestream, token, ' ')) {
		++g_num_tokens;
		//result += 1.0;
		tokens.push_back(token);
    }

	
    label = atof(tokens[0].c_str());
    if(label < 0.0) {label = 0.0;}

    for(size_t i = 1; i < tokens.size(); ++i) {
      size_t colon = tokens[i].find(':');
      ASSERT(colon > 0, "line=" << linenum << " i=" << i << " token=" << tokens[i]);

      char *str = const_cast<char *>(tokens[i].c_str());
      str[colon] = 0;
      int idx = atoi(str);
      double val = atof(str + colon + 1);

      if(idx > numFeatures) {numFeatures = idx;}

	  result += label + idx + val;
    }

    //if(linenum % 1000 == 0) {LOG("Read " << linenum << " examples");}
  }

  LOG("# of features = " << linenum << " " << numFeatures << " " << result);

  /*for(size_t i = 0; i < indices.size(); ++i) {
    data.push_back(SparseVector<double>(numFeatures));
    SparseVector<double> &newVector = data.back();

    for(size_t j = 0; j < indices[i].size(); ++j) {
      newVector.insert(indices[i][j]) = vals[i][j];
    }
	}*/
}

void processFileObj() {
	SVMDataReader reader(file_name);
	reader.init();

	SparseExample example;
	example.feats.resize(1000000000);

	int num_lines = 0;

	while(reader.read(&example)) {
		++num_lines;

		//if(num_lines % 1000 == 0) {LOG(num_lines);}
	}

	reader.close();

	LOG(num_lines);
}

int main() {   
	Platform::init();
	Platform::setNumLocalThreads(1);

	Platform::Time start = Platform::getCurrentTime();
	processFileObj();
	Platform::Time end = Platform::getCurrentTime();

	cout << "Time duration = " << Platform::getDurationms(start, end) << endl;
	LOG(g_num_tokens);
	
	return 0;
}
