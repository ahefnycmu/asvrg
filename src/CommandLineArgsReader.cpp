#include "CommandLineArgsReader.h"

using namespace std;

void CommandLineArgsReader::read(int argc, const char **argv) {
	for(int i = 1; i < argc; ++i) {
		string key = argv[i];
		string value = "1";

		int pos = key.find('=');

		if(pos >= 0) {
			value = key.substr(pos+1);
			key = key.substr(0, pos);
		}

		args_[key] = value;
	}
}

std::string CommandLineArgsReader::getParam(const std::string &key,
											const std::string &defaultValue) const {
	auto it = args_.find(key);

	if(it == args_.end()) {
		return defaultValue;
	} else {
		return it->second;
	}
}
