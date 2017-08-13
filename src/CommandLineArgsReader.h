#ifndef _RCD_COMMANDLINEARGSREADER_H_
#define _RCD_COMMANDLINEARGSREADER_H_

#include <map>
#include <string>

class CommandLineArgsReader {
 public:
	void read(int argc, const char **argv);
	std::string getParam(const std::string &key, const std::string &defaultValue) const;

 private:
	std::map<std::string, std::string> args_;
};

#endif
