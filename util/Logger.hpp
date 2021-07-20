#include <iostream>
#include <fstream>
using namespace std;

#ifndef _J_ACTIVITY_LOGGER
#define _J_ACTIVITY_LOGGER

class Logger {
private:
  fstream file;
public:
  Logger(string s, string m_entry = " Logfile ");
  ~Logger();
  void log(string s, string end = "\n", bool announce = false);
};

#endif
