#include <iostream>
#include <fstream>

#include "Logger.hpp"

using namespace std;

Logger::Logger(string s, string m_entry) {
  file.open(s, ifstream::trunc);

  string p = "+";
  for (size_t i = 0; i < 50; i++) p+= "-";

  log(p); log("|\t" + m_entry); log(p);
}
Logger::~Logger() {
  file.close();
}
void Logger::log(string s, string end, bool announce) {
  if (announce) cout << s << end;
  file << s << end;
}
