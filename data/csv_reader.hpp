#ifndef __CSV_READER_H_
#define __CSV_READER_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string.h>
#include <algorithm>
#include <boost/algorithm/string.hpp>

#include <stdio.h>
#include "../util/Matrices.cpp"

/*
 * A class to read data from a csv file.

 // compile with: g++ –std=c++11
 */

/*
 * Parses through csv file line by line and returns the data
 * in vector of vector of strings.
 */
class CSVReader {
private:
    std::string fileName;
    std::string delimeter;
public:
    std::vector<std::vector<std::string> > raw = std::vector<std::vector<std::string>>(0);
    char file_state = -1;

    CSVReader(std::string filename, std::string delm);
    // Function to fetch data from a CSV File
    void read();
    std::vector<std::vector<std::string>> get_raw();
    size_t outer();
    size_t inner();
};

CSVReader::CSVReader(std::string filename, std::string delm = ",")
{
  fileName = filename;
  delimeter= delm;
}
void CSVReader::read()
{
  std::ifstream file(fileName);
  std::string line = "";
  // Iterate through each line and split the content using delimeter
  while (getline(file, line))
  {
    std::vector<std::string> vec;
    boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
    CSVReader::raw.push_back(vec);
  }
  // Close the File
  file.close();
}
std::vector<std::vector<std::string>> CSVReader::get_raw() {
  return CSVReader::raw;
};
size_t CSVReader::outer() {return CSVReader::raw.size();};
size_t CSVReader::inner() {return CSVReader::raw[0].size();};



int read_csv(string path, Matrix_2D &data) {
  CSVReader reader(path, ",");
  reader.read();
  vector<vector<string>> input = reader.get_raw();
  unsigned int cont = 0;
  for (unsigned int i = 0; i < input.size(); i++) {
    for (unsigned int j = 0; j < input[i].size(); j++) {
      data[i][j] = stof(input[i][j]);
    }
  }
  return 0;
}

#endif
