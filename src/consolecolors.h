// consolecolors.h
//
// Author: J.A. de Jong 
//
// Description:
// Print text from C++ to stdout in color
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef CONSOLECOLORS_H
#define CONSOLECOLORS_H
#include <iostream>

#ifndef SWIG
#define red "\e[31m"
#define green "\e[32m"
#define def " \e[39m"

#endif  // SWIG

// Command to clear the content of a console
inline void clearConsole(){
  std::cout << "\033c" << std::endl;
}

#endif // CONSOLECOLORS_H
//////////////////////////////////////////////////////////////////////

