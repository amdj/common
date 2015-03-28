// ConsoleColor.h
#pragma once
#ifndef _CONSOLECOLORS_H_
#define _CONSOLECOLORS_H_

#include <iostream>
#ifndef SWIG
#define red "\e[31m"
#define green "\e[32m"
#define def " \e[39m"

        // FG_RED      = 31,
        // FG_GREEN    = 32,
        // FG_BLUE     = 34,
        // FG_DEFAULT  = 39,
        // BG_RED      = 41,
        // BG_GREEN    = 42,
        // BG_BLUE     = 44,
        // BG_DEFAULT  = 49
#endif

inline void clearConsole(){
  std::cout << "\033c";
}

#endif /* _CONSOLECOLORS_H_ */
