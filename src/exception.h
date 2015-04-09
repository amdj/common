// exception.h
//
// Author: J.A. de Jong 
//
// Description:
// My exception class
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef EXCEPTION_H
#define EXCEPTION_H 1
#include <string>
#include <stdexcept>

class MyError : public std::runtime_error {
public:
  MyError(const std::string& msg = "") : runtime_error(msg) {}
};



#endif // EXCEPTION_H
//////////////////////////////////////////////////////////////////////

