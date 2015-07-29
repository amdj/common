// staticmsg.h
//
// Author: J.A. de Jong 
//
// Description: This class contains a message for which memory is
// pre-allocated. The message can dynamically be copied from a
// string. It is of use when a MyError() class is thrown, where a
// custom message is applied.
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef STATICMSG_H
#define STATICMSG_H
#include <stdarg.h>

namespace common {
  template<int bufsize=200> 
  class StaticMsg {
    char buf[bufsize];
  public:
    StaticMsg() {buf[0]='\0';}
    StaticMsg(const StaticMsg&)=delete;

    const char* operator()(const char* format, ...) {
      va_list args;
      va_start(args,format);
      vsprintf(buf,format,args);
      va_end(args);

      return &buf[0];
    }
    ~StaticMsg(){}
  };
  
  
} // namespace common


#endif // STATICMSG_H
//////////////////////////////////////////////////////////////////////
