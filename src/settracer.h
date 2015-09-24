// settracer.h
//
// Author: J.A. de Jong 
//
// Description:
//
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef SETTRACER_H
#define SETTRACER_H
#include "tracer.h"

namespace tracer {

  template<int& tracername>
  void setTracer(int tracelevel){
    #if TRACER==1
    tracername=tracelevel;
    #endif
    
  }      
} // namespace tracer

#endif // SETTRACER_H
//////////////////////////////////////////////////////////////////////




