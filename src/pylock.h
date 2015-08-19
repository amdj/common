// pylock.h
//
// Author: J.A. de Jong 
//
// Description:
//
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef PYLOCK_H
#define PYLOCK_H

namespace python{
  class PyLock{
  public:
    PyLock();		// Acquires Global Interpreter Lock if necessary
    ~PyLock();		// Releases Global Interpreter Lock if necessary
  };
} // namespace python


#endif // PYLOCK_H
//////////////////////////////////////////////////////////////////////
