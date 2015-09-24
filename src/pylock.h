// pylock.h
//
// Author: J.A. de Jong 
//
// Description:
// When an instance of this class is created, the Global interpreter
// Lock on python is activated. Once the descructor is called, the
// lock is removed.
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
