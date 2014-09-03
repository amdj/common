#pragma once
#ifndef LOGGER_H
#define LOGGER_H
#include "consolecolors.h"
#include <iostream>
#include <string.h>

#ifndef __cplusplus
#error The c++ compiler should be used.
#endif

// Not so interesting part
#define rawstr(x) #x
#define namestr(x) rawstr(x)
#define annestr(x) namestr(x)
#define FILEWITHOUTPATH ( strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__ )
#define POS FILEWITHOUTPATH << ":" << __LINE__ <<  ": "
// End not so interesting part

#define RAWWARNING(a) cout << red << a << def << "\n";
#define WARN(a) RAWWARNING(POS << "WARNING: " << a)
#define FATAL(a) WARN(a); abort(); 



// **************************************** Tracer code
// If PP variable TRACER is not defined, we automatically set it on.
#ifndef TRACER
   #define TRACER 1
// This pp var is used to increase the level of all traces in one translational unit
#endif

#ifndef TRACERPLUS
#define TRACERPLUS (0)
#endif


#ifndef TRACERNAME
#warning TRACERNAME name not set, sol TRACERNAME set to 'defaulttracer'
#define TRACERNAME defaulttracer
#endif

// Initialize MAXTRACELEVEL and BUILDINTRACELEVEL
#ifndef MAXTRACELEVEL
#define MAXTRACELEVEL (5000) 	// To which the TRACELEVEL initially is set
#endif
// Define this preprocessor definition to overwrite
// Use -O flag for compiler to remove the dead functions!
// In that case all cout's for TRACE() are removed from code
#ifndef BUILDINTRACELEVEL
#define BUILDINTRACELEVEL (-10)
#endif

#if TRACER == 1

   extern int TRACERNAME;
// Use this preprocessor command to introduce one TRACERNAME integer per unit
/* Introduce one static logger */
// We trust that the compiler will eliminate 'dead code', which means
// that if variable BUILDINTRACERLEVEL is set, the inner if statement
// will not be reached.
   #define WRITETRACE(l,a)				\
     if(l>=BUILDINTRACELEVEL)\
       if(l>=TRACERNAME)				\
         std::cout << a << "\n"; 

   #define TRACETHIS		\
     int TRACERNAME=MAXTRACELEVEL;

   #define TRACE(l,a) WRITETRACE(l+TRACERPLUS,"TRACE - " << (l) <<  " - " << annestr(TRACERNAME) << " - " << POS << a)

   #define inittrace(ll)								\
     std::cout << "inittrace with tracelevel " << ll << " for " << annestr(TRACERNAME) << "\n"; \
     TRACERNAME=ll;

   #define initothertrace(ll,mylogger)						\
     std::cout << "Inittrace with loglevel " << ll << " for " << annestr(mylogger) << "\n"; \
     extern int mylogger; \
     mylogger=ll;

#else  // TRACER !=1

   #define TRACETHIS
   #define TRACE(l,a)
   #define inittrace(a)
   #define initothertrace(a,mylogger)
#endif	// ######################################## TRACER ==1

#endif	// LOGGER_H

