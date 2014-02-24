#pragma once
#include "vtypes.h"
#ifndef LOGGER_H
#define LOGGER_H

#define rawstr(x) #x
#define namestr(x) rawstr(x)
#define annestr(x) namestr(x)


#define MAXLOGLEVEL 50000
/* The logger */
extern int LogLevel;

/*

For traces, we have different levels:
0: ALL Traces
1: Not in deepest loops
2: Subloops
3: Main loops, initialization

9: Highest TRACE level

10: DEBUG

19: highest DEBUG log
20: INFO

30: WARNING
*/

#ifndef ANNELOGGER
#error "Variable ANNELOGGER undefined"
#endif
/* Introduce one static logger */


#define FILE (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#define POS FILE << ":" << __LINE__ <<  ": "
#define LOG(l,a) if(l>=LogLevel) { \
	cout << a << endl; }

//else { cout << "LogLevel:" << LogLevel << a << endl;}

//cout << "Loglevel:" << LogLevel << endl;


#define TRACELOG(a) TRACE(9,a)
#define TRACE(l,a) LOG(l,"TRACE" << (l) <<  " - " << POS << a)

#define DEBUGLOG(a) DBG(19,a)
#define DBG(l,a) LOG(l,"DEBUG - " << POS << a)

#define RAWWARNINGLOG(a) cout << "WARNING " << a
#define WARNING(a) RAWWARNINGLOG(FILE  << __LINE__ <<  ": " a)



#define RAWDEBUGLOG(a) LOG(10,"DEBUG - " << a)



#define FATAL(a) TRACE(100,a) \



inline void initlog(int loglevel){
		cout << "Initlog called with loglevel " << loglevel << endl;
		LogLevel=loglevel;
		DBG(10,"Loglevel " << annestr(ANNELOGGER) << " set to " << LogLevel);
}
#else
#define TRACELOG(a)
#define TRACE(l,a)

#define DEBUGLOG(a)
#define DBG(l,a)

#define RAWWARNINGLOG(a)
#define WARNING(a)


#endif
