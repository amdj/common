#ifndef VTYPES_H
#define VTYPES_H



#include <iostream>
#include <algorithm>
#include <iomanip>
#include <stdio.h>
#include <vector>
#include <string>
#include <complex>
#include <armadillo>
//#include <Eigen/Sparse>




//For disabling bounds checking (and increasing speed)
//#define ARMA_NO_DEBUG

#ifndef PRECISION
#define PRECISION 12
#endif


#ifndef ANNE_WARNINGS
#define ANNE_WARNINGS
#endif

#define pi arma::datum::pi

//typedef Matrix cmatEigen


using std::cout;
using std::endl;
using std::setprecision;
using std::setiosflags;
using std::cin;
using std::ostream;
using std::cerr;
using std::string;
using std::ios;
using std::real;
using std::vector;

inline void CoutSetup(){
	cout << setprecision(PRECISION);
	cout << setiosflags(ios::scientific);
}
template < class T >
void prn(string text,T data){
	CoutSetup();
	cout << text << data << endl;
}
inline void prn(string text){
	cout << text << endl;
}

using arma::ones;
using arma::zeros;
//using arma::vectorise;
using arma::eye;
using arma::inv;
using arma::det;
#define fillwith arma::fill
using arma::log_det;
using arma::norm;
using arma::span;
using arma::linspace;

typedef double d;
typedef std::complex<double> c;
typedef unsigned us;
typedef arma::cx_mat cmat;
typedef arma::vec vd;
typedef arma::cx_vec vc;
typedef arma::Mat<double> dmat;


inline void pyprn(string text,vd data){
	CoutSetup();
	cout << text << endl;
	cout << "n.array([" ;
	for(us i=0;i<data.size()-1;i++){
		cout << data(i) << "," <<endl;
	}
	cout << data(data.size()-1) ;
	cout << "])" <<endl;
}
inline void pyprn(string text,vc data){

	CoutSetup();

	string sign;
	cout << text << endl;
	cout << "n.array([" ;
	for(us i=0;i<data.size()-1;i++){
		if((imag(data(i))>=0.0)) { sign="+";} else { sign="";}
		cout << real(data(i)) << sign <<imag(data(i)) << "j" << "," <<endl;
	}

	us last=data.size()-1;
	if((imag(data(last))>=0.0)) { sign="+";} else { sign="";}
	cout << real(data(last)) << sign <<imag(data(last)) << "j" ;
	cout << "])" <<endl;
}
const c I=c(0,1); //Complex unity
const c minI=c(0,-1); //Minus complex unity

inline vd vdzeros(us Nelem){
	return zeros<vd>(Nelem);
}
inline vc vczeros(us Nelem){
	return zeros<vc>(Nelem);
}
namespace nanh{


}
namespace gases{

using std::cout;
using std::endl;

}
namespace variable{
using std::cout;
using std::endl;

typedef arma::vec vd;
typedef arma::cx_vec vc;

}


inline void showvec(vc vec){ //cout a vector to stout
	cout << vec << endl;
}



#include "logger.h"


#endif // VTYPES_H
