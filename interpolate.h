#ifndef INTERPOLATE_H
#define INTERPOLATE_H
#include "vtypes.h"

#include <fstream>

#include <gsl/gsl_interp.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>


namespace math_common{

// using std::ifstream;

// class data{
// 	public:
// 		data(string filename,us ncols);
// 		~data();
// 		vd getdata(us colnr) {return datamat.col(colnr); }

// 	protected:


// 		string filename;
// 		us ncols;
// 		us size=0;

// 		void loaddata();
// 		dmat datamat;
// };


// class interpolate
// {
// 	public:
// 		interpolate(string datafile);
// 		interpolate(vd xx,vd yy);
// 		d operator()(d x);
// 		virtual ~interpolate();
// 	protected:
// 	private:
// 		void alloc_interpolator();
// 		us size;
// 		vd xx,yy;
// 		gsl_interp_accel *acc;
// 		gsl_spline* spline;
// };

}
#endif // INTERPOLATE_H
