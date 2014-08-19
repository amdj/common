#include "interpolate.h"
namespace math_common{

// data::data(string filename,us ncols):filename(filename),ncols(ncols){
// //	prn("ncols:",ncols);

// 	loaddata();
// }
// void data::loaddata() {
// 	std::vector< std::vector<double> > vdata;
// 	for (us i=0;i<ncols;i++){ vdata.push_back(std::vector<double>());}
// 	ifstream file1;
// 	file1.open(filename);
// 	string line;
// //	us counter=0;
// 	if (file1.good()){
// //		prn("File ok: ",filename);
// 		std::getline(file1,line);
// 		while(file1.good()){
// 			string subline=line;
// 			us spacepos=line.find(" ");
// 			for(us j=0;j<ncols;j++){
// 				subline=line.substr(j*spacepos,spacepos);
// 				//cout << "j:" << j << ", subline:" << subline << endl;
// 				vdata[j].push_back(std::strtod(&subline[0],NULL));
// 				//if(j==0){ cout << spacepos <<endl; }
// 			}
// 			//prn("line:",counter);
// 			std::getline(file1,line);
// //			counter++;
// 		}
// 		size=vdata[0].size();
// //		cout << vdata[0][vdata[0].size()-2] << endl;
// //		cout << vdata[1][vdata[0].size()-1] << endl;
// 	}
// 	else {
// 		prn("Trouble with file ",filename);
// 		return;
// 	}
// 	file1.close();
// 	datamat=zeros<dmat>(size,ncols);
// 	for(us j=0;j<ncols;j++){
// 		datamat.col(j)=vd(vdata[j]);
// 	}


// }
// data::~data(){}


// interpolate::interpolate(string datafile){
// 	data dat=data(datafile,2);
// 	cout << "INTERPOLATION CODE NEEDS TO BE FULLY TESTED. THIS IS NOT DONE YET" << endl;
// 	xx=dat.getdata(0);
// 	yy=dat.getdata(1);
// 	size=xx.size();
// 	//cout << x2 << endl;
// 	alloc_interpolator();
// }
// interpolate::interpolate(vd xx,vd yy): yy(yy),xx(xx),size(xx.size()){
// 	//ctor
// 	//cout << "size:" << size << endl;
// 	alloc_interpolator();
// }
// void interpolate::alloc_interpolator(){
// 	acc=gsl_interp_accel_alloc();
// 	spline=gsl_spline_alloc(gsl_interp_cspline,size);
// 	gsl_spline_init(spline,xx.memptr(),yy.memptr(),size);
// }
// d interpolate::operator()(d x){
// 	return gsl_spline_eval(spline,x,acc);
// }

// interpolate::~interpolate(){
// 	//dtor
// 	//prn("Interpolate destructor called");
// 	gsl_spline_free(spline);
// 	gsl_interp_accel_free(acc);
// }


} //namespace math_anne
