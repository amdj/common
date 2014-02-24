#include "material.h"

namespace gases {

idealgas::idealgas() {
    cpc[0]=0;
    cpc[1]=0;
    cpc[2]=0;
    cpc[3]=0;
    cpc[4]=0;
    Rs = 1;
}
idealgas::~idealgas() {}

double idealgas::rho(double T,double p) {
	checkzero(T);
    return p/Rs/T;
}
double idealgas::p(double T,double rho) {
    return rho*Rs*T;
}
double idealgas::cp(double T) {
    return cpc[0]+cpc[1]*T+cpc[2]*pow(T,2)+cpc[3]*pow(T,3)+cpc[4]*pow(T,4);
}
double idealgas::pr(double T) {
    return mu(T)*cp(T)/kappa(T);
}
double idealgas::beta(double T) {
	checkzero(T);
    return 1/T;
}
double idealgas::cm(double T) {
    double csq=gamma(T)*Rs*T;
    return sqrt(csq);
}
double idealgas::gamma(double T) {
	checkzero(T);
    return cp(T)/cv(T);
}
double idealgas::cv(double T) {
    return cp(T)-Rs;
}
double idealgas::e(double T) {
    return h(T)-Rs*T;
}
double idealgas::h(double T) {
    return cpc[0]*T+0.5*cpc[1]*pow(T,2)+(1/3.0)*cpc[2]*pow(T,3)+cpc[3]*0.25*pow(T,4)+cpc[4]*(0.2)*pow(T,5);
}

vd idealgas::rho(vd &T,vd &p) {
	checkzero(T);
    return p/Rs/T;
}
vd idealgas::rho(vd &T,double p) {
	checkzero(T);
    return p/Rs/T;
}
vd idealgas::p(vd& T,vd& rho) {
    return rho%(Rs*T);
}
vd idealgas::cp(vd& T) {
    return cpc[0]+cpc[1]*T+cpc[2]*pow(T,2)+cpc[3]*pow(T,3)+cpc[4]*pow(T,4);
}
vd idealgas::pr(vd& T) {
    return mu(T)%cp(T)/kappa(T);
}
vd idealgas::beta(vd& T) {
    return 1/T;
}
vd idealgas::cm(vd& T) {
    vd csq=Rs*T%gamma(T);
    return sqrt(csq);
}
vd idealgas::gamma(vd& T) {
    return cp(T)/cv(T);
}
vd idealgas::cv(vd& T) {
    return cp(T)-Rs;
}
vd idealgas::e(vd& T) {
    return h(T)-Rs*T;
}
vd idealgas::h(vd& T) {
    return cpc[0]*T+0.5*cpc[1]*pow(T,2)+(1/3.0)*cpc[2]*pow(T,3)+cpc[3]*0.25*pow(T,4)+cpc[4]*(0.2)*pow(T,5);
}



air::air() {
//     TODO Auto-generated constructor stub
    cpc[0]=1047.63657;
    cpc[1]=-0.372589265;
    cpc[2]=9.45304214E-4;
    cpc[3]=-6.02409443E-7;
    cpc[4]=1.2858961E-10 ;
    kappac[0]=-0.00227583562;
    kappac[1]=1.15480022E-4;
    kappac[2]=-7.90252856E-8;
    kappac[3]=4.11702505E-11;
    kappac[4]=-7.43864331E-15;
    muc[0]=-8.38278E-7;
    muc[1]=8.35717342E-8;
    muc[2]=-7.69429583E-11;
    muc[3]=4.6437266E-14;
    muc[4]=-1.06585607E-17;
    Rs = 287;
}
air::~air() {}
vd air::kappa(vd& T) {
    return kappac[0]+kappac[1]*T+kappac[2]*pow(T,2)+kappac[3]*pow(T,3)+kappac[4]*pow(T,4);
}
double air::kappa(double T) {
    return kappac[0]+kappac[1]*T+kappac[2]*pow(T,2)+kappac[3]*pow(T,3)+kappac[4]*pow(T,4);
}
vd air::mu(vd& T) {
    return muc[1]*T+muc[2]*pow(T,2)+muc[3]*pow(T,3)+muc[4]*pow(T,4)+muc[0];
}
double air::mu(double T) {
    return muc[1]*T+muc[2]*pow(T,2)+muc[3]*pow(T,3)+muc[4]*pow(T,4)+muc[0];
}

helium::helium() {
    // TODO Auto-generated constructor stub
    cpc[0] = 5195;
    cpc[1]=0;
    cpc[2]=0;
    cpc[3]=0;
    cpc[4]=0;
    Rs = 2077;
}
helium::~helium() {
    // TODO Auto-generated destructor stub
}
double helium::mu(double T) {
    return 0.412e-6*pow(T,0.68014);
}
double helium::kappa(double T) {
    return 0.0025672*pow(T,0.716);
}
vd helium::mu(vd& T) {
    return 0.412e-6*pow(T,0.68014);
}
vd helium::kappa(vd& T) {
    return 0.0025672*pow(T,0.716);
}


} /* namespace gases */


namespace solids {


//Container class
Solid::Solid(string name){
	TRACE(3,"solid constructor called");

	if(name.compare("stainless")==0){
		sol=new stainless();
		TRACE(3,"Solid set to stainless");
	}
	else if(name.compare("copper")==0){
		sol=new copper();
		TRACE(3,"Solid set to copper");

	}
	else if(name.compare("kapton")==0){
		sol=new kapton();
		TRACE(3,"Solid set to kapton");
	}
	else {
		cerr << "Error: no matching solid material found with: " << name << endl;
		exit(1);
		sol=new stainless();

	}
}

vd Solid::kappa(vd& T){
#ifdef ANNE_DEBUG
	cout << "Solid--> kappa called" << endl;
	cout << sol << endl;
#endif
	vd res=sol->kappa(T);
	return res;
}
d Solid::kappa(d T){
	return sol->kappa(T);
}
vd Solid::cs(vd& T){
	return sol->cs(T);
}
d Solid::cs(d T){
	return sol->cs(T);
}
vd Solid::rho(vd& T){
	return sol->rho(T);
}
d Solid::rho(d T){
	return sol->rho(T);
}
Solid::~Solid(){
	//dtor
	delete sol;
}


//Stainless steel
stainless::stainless(){
#if ANNE_DEBUG
	cout << "Stainless constructor called" << endl;
#endif

}
stainless::~stainless(){
	#if ANNE_DEBUG
	cout << "Stainless destructor called" << endl;
	#endif
}
vd stainless::cs(vd& T){

	vd arg=1.7054e-6*pow(T,-0.88962)+22324.0/pow(T,6);
	return pow(arg,-1.0/3.0)+15/T;

}
d stainless::cs(d T){

	d arg=1.7054e-6*pow(T,-0.88962)+22324.0/pow(T,6);
	return pow(arg,-1.0/3.0)+15/T;

}
vd stainless::rho(vd& T){
	return 8274.55-1055.23*exp(-1.0*pow((T-273.15-2171.05)/2058.08,2));
}
d stainless::rho(d T){
	return 8274.55-1055.23*exp(-1.0*pow((T-273.15-2171.05)/2058.08,2));
}
vd stainless::kappa(vd& T) {
#if ANNE_DEBUG
	cout << "Solid--> stainless kappa called" << endl;
#endif
	return pow(266800.0*pow(T,-5.2)+0.21416*pow(T,-1.6),-0.25);
}
d stainless::kappa(d T) {
	return pow(266800.0*pow(T,-5.2)+0.21416*pow(T,-1.6),-0.25);
}

//Copper
copper::copper(){}
copper::~copper(){}
vd copper::kappa(vd& T){
	return 398.0-0.567*(T-300.0);
}
d copper::kappa(d T){ return 398.0-0.567*(T-300.0); }
vd copper::cs(vd& T){return 420.0*pow(T,0);}
d copper::cs(d T){return 420.0*pow(T,0);}
vd copper::rho(vd& T){return 9000.0*pow(T,0);}
d copper::rho(d T){return 9000.0*pow(T,0);}

} //namespace solids

