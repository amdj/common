/*
 * air_test.cpp
 *
 *  Created on: Sep 3, 2013
 *      Author: anne
 */

#include "gas.h"

using namespace std;
int main() {

    double T0=293.15;
    double p0=40*101325;
    //dvar *T=new dvar(6,T0);
    //dvar *p=new dvar(6,p0);

    mat::mat air = mat::mat("helium");
    /*	//double *cp = air.cp(T);
    	//dvar *kappa = air.kappa(T);
    	//dvar *rho = air.rho(T,p);

    	//cp->show();
    	//double bla=air.cp(T0);
    	//cout << bla << endl;
    	//kappa->show();
    	//rho->show();
    */
    cout << air.cp(300) << endl;
    cin.get();
    return 0;
}
