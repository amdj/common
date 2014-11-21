// Copied from Jean-Pierre Moreau's Home Page
// I would like to thank him for sharing his code.
/*****************************************************************
 *    Complex Bessel Function of the 1st Kind of integer order    *
 * -------------------------------------------------------------- *
 * SAMPLE RUN:                                                    *
 *                                                                *
 * Complex Bessel Function of the 1st Kind of integer order       *
 *                                                                * 
 * Input complex argument (real imaginary): 1 2                   *
 * Input integer order: 1                                         *
 *                                                                *
 * Function value: 1.291848  1.010488                             *
 *                                                                *
 *                                                                *
 *                          C++ Release 1.2 By J-P Moreau, Paris. *
 *                                   (www.jpmoreau.fr)            *
 * -------------------------------------------------------------- *
 * Release 1.1: Corrected bug in Function ZLn (atan replaced by   *
 *              function ATAN2) 11/10/2005.                       *
 * Release 1.2: ZPower replaced by IZpower (integer exponant).    *
 *              Limitations increased in CDIV, MAXK=20            * 
 *****************************************************************/
#include <stdio.h>
#include <math.h>

#define  MAXK  20    //09/21/2009 (15 before)
#define  FALSE  0
#define  TRUE   1

double HALF = 0.5, ONE = 1.0, FPF = 5.5;
double PI=4.0*atan(1.0);

int nu;              // order of complex Bessel
double z[2], z1[2];  // Complex numbers
double x,y;

// Z=Z1/Z2
void CDIV(double *Z1, double *Z2, double *Z) {
  double D;
  D=Z2[0]*Z2[0]+Z2[1]*Z2[1];
  if (D > 1e30) {
    Z[0]=0.0; Z[1]=0.0; return;
  }
  if (D == 0) return;
  Z[0]=(Z1[0]*Z2[0]+Z1[1]*Z2[1])/D;
  Z[1]=(Z1[1]*Z2[0]-Z1[0]*Z2[1])/D;
}

// Z=Z1*Z2
void CMUL(double *Z1, double *Z2, double *Z) {
  Z[0]=Z1[0]*Z2[0] - Z1[1]*Z2[1];
  Z[1]=Z1[0]*Z2[1] + Z1[1]*Z2[0];
}

// compute Z^N
void IZPower(double *z, int n, double *z1) {
  double temp[2],temp1[2]; int i;
  if (n==0) {
    z1[0]=1.0;
    z1[1]=0.0;
  }
  else if (n==1) {
    z1[0]=z[0];
    z1[1]=z[1];
  }
  else {
    temp1[0]=z[0]; temp1[1]=z[1];
    for (i=2; i<=n; i++) {
      CMUL(temp1,z,temp);
      temp1[0]=temp[0];
      temp1[1]=temp[1];
    }
    z1[0]=temp[0];
    z1[1]=temp[1];
  }
}

double Fact(int k) {
  int i; double f;
  f=ONE;
  for (i=2; i<=k; i++)  f *= ONE*i;
  return f;
}

/******************************************
 *           FUNCTION  GAMMA(X)            *
 * --------------------------------------- *
 * Returns the value of Gamma(x) in double *
 * precision as EXP(LN(GAMMA(X))) for X>0. *
 ******************************************/
double Gamma(double xx) {
  double cof[6];
  double stp,x,tmp,ser;
  int j;
  cof[0]=76.18009173;
  cof[1]=-86.50532033;
  cof[2]=24.01409822;
  cof[3]=-1.231739516;
  cof[4]=0.120858003e-2;
  cof[5]=-0.536382e-5;
  stp=2.50662827465;
  x=xx-ONE;
  tmp=x+FPF;
  tmp=(x+HALF)*log(tmp)-tmp;
  ser=ONE;
  for (j=0; j<6; j++) {
    x += ONE;
    ser += cof[j]/x;
  }
  return (exp(tmp+log(stp*ser)));
}

// main subroutine
void CBESSJ(double *z, int nu, double *z1)  {
  /*--------------------------------------------------
    inf.     (-z^2/4)^k
    Jnu(z) = (z/2)^nu x Sum  ------------------
    k=0  k! x Gamma(nu+k+1)
    (nu must be >= 0). Here k=15.
    ---------------------------------------------------*/
  int k;
  double sum[2],tmp[2],tmp1[2],tmp2[2],tmp3[2];
  // calculate (z/2)^nu in tmp3
  tmp[0]=2.0; tmp[1]=0.0;
  CDIV(z,tmp,tmp1);
  IZPower(tmp1,nu,tmp3);
  sum[0]=0.0; sum[1]=0.0;
  //calculate Sum
  for (k=0; k<=MAXK; k++) {
    // calculate (-z^2/4)^k
    IZPower(z,2,tmp);
    tmp[0]=-tmp[0]; tmp[1]=-tmp[1];
    tmp1[0]=4.0; tmp1[1]=0.0;
    CDIV(tmp,tmp1,tmp2);

    if (z[1]==0) {             //case real number
      tmp[0]=pow(tmp2[0],k);
      tmp[1]=0.0;
    }
    else                       //case complex number
      IZPower(tmp2,k,tmp);

    // divide by k!
    tmp1[0]=Fact(k); tmp1[1]=0.0;
    if (z[1]==0) {
      tmp2[0]=tmp[0]/tmp1[0];
      tmp2[1]=0.0;
    }
    else
      CDIV(tmp,tmp1,tmp2);
    // divide by Gamma(nu+k+1)
    tmp1[0]=Gamma(ONE*(nu+k+1)); tmp1[1]=0.0;

    if (z[1]==0) {
      tmp[0]=tmp2[0]/tmp1[0];
      tmp[1]=0.0;
    }
    else
      CDIV(tmp2,tmp1,tmp);
    // actualize sum
    sum[0] += tmp[0];
    sum[1] += tmp[1];

  }
  // multiply (z/2)^nu by sum
  CMUL(tmp3,sum,z1);
}


// int main() {

//   double z1[2];

//   printf("\n Complex Bessel Function of the 1st Kind of integer order\n\n");
//   printf(" Input complex argument (real imaginary): ");
//   scanf("%lf %lf", &x, &y);
//   z[0]=x; z[1]=y;
//   printf(" Input integer order: "); scanf("%d", &nu);

//   CBESSJ(z,nu,z1);

//   printf("\n Function value: %f  %f\n\n", z1[0], z1[1]);

// }

// end of file cbessj.cpp
