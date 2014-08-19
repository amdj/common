#include "arma_eigen.h"

namespace math_common{

  esdmat ArmaToEigen(const sdmat& amat){ // Conversion from Armadillo
    TRACE(1,"ArmaToEigen(sdmat&)");
    const d* vals=amat.values;
    us n_nonzero=amat.n_nonzero;
    us ncols=amat.n_cols;
    us nrows=amat.n_rows;
    
    arma::Col<us> col_ptrs(amat.col_ptrs,ncols);
    arma::Col<us> row_indices(amat.row_indices,n_nonzero);

    us r,c;			// Current row, column
    // TRACE(1,"n_nonzero:"<< n_nonzero);
    vector<triplet> tr; tr.reserve(n_nonzero);
    // TRACE(1,"ncols:"<<ncols);
    // Data in the Armadillo sparse matrix is stored as a compressed
    // sparse column format. This means that a pointer is present
    // which points to the first nonzero value in column j as being
    // col_ptr[j]. However, this gives problems for empty columns. The
    // solution armadillo uses for this case is is that index of the
    // previous column is filled in on that place. This makes it all
    // quite complicated.
    // TRACE(-1,"colptrs:"<<arma::Col<us>(col_ptrs,ncols));

    arma::Col<us> col_indices(n_nonzero,fillwith::zeros);
    // TRACE(1,"col_indices created");
    us colptrcounter=0;
    // TRACE(1,"colptrs:"<< col_ptrs);
    for(us curcolumn=0;curcolumn<ncols;curcolumn++){
      // Check if we are not in the last column
      // TRACE(1,"Current column:"<<curcolumn);
      if(curcolumn<ncols-1){
      // First: check if next value of col_ptr is same, if so: no values in this column
	if(col_ptrs(colptrcounter)!=col_ptrs(colptrcounter+1)){
	  // OK, we can add values to this column
	  // TRACE(1,"")
	    for(us ind=col_ptrs(colptrcounter);ind<col_ptrs(colptrcounter+1);ind++){
	      // TRACE(1,"Index:"<<ind);
	      col_indices(ind)=curcolumn;
	    }
	}
      }
      else{			// Do the things for the last column
	// TRACE(-2,"Last column reached")
	  for(us ind=col_ptrs(colptrcounter);ind<n_nonzero;ind++){
	      // TRACE(1,"Index:"<<ind);	    
	      col_indices(ind)=curcolumn;
	  }
      }
      
      colptrcounter++;
      // TRACE(-1,"Curcolumn:"<<curcolumn);
    }
    // TRACE(-1,"Col indices:"<< col_indices);
    for(us n=0;n<n_nonzero;n++){
      r=row_indices(n);
      c=col_indices(n);
      TRACE(-5,"c:"<<c);
      tr.push_back(triplet(r,c,vals[n]));
    }
    // TRACE(-3,"Triplet making survived");
    esdmat result(nrows,ncols);
    TRACE(1,"ArmaToEigen: filling matrix..");
    result.setFromTriplets(tr.begin(),tr.end());
    // TRACE(-3,"Triplet setting survived");

    return result;
  }
  evd ArmaToEigen(const vd& avd){ // 
    //
    us size=avd.size();
    // Stupid hack to const-cast
    d* ptr=const_cast<d*>(avd.memptr());
    evd result(size);
    result=Eigen::Map<evd>(ptr,size);
    return result;
  }
  vd EigenToArma(const evd& Evd){
    us size=Evd.rows();
    
    vd result(&Evd[0],size);
    return result;
  }

  dmat armaView(edmat Eigenmat)  {
    return dmat(Eigenmat.data(),Eigenmat.rows(),Eigenmat.cols(),false,false);
  }
  vd armaView(evd& vec)  {
    return vd(vec.data(),vec.rows(),false,false);
  }
  
  
  
} // namespace math_common
