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
  
  void insertInRowFromLeftSide(esdmat& Mat,const vd& data,us rownr){
    TRACE(15,"InserInRowFromLeftSide");
    assert(data.size()<=Mat.cols());
    assert(rownr<Mat.rows());
    us cursize=Mat.nonZeros();
    us newsize=cursize+data.size(); // Allocates more than enough
    us datasize=data.size();
    Mat.reserve(newsize);
    for(us i=0;i<datasize;i++){
      if(data(i)!=0)
	Mat.insert(rownr,i)=data(i);
    } // for
  
  } // insertInRowFromLeftSide

  void insertSubMatrixInMatrixTopLeft(esdmat& target,const esdmat& source) {
    TRACE(15,"insertSubMatrixInMatrixTopLeft()");
    assert(source.cols()<=target.cols());
    assert(source.rows()<=target.rows());
    
    vtriplet tr=math_common::getTriplets(source);


  }
  
  vtriplet getTriplets(const esdmat & mat){
    //only for ColMajor Sparse Matrix
    assert(mat.rows()==mat.cols());
    int size=mat.rows();
    int i,j,currOuterIndex,nextOuterIndex;
    vtriplet tripletList;
    tripletList.reserve(mat.nonZeros());
    for(j=0; j<size; j++){
      currOuterIndex = mat.outerIndexPtr()[j];
      nextOuterIndex = mat.outerIndexPtr()[j+1];

      for(int a = currOuterIndex; a<nextOuterIndex; a++){
	i=mat.innerIndexPtr()[a];
	if(i < 0) continue;
	if(i >= size) break;
	tripletList.push_back(triplet(i,j,mat.valuePtr()[a]));
      } // inner for
    } // for
    return tripletList;
  } // getTriplets

  vtriplet getTripletsBlock(const esdmat& mat,us startrow,us startcol,us nrows,us ncols){
    assert(startrow+nrows <= mat.rows());
    assert(startcol+ncols <= mat.cols());
    us Mj,Mi,i,j,currOuterIndex,nextOuterIndex;
    vtriplet tripletList;
    tripletList.reserve(mat.nonZeros());

    for(j=0; j<ncols; j++){
      Mj=j+startcol;
      currOuterIndex = mat.outerIndexPtr()[Mj];
      nextOuterIndex = mat.outerIndexPtr()[Mj+1];

      for(us a = currOuterIndex; a<nextOuterIndex; a++){
	Mi=mat.innerIndexPtr()[a];

	if(Mi < startrow) continue;
	if(Mi >= startrow + nrows) break;

	i=Mi-startrow;    
	tripletList.push_back(triplet(i,j,mat.valuePtr()[a]));
      }
    }
    return tripletList;
  }

  void shiftTriplets(vtriplet& triplets,int nrows,int ncols){
    TRACE(15,"shiftTriplets()");
    us size=triplets.size();
    for(us j=0;j<size;j++){
      const_cast<int&>(triplets[j].col())=triplets[j].col()+ncols;
      const_cast<int&>(triplets[j].row())=triplets[j].row()+nrows;
    }
  }
  void multiplyTriplets(vtriplet& triplets,d factor){
    TRACE(15,"multiplyTriplets()");
    us size=triplets.size();
    for(us j=0;j<size;j++){
      const_cast<d&>(triplets[j].value())=triplets[j].value()*factor;
    }
  }
  void reserveExtraDofs(vtriplet& trip,us n){
    TRACE(15,"reserveExtraDofs()");
    us cursize=trip.size();
    trip.reserve(cursize+n);
  }
  
  
} // namespace math_common



