

%{
  #include <exception>
  
%}

%typemap(throws) std::exception %{
  PyErr_SetString(PyExc_RuntimeError, $1.what());
  SWIG_fail;
%}
