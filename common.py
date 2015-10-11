# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.7
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.





from sys import version_info
if version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_common', [dirname(__file__)])
        except ImportError:
            import _common
            return _common
        if fp is not None:
            try:
                _mod = imp.load_module('_common', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _common = swig_import_helper()
    del swig_import_helper
else:
    import _common
del version_info
try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.


def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr_nondynamic(self, class_type, name, static=1):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    if (not static):
        return object.__getattr__(self, name)
    else:
        raise AttributeError(name)

def _swig_getattr(self, class_type, name):
    return _swig_getattr_nondynamic(self, class_type, name, 0)


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object:
        pass
    _newclass = 0



def setCommonTracer(arg1: 'int') -> "void":
    return _common.setCommonTracer(arg1)
setCommonTracer = _common.setCommonTracer
class RottFuncs(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, RottFuncs, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, RottFuncs, name)
    __repr__ = _swig_repr

    def __init__(self, *args):
        this = _common.new_RottFuncs(*args)
        try:
            self.this.append(this)
        except:
            self.this = this
    __swig_destroy__ = _common.delete_RottFuncs
    __del__ = lambda self: None

    def fx(self, *args) -> "c":
        return _common.RottFuncs_fx(self, *args)

    def getCShape(self) -> "string":
        return _common.RottFuncs_getCShape(self)
RottFuncs_swigregister = _common.RottFuncs_swigregister
RottFuncs_swigregister(RottFuncs)

class Gas(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Gas, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Gas, name)
    __repr__ = _swig_repr

    def __init__(self, *args):
        this = _common.new_Gas(*args)
        try:
            self.this.append(this)
        except:
            self.this = this
    __swig_destroy__ = _common.delete_Gas
    __del__ = lambda self: None

    def pr(self, *args) -> "d":
        return _common.Gas_pr(self, *args)

    def gamma(self, *args) -> "d":
        return _common.Gas_gamma(self, *args)

    def Rs(self) -> "d":
        return _common.Gas_Rs(self)

    def rho(self, *args) -> "d":
        return _common.Gas_rho(self, *args)

    def p(self, *args) -> "d":
        return _common.Gas_p(self, *args)

    def cp(self, *args) -> "d":
        return _common.Gas_cp(self, *args)

    def h(self, *args) -> "d":
        return _common.Gas_h(self, *args)

    def cv(self, *args) -> "d":
        return _common.Gas_cv(self, *args)

    def e(self, *args) -> "d":
        return _common.Gas_e(self, *args)

    def beta(self, *args) -> "d":
        return _common.Gas_beta(self, *args)

    def cm(self, *args) -> "d":
        return _common.Gas_cm(self, *args)

    def mu(self, *args) -> "d":
        return _common.Gas_mu(self, *args)

    def kappa(self, *args) -> "d":
        return _common.Gas_kappa(self, *args)
Gas_swigregister = _common.Gas_swigregister
Gas_swigregister(Gas)

class Solid(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Solid, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Solid, name)
    __repr__ = _swig_repr

    def __init__(self, *args):
        this = _common.new_Solid(*args)
        try:
            self.this.append(this)
        except:
            self.this = this

    def kappa(self, *args) -> "d":
        return _common.Solid_kappa(self, *args)

    def c(self, *args) -> "d":
        return _common.Solid_c(self, *args)

    def rho(self, *args) -> "d":
        return _common.Solid_rho(self, *args)
    __swig_destroy__ = _common.delete_Solid
    __del__ = lambda self: None
Solid_swigregister = _common.Solid_swigregister
Solid_swigregister(Solid)

# This file is compatible with both classic and new-style classes.


