#include <stdexcept>

class MyError : public std::runtime_error {
public:
  MyError(const string& msg = "") : runtime_error(msg) {}
};
