#include "cpp11/integers.hpp"
using namespace cpp11;

[[cpp11::register]]
int cpp11_scalar() {
    return 99;
}
