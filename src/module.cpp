#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

int hamming_distance(const uint64_t* const x, const uint64_t* const y)
{
    uint64_t count[4] = { 0, 0, 0, 0 };

    // Inline assembly documentation: https://www.ibiblio.org/gferg/ldp/GCC-Inline-Assembly-HOWTO.html#s4
    // Implementation based on https://danluu.com/assembly-intrinsics/
    // Note: there may be some compilation issues on certain CPUs that do not have sufficient registers
    // to hold all the intermediate xor results.
    // TODO: make a general path that does not use inline assembly
    __asm__("popcnt %4, %4    \n\t"
            "add %4, %0       \n\t"
            "popcnt %5, %5    \n\t"
            "add %5, %1       \n\t"
            "popcnt %6, %6    \n\t"
            "add %6, %2       \n\t"
            "popcnt %7, %7    \n\t"
            "add %7, %3       \n\t"
            "popcnt %8, %8    \n\t"
            "add %8, %0       \n\t"
            "popcnt %9, %9    \n\t"
            "add %9, %1       \n\t"
            "popcnt %10, %10  \n\t"
            "add %10, %2      \n\t"
            "popcnt %11, %11  \n\t"
            "add %11, %3      \n\t"  // +r means input/output, r means intput
            : "+r"(count[0]), "+r"(count[1]), "+r"(count[2]), "+r"(count[3])
            : "r"(x[0] ^ y[0]),
              "r"(x[1] ^ y[1]),
              "r"(x[2] ^ y[2]),
              "r"(x[3] ^ y[3]),
              "r"(x[4] ^ y[4]),
              "r"(x[5] ^ y[5]),
              "r"(x[6] ^ y[6]),
              "r"(x[7] ^ y[7]));

    return static_cast<int>(count[0] + count[1] + count[2] + count[3]);
}

py::array_t<uint16_t> hamming_distance_matrix(py::array_t<uint8_t> X, py::array_t<uint8_t> Y)
{
    auto X_a = X.unchecked<2>();
    auto Y_a = Y.unchecked<2>();

    auto result = py::array_t<uint16_t>({ X_a.shape(0), Y_a.shape(0) });
    auto result_a = result.mutable_unchecked<2>();

    for (py::ssize_t i = 0; i < X_a.shape(0); i++)
    {
        for (py::ssize_t j = 0; j < Y_a.shape(0); j++)
        {
            result_a(i, j) = hamming_distance(
                reinterpret_cast<const uint64_t*>(&X_a(i, 0)), reinterpret_cast<const uint64_t*>(&Y_a(j, 0)));
        }
    }

    return result;
}

PYBIND11_MODULE(_ext, m)
{
    m.doc() = "Hamming Distance Matrix";

    m.def("hamming_distance_matrix", &hamming_distance_matrix, "", "X"_a, "Y"_a);
}
