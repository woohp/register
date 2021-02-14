#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;
using pybind11::array_t;

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

array_t<uint16_t> hamming_distance_matrix(array_t<uint8_t> X, array_t<uint8_t> Y)
{
    auto X_a = X.unchecked<2>();
    auto Y_a = Y.unchecked<2>();

    auto result = array_t<uint16_t>({ X_a.shape(0), Y_a.shape(0) });
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

std::pair<array_t<float>, array_t<float>> generate_quadratic_remap_maps(int height, int width, array_t<float> T)
{
    array_t<float> map_x({ height, width });
    array_t<float> map_y({ height, width });

    // create the accessor objects
    auto T_a = T.unchecked<2>();
    auto map_x_a = map_x.mutable_unchecked<2>();
    auto map_y_a = map_y.mutable_unchecked<2>();

    for (int y = 0; y < height; y++)
    {
        const float yy = y * y;

        for (int x = 0; x < width; x++)
        {
            const auto xx = x * x;
            const auto xy = x * y;

            map_x_a(y, x)
                = (T_a(0, 0) + T_a(1, 0) * x + T_a(2, 0) * y + T_a(3, 0) * xx + T_a(4, 0) * xy + T_a(5, 0) * yy);
            map_y_a(y, x)
                = (T_a(0, 1) + T_a(1, 1) * x + T_a(2, 1) * y + T_a(3, 1) * xx + T_a(4, 1) * xy + T_a(5, 1) * yy);
        }
    }

    return { map_x, map_y };
}

PYBIND11_MODULE(_ext, m)
{
    m.doc() = "Hamming Distance Matrix";

    m.def("hamming_distance_matrix", &hamming_distance_matrix, "", "X"_a, "Y"_a);
    m.def("generate_quadratic_remap_maps", &generate_quadratic_remap_maps, "", "height"_a, "width"_a, "T"_a);
}
