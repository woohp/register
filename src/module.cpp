#include <bit>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;
using pybind11::array_t;

array_t<uint16_t>
hamming_distance_matrix(array_t<uint8_t, py::array::c_style> X, array_t<uint8_t, py::array::c_style> Y)
{
    auto X_a = X.unchecked<2>();
    auto Y_a = Y.unchecked<2>();

    auto result = array_t<uint16_t>({ X_a.shape(0), Y_a.shape(0) });
    auto result_a = result.mutable_unchecked<2>();

    for (py::ssize_t i = 0; i < X_a.shape(0); i++)
    {
        for (py::ssize_t j = 0; j < Y_a.shape(0); j++)
        {
            using std::popcount;
            const uint64_t* const __restrict__ x = reinterpret_cast<const uint64_t*>(&X_a(i, 0));
            const uint64_t* const __restrict__ y = reinterpret_cast<const uint64_t*>(&Y_a(j, 0));
            result_a(i, j) = popcount(x[0] ^ y[0]) + popcount(x[1] ^ y[1]) + popcount(x[2] ^ y[2])
                + popcount(x[3] ^ y[3]) + popcount(x[4] ^ y[4]) + popcount(x[5] ^ y[5]) + popcount(x[6] ^ y[6])
                + popcount(x[7] ^ y[7]);
        }
    }

    return result;
}

std::pair<array_t<float>, array_t<float>>
generate_quadratic_remap_maps(int height, int width, array_t<float, py::array::c_style> T)
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

std::vector<uint64_t> sord(const uint64_t k, const uint64_t sk)
{
    // implementation copied from here:
    // https://math.stackexchange.com/a/1370461
    std::vector<uint64_t> idx(k);
    if (k == 0)
        return idx;
    else if (k == 1)
    {
        idx[0] = sk;
        return idx;
    }

    uint64_t s = sk;
    // Find j such that binomial(j+k-1,k) <= s < binomial(j+k,k)
    uint64_t j = 0;
    uint64_t prev_b = 0;
    uint64_t b = 1;
    while (b <= s)
    {
        // Maintain invariants:
        // prev_b == binomial(j+k-1,k), b == binomial(j+k,k)
        prev_b = b;
        j += 1;
        b *= j + k;
        b /= j;
    }
    b = prev_b;

    // j is now the greatest index occurring in the tuple.
    // From now on, we will search backwards, decreasing j.
    for (int r = k - 1; r > 1; r--)
    {
        // b == binomial(j+r,r+1) <= s < binomial(j+r+1,r+1)
        idx[r] = j;
        s -= b;
        // Update to b = binomial(j+r-1,r)
        b *= r + 1;
        b /= j + r;
        // Find j such that binomial(j+r-1,r) <= s < binomial(j+r,r)
        while (b > s)
        {
            j -= 1;
            b *= j;
            b /= j + r;
        }
    }
    // Simplified handling of r = 1
    // b == binomial(j+r,r+1) <= s < binomial(j+r+1,r+1)
    idx[1] = j;
    s -= b;

    // Simplified handling of r = 0
    // binomial(j+r,r+1) == s iff j == s
    idx[0] = s;
    return idx;
}

PYBIND11_MODULE(_ext, m)
{
    m.doc() = "Utility functions implemented in C++ for greater speed";

    m.def("hamming_distance_matrix", &hamming_distance_matrix, "", "X"_a.noconvert(), "Y"_a.noconvert());
    m.def(
        "generate_quadratic_remap_maps", &generate_quadratic_remap_maps, "", "height"_a, "width"_a, "T"_a.noconvert());
    m.def("sord", &sord, "", "k"_a, "sk"_a);
}
