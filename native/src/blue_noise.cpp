/**
*  MIT License
*
*  Copyright (c) 2025 Anton Schreiner
*
*  Permission is hereby granted, free of charge, to any person obtaining a copy
*  of this software and associated documentation files (the "Software"), to deal
*  in the Software without restriction, including without limitation the rights
*  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*  copies of the Software, and to permit persons to whom the Software is
*  furnished to do so, subject to the following conditions:
*
*  The above copyright notice and this permission notice shall be included in all
*  copies or substantial portions of the Software.
*
*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
*  SOFTWARE.
*/

#include "common.h"

// https://eheitzresearch.wordpress.com/762-2/

namespace blue_noise_128x128_2d2d2d2d_1spp {
#include <samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_1spp.cpp>

static uint32_t const *sobol_ptr            = (uint32_t const *)sobol_256spp_256d;
static uint32_t const  sobol_size           = sizeof(sobol_256spp_256d);
static uint32_t const *scrambling_tile_ptr  = (uint32_t const *)scramblingTile;
static uint32_t const  scrambling_tile_size = sizeof(scramblingTile);
static uint32_t const *ranking_tile_ptr     = (uint32_t const *)rankingTile;
static uint32_t const  ranking_tile_size    = sizeof(rankingTile);
} // namespace blue_noise_128x128_2d2d2d2d_1spp

namespace blue_noise_128x128_2d2d2d2d_2spp {
#include <samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_2spp.cpp>

static uint32_t const *sobol_ptr            = (uint32_t const *)sobol_256spp_256d;
static uint32_t const  sobol_size           = sizeof(sobol_256spp_256d);
static uint32_t const *scrambling_tile_ptr  = (uint32_t const *)scramblingTile;
static uint32_t const  scrambling_tile_size = sizeof(scramblingTile);
static uint32_t const *ranking_tile_ptr     = (uint32_t const *)rankingTile;
static uint32_t const  ranking_tile_size    = sizeof(rankingTile);
} // namespace blue_noise_128x128_2d2d2d2d_2spp

namespace blue_noise_128x128_2d2d2d2d_4spp {
#include <samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_4spp.cpp>

static uint32_t const *sobol_ptr            = (uint32_t const *)sobol_256spp_256d;
static uint32_t const  sobol_size           = sizeof(sobol_256spp_256d);
static uint32_t const *scrambling_tile_ptr  = (uint32_t const *)scramblingTile;
static uint32_t const  scrambling_tile_size = sizeof(scramblingTile);
static uint32_t const *ranking_tile_ptr     = (uint32_t const *)rankingTile;
static uint32_t const  ranking_tile_size    = sizeof(rankingTile);
} // namespace blue_noise_128x128_2d2d2d2d_4spp

namespace blue_noise_128x128_2d2d2d2d_8spp {
#include <samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_8spp.cpp>

static uint32_t const *sobol_ptr            = (uint32_t const *)sobol_256spp_256d;
static uint32_t const  sobol_size           = sizeof(sobol_256spp_256d);
static uint32_t const *scrambling_tile_ptr  = (uint32_t const *)scramblingTile;
static uint32_t const  scrambling_tile_size = sizeof(scramblingTile);
static uint32_t const *ranking_tile_ptr     = (uint32_t const *)rankingTile;
static uint32_t const  ranking_tile_size    = sizeof(rankingTile);
} // namespace blue_noise_128x128_2d2d2d2d_8spp

namespace blue_noise_128x128_2d2d2d2d_16spp {
#include <samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_16spp.cpp>

static uint32_t const *sobol_ptr            = (uint32_t const *)sobol_256spp_256d;
static uint32_t const  sobol_size           = sizeof(sobol_256spp_256d);
static uint32_t const *scrambling_tile_ptr  = (uint32_t const *)scramblingTile;
static uint32_t const  scrambling_tile_size = sizeof(scramblingTile);
static uint32_t const *ranking_tile_ptr     = (uint32_t const *)rankingTile;
static uint32_t const  ranking_tile_size    = sizeof(rankingTile);
} // namespace blue_noise_128x128_2d2d2d2d_16spp

namespace blue_noise_128x128_2d2d2d2d_32spp {
#include <samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_32spp.cpp>

static uint32_t const *sobol_ptr            = (uint32_t const *)sobol_256spp_256d;
static uint32_t const  sobol_size           = sizeof(sobol_256spp_256d);
static uint32_t const *scrambling_tile_ptr  = (uint32_t const *)scramblingTile;
static uint32_t const  scrambling_tile_size = sizeof(scramblingTile);
static uint32_t const *ranking_tile_ptr     = (uint32_t const *)rankingTile;
static uint32_t const  ranking_tile_size    = sizeof(rankingTile);
} // namespace blue_noise_128x128_2d2d2d2d_32spp

namespace blue_noise_128x128_2d2d2d2d_64spp {
#include <samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_64spp.cpp>

static uint32_t const *sobol_ptr            = (uint32_t const *)sobol_256spp_256d;
static uint32_t const  sobol_size           = sizeof(sobol_256spp_256d);
static uint32_t const *scrambling_tile_ptr  = (uint32_t const *)scramblingTile;
static uint32_t const  scrambling_tile_size = sizeof(scramblingTile);
static uint32_t const *ranking_tile_ptr     = (uint32_t const *)rankingTile;
static uint32_t const  ranking_tile_size    = sizeof(rankingTile);
} // namespace blue_noise_128x128_2d2d2d2d_64spp

namespace blue_noise_128x128_2d2d2d2d_128spp {
#include <samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_128spp.cpp>

static uint32_t const *sobol_ptr            = (uint32_t const *)sobol_256spp_256d;
static uint32_t const  sobol_size           = sizeof(sobol_256spp_256d);
static uint32_t const *scrambling_tile_ptr  = (uint32_t const *)scramblingTile;
static uint32_t const  scrambling_tile_size = sizeof(scramblingTile);
static uint32_t const *ranking_tile_ptr     = (uint32_t const *)rankingTile;
static uint32_t const  ranking_tile_size    = sizeof(rankingTile);
} // namespace blue_noise_128x128_2d2d2d2d_128spp

namespace blue_noise_128x128_2d2d2d2d_256spp {
#include <samplerCPP/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_256spp.cpp>

static uint32_t const *sobol_ptr            = (uint32_t const *)sobol_256spp_256d;
static uint32_t const  sobol_size           = sizeof(sobol_256spp_256d);
static uint32_t const *scrambling_tile_ptr  = (uint32_t const *)scramblingTile;
static uint32_t const  scrambling_tile_size = sizeof(scramblingTile);
static uint32_t const *ranking_tile_ptr     = (uint32_t const *)rankingTile;
static uint32_t const  ranking_tile_size    = sizeof(rankingTile);
} // namespace blue_noise_128x128_2d2d2d2d_256spp

PYBIND11_MODULE(blue_noise, m) {

    m.def("_128x128_2d2d2d2d_256spp_sobol_ptr", []() { return (uint64_t)blue_noise_128x128_2d2d2d2d_256spp::sobol_ptr; });
    m.def("_128x128_2d2d2d2d_256spp_sobol_size_bytes", []() { return (uint64_t)blue_noise_128x128_2d2d2d2d_256spp::sobol_size; });
    m.def("_128x128_2d2d2d2d_256spp_scrambling_tile_ptr", []() { return (uint64_t)blue_noise_128x128_2d2d2d2d_256spp::scrambling_tile_ptr; });
    m.def("_128x128_2d2d2d2d_256spp_scrambling_tile_size_bytes", []() { return (uint64_t)blue_noise_128x128_2d2d2d2d_256spp::scrambling_tile_size; });
    m.def("_128x128_2d2d2d2d_256spp_ranking_tile_ptr", []() { return (uint64_t)blue_noise_128x128_2d2d2d2d_256spp::ranking_tile_ptr; });
    m.def("_128x128_2d2d2d2d_256spp_ranking_tile_size_bytes", []() { return (uint64_t)blue_noise_128x128_2d2d2d2d_256spp::ranking_tile_size; });

}