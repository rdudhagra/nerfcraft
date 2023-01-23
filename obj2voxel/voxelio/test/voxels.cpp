#include "voxelio/test/voxels.hpp"

#include "voxelio/log.hpp"
#include "voxelio/palette.hpp"
#include "voxelio/stringmanip.hpp"

#include <iostream>

namespace voxelio::test {

namespace {

constexpr const argb32 GOLDEN_MODEL_DATA[8 * 8 * 8] = {
    0xff14121e, 0xff3c574b, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff3c574b, 0xff27438a, 0xff3c574b,
    0xff5e877b, 0xff5e877b, 0xff5e877b, 0xff5e877b, 0xff5e877b, 0xff5e877b, 0xff3c574b, 0xff6baa97, 0xff6baa97,
    0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff5e877b, 0xff5e877b, 0xff5e877b,
    0xffab1c09, 0xffab1c09, 0xff5e877b, 0xff5e877b, 0xff5e877b, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xffab1c09,
    0xffab1c09, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff5e877b, 0xff5e877b, 0xff5e877b, 0xffab1c09, 0xffab1c09,
    0xff5e877b, 0xff5e877b, 0xff5e877b, 0xff3c574b, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xffab1c09, 0xff6baa97,
    0xff6baa97, 0xff3c574b, 0xff8593ff, 0xff3c574b, 0xff5e877b, 0xff5e877b, 0xff5e877b, 0xff5e877b, 0xff3c574b,
    0xff2f6f89, 0xff3c574b, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xff3c574b,
    0xff5e877b, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xff6baa97,
    0xff5e877b, 0xff5e877b, 0xff5e877b, 0xff5e877b, 0xff5e877b, 0xff5e877b, 0xff6baa97, 0xff5e877b, 0xff6baa97,
    0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff5e877b, 0xff6baa97, 0xff6baa97, 0xff6baa97,
    0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff5e877b, 0xff6baa97, 0xff6baa97, 0xff6baa97,
    0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff5e877b, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97,
    0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff3c574b, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97,
    0xff6baa97, 0xff3c574b, 0xff6baa97, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0xff6baa97, 0xff5e877b, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0xff6baa97, 0xff5e877b, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff5e877b, 0xff6baa97, 0xff5e877b,
    0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff5e877b, 0xff6baa97, 0xff6baa97,
    0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff5e877b, 0xff6baa97, 0xff6baa97,
    0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff5e877b, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97,
    0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff5e877b, 0xff6baa97, 0xffab1c09, 0xffab1c09, 0xffab1c09,
    0xff5e877b, 0xff6baa97, 0xff5e877b, 0xff6baa97, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0xff6baa97, 0xff5e877b, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0xff6baa97, 0xff5e877b, 0xff6baa97, 0xff5e877b, 0xff5e877b, 0xff6baa97, 0xff5e877b, 0xff6baa97,
    0xffab1c09, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xffab1c09, 0xffab1c09,
    0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xffab1c09, 0xff6baa97,
    0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff5e877b, 0xffab1c09, 0xff6baa97, 0xff6baa97,
    0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff5e877b, 0xff6baa97, 0xff5e877b, 0xffab1c09,
    0xff6baa97, 0xff5e877b, 0xff6baa97, 0xff5e877b, 0xff6baa97, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0xff6baa97, 0xff5e877b, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0xff6baa97, 0xff5e877b, 0xff6baa97, 0xff5e877b, 0xff5e877b, 0xff6baa97, 0xff5e877b,
    0xff6baa97, 0xffab1c09, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xffab1c09,
    0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xffab1c09, 0xffab1c09,
    0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xffab1c09, 0xffab1c09, 0xff6baa97,
    0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xffab1c09, 0xff5e877b, 0xff6baa97, 0xff5e877b,
    0xffab1c09, 0xff6baa97, 0xff5e877b, 0xff6baa97, 0xff5e877b, 0xff6baa97, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0xff6baa97, 0xff5e877b, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0xff6baa97, 0xff5e877b, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97,
    0xff5e877b, 0xff6baa97, 0xff5e877b, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97,
    0xff5e877b, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97,
    0xff5e877b, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff5e877b, 0xff6baa97,
    0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff5e877b, 0xff6baa97,
    0xff5e877b, 0xffab1c09, 0xff5e877b, 0xff5e877b, 0xff6baa97, 0xff5e877b, 0xff3c574b, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xff6baa97, 0xff5e877b, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xff6baa97, 0xff5e877b, 0xff5e877b, 0xff5e877b, 0xff5e877b,
    0xff5e877b, 0xff5e877b, 0xff6baa97, 0xff5e877b, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97,
    0xff6baa97, 0xff5e877b, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97,
    0xff6baa97, 0xff5e877b, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff5e877b,
    0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff3c574b,
    0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff3c574b, 0xff27438a, 0xff3c574b,
    0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff7f3eb6, 0xff3c574b, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97,
    0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xff5e877b, 0xff5e877b, 0xff5e877b, 0xffab1c09, 0xff5e877b,
    0xff5e877b, 0xff5e877b, 0xff5e877b, 0xff6baa97, 0xff6baa97, 0xff6baa97, 0xffab1c09, 0xffab1c09, 0xff6baa97,
    0xff6baa97, 0xff6baa97, 0xff5e877b, 0xff5e877b, 0xff5e877b, 0xffab1c09, 0xff5e877b, 0xff5e877b, 0xff5e877b,
    0xff5e877b, 0xff3c574b, 0xff6baa97, 0xff6baa97, 0xffab1c09, 0xffab1c09, 0xff6baa97, 0xff6baa97, 0xff3c574b,
    0xff2f6f89, 0xff3c574b, 0xff5e877b, 0xff5e877b, 0xff5e877b, 0xff5e877b, 0xff3c574b, 0xff42ae39,
};

VoxelArray makeDebugGoldenModel()
{
    VoxelArray result{8, 8, 8};
    for (usize z = 0, i = 0; z < 8; ++z) {
        for (usize y = 0; y < 8; ++y) {
            for (usize x = 0; x < 8; ++x, ++i) {
                result[{x, y, z}] = GOLDEN_MODEL_DATA[i];
            }
        }
    }
    return result;
}

}  // namespace

void verifyDebugModelVoxels(const VoxelArray &voxels)
{
    static const auto goldenModel = makeDebugGoldenModel();

    VXIO_ASSERT_EQ(voxels.dimensions(), Vec3size(8, 8, 8));
    VXIO_ASSERT_EQ(goldenModel, voxels);
}

void writeRandomVoxels(VoxelArray &out, bool transparency, u32 seed)
{
    const Vec3size outDims = out.dimensions();

    default_rng rng{seed};

    u32 alphaFix = not transparency * (u32{0xff} << 24);

    std::uniform_int_distribution<usize> distrX{0, outDims.x() - 1};
    std::uniform_int_distribution<usize> distrY{0, outDims.y() - 1};
    std::uniform_int_distribution<usize> distrZ{0, outDims.z() - 1};
    std::uniform_int_distribution<argb32> distrArgb;

    for (usize z = 0; z < outDims.z(); ++z) {
        for (usize y = 0; y < outDims.y(); ++y) {
            for (usize x = 0; x < outDims.x(); ++x) {
                Vec3size pos = {distrX(rng), distrY(rng), distrZ(rng)};
                out[pos] = distrArgb(rng) | alphaFix;
            }
        }
    }
}

Palette32 paletteFromVoxels(const VoxelArray &voxels)
{
    Palette32 pal;
    for (const Voxel32 voxel : voxels) {
        pal.insert(voxel.argb);
    }
    return pal;
}

}  // namespace voxelio::test
