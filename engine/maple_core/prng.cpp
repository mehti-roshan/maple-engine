#include "prng.h"

#include <cassert>
#include <engine/third_party/pcg/pcg_random.hpp>
#include <memory>
#include <random>

namespace maple {
struct PRNG::Impl {
  pcg64 pcg;
};

PRNG::~PRNG() {}
PRNG::PRNG(uint64_t seed) { mImpl = std::make_unique<PRNG::Impl>(pcg64(seed)); }

float PRNG::NextFloat(float min, float max) const {
  std::uniform_real_distribution<float> dist(min, max);
  return dist(mImpl->pcg);
}

double PRNG::NextDouble(double min, double max) const {
  std::uniform_real_distribution<double> dist(min, max);
  return dist(mImpl->pcg);
}

int64_t PRNG::NextInt64(int64_t min, int64_t max) const {
  std::uniform_int_distribution<int64_t> dist(min, max);
  return dist(mImpl->pcg);
}

uint64_t PRNG::NextUInt64(uint64_t min, uint64_t max) const {
  std::uniform_int_distribution<uint64_t> dist(min, max);
  return dist(mImpl->pcg);
}

bool PRNG::NextChance(double probability) const {
  assert(probability > 0.0 && probability <= 1.0);
  std::bernoulli_distribution dist(probability);
  return dist(mImpl->pcg);
}

}  // namespace maple