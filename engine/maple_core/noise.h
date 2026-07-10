#pragma once

#include <cstdint>
#include <memory>

class FastNoiseLite;

namespace maple {
class Noise {
 public:
  enum Type {
    OpenSimplex2,
    OpenSimplex2S,
    Cellular,
    Perlin,
    ValueCubic,
    Value,
  };

  enum FractalType { None, FBm, Ridged, PingPong, DomainWarpProgressive, DomainWarpIndependent };

  enum DomainWarpType {
    OpenSimplex2Warp,
    OpenSimplex2ReducedWarp,
    BasicGridWarp,
  };

  Noise() = default;
  explicit Noise(uint64_t seed, Type type);
  ~Noise();

  // Sampling
  float GetNoisef(float x, float y) const;
  float GetNoisef(float x, float y, float z) const;
  float GetNoised(double x, double y) const;
  float GetNoised(double x, double y, double z) const;

  // Domain warping
  void DomainWarp(float& x, float& y) const;
  void DomainWarp(float& x, float& y, float& z) const;

  // Settings
  Noise& SetSeed(uint64_t seed);
  Noise& SetType(Type type);

  Noise& SetFrequency(float frequency);

  Noise& SetFractalType(FractalType type);
  Noise& SetFractalOctaves(int octaves);
  Noise& SetFractalLacunarity(float lacunarity);
  Noise& SetFractalGain(float gain);
  Noise& SetFractalWeightedStrength(float strength);
  Noise& SetFractalPingPongStrength(float strength);

  Noise& SetDomainWarpType(DomainWarpType type);
  Noise& SetDomainWarpAmplitude(float amplitude);

 private:
  std::unique_ptr<FastNoiseLite> mNoise;
};
}  // namespace maple