#include "noise.h"

#include "third_party/FastNoiseLite.h"

namespace maple {

Noise::Noise(uint64_t seed, Type type) {
  mNoise = std::make_unique<FastNoiseLite>();

  SetType(type);
  SetSeed(seed);
}

Noise::~Noise() {};

float Noise::GetNoisef(float x, float y) const { return mNoise->GetNoise(x, y); }
float Noise::GetNoisef(float x, float y, float z) const { return mNoise->GetNoise(x, y, z); }
float Noise::GetNoised(double x, double y) const { return mNoise->GetNoise(x, y); }
float Noise::GetNoised(double x, double y, double z) const { return mNoise->GetNoise(x, y, z); }

Noise& Noise::SetSeed(uint64_t seed) {
  mNoise->SetSeed(static_cast<int>(seed));
  return *this;
}

Noise& Noise::SetType(Type type) {
  mNoise->SetNoiseType(static_cast<FastNoiseLite::NoiseType>(type));
  return *this;
}

Noise& Noise::SetFrequency(float frequency) {
  mNoise->SetFrequency(frequency);
  return *this;
}

Noise& Noise::SetFractalType(FractalType type) {
  mNoise->SetFractalType(static_cast<FastNoiseLite::FractalType>(type));
  return *this;
}

Noise& Noise::SetFractalOctaves(int octaves) {
  mNoise->SetFractalOctaves(octaves);
  return *this;
}

Noise& Noise::SetFractalLacunarity(float lacunarity) {
  mNoise->SetFractalLacunarity(lacunarity);
  return *this;
}

Noise& Noise::SetFractalGain(float gain) {
  mNoise->SetFractalGain(gain);
  return *this;
}

Noise& Noise::SetFractalWeightedStrength(float strength) {
  mNoise->SetFractalWeightedStrength(strength);
  return *this;
}

Noise& Noise::SetFractalPingPongStrength(float strength) {
  mNoise->SetFractalPingPongStrength(strength);
  return *this;
}

Noise& Noise::SetDomainWarpType(DomainWarpType type) {
  mNoise->SetDomainWarpType(static_cast<FastNoiseLite::DomainWarpType>(type));
  return *this;
}

Noise& Noise::SetDomainWarpAmplitude(float amplitude) {
  mNoise->SetDomainWarpAmp(amplitude);
  return *this;
}

void Noise::DomainWarp(float& x, float& y) const { mNoise->DomainWarp(x, y); }

void Noise::DomainWarp(float& x, float& y, float& z) const { mNoise->DomainWarp(x, y, z); }
};  // namespace maple