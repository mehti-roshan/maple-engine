#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <glm/exponential.hpp>
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace maple {

template <typename Stream>
bool SerializeBits(Stream& stream, uint64_t& value, uint32_t numBits) {
  return stream.SerializeBits(value, numBits);
}

template <typename Stream>
bool SerializeBool(Stream& stream, bool& value) {
  return stream.SerializeBool(value);
}

template <typename Stream>
bool SerializeInt(Stream& stream, int64_t& value, int64_t minInclusive, int64_t maxExclusive) {
  return stream.SerializeInt(value, minInclusive, maxExclusive);
}

template <typename Stream>
bool SerializeFloat(Stream& stream, float& value) {
  return stream.SerializeFloat(value);
}

template <typename Stream>
bool SerializeFloatCompressed(Stream& stream, float& value, float minInclusive, float maxInclusive, float res) {
  return stream.SerializeFloatCompressed(value, minInclusive, maxInclusive, res);
}

template <typename Stream>
bool SerializeDouble(Stream& stream, double& value) {
  return stream.SerializeDouble(value);
}

template <typename Stream>
bool SerializeVec3f(Stream& stream, glm::vec3& value) {
  if (!SerializeFloat(stream, value.x)) return false;
  if (!SerializeFloat(stream, value.y)) return false;
  if (!SerializeFloat(stream, value.z)) return false;
  return true;
}

template <typename Stream>
bool SerializeVec3fCompressed(Stream& stream, glm::vec3& value, glm::vec3& minInclusive, glm::vec3& maxInclusive, glm::vec3& res) {
  if (!SerializeFloatCompressed(stream, value.x, minInclusive.x, maxInclusive.x, res.x)) return false;
  if (!SerializeFloatCompressed(stream, value.y, minInclusive.y, maxInclusive.y, res.y)) return false;
  if (!SerializeFloatCompressed(stream, value.z, minInclusive.z, maxInclusive.z, res.z)) return false;
  return true;
}

template <typename Stream>
bool SerializeQuat(Stream& stream, glm::quat& value) {
  if (!SerializeFloat(stream, value.x)) return false;
  if (!SerializeFloat(stream, value.y)) return false;
  if (!SerializeFloat(stream, value.z)) return false;
  if (!SerializeFloat(stream, value.w)) return false;
  return true;
}

// TODO: serialize functions for
// string
// compressed quaternion
// compressed normalized vec3 (with 1 component not sent) and a float for length

}  // namespace maple