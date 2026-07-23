#include "maple_audio.h"

#include <log_macros.h>

#include <cstdint>
#include <glm/ext/matrix_transform.hpp>
#include <limits>
#include <unordered_map>
#include <vector>

#include "SDL3/SDL_init.h"
#include "al.h"
#include "alc.h"
#include "alext.h"
#include "ring_buffer.h"

namespace maple {

static constexpr uint32_t SampleRate = 48000;

struct Audio::Impl {
  bool mInitialized = false;
  std::optional<SDL_AudioDeviceID> mRecordDevice = std::nullopt;
  RingBuffer<float> mCaptureBuffer = RingBuffer<float>(SampleRate / 10);   // 100ms @ SampleRate
  RingBuffer<float> mPlaybackBuffer = RingBuffer<float>(SampleRate / 10);  // 100ms @ SampleRate
  ALCcontext* mContext = nullptr;
  std::vector<ALuint> mSources;

  static void SDLCALL sRecordCallback(void* userdata, SDL_AudioStream* stream, int additional_amount_bytes, int total_amount) {
    int numFloats = additional_amount_bytes / sizeof(float);

    std::vector<float> samples(numFloats);

    SDL_GetAudioStreamData(stream, samples.data(), additional_amount_bytes);

    float sum = 0.0f;

    for (float sample : samples) {
      sum += sample * sample;
    }

    float rms = std::sqrt(sum / samples.size());

    // MAPLE_DEBUG("Volume: {}", rms);
  }
};

void chkal() {
  auto err = alGetError();
  if (err != AL_NO_ERROR) {
    MAPLE_WARN("OpenAL Error '{}'", err);
  }
}

Audio::Audio() = default;
Audio::Audio(Audio&&) noexcept = default;
Audio& Audio::operator=(Audio&&) noexcept = default;
Audio::~Audio() { Destroy(); }
Audio::Audio(const CreateInfo& info) : impl(std::make_unique<Impl>()) {
  if (!SDL_Init(SDL_INIT_AUDIO)) MAPLE_FATAL("SDL audio init failed: {}", SDL_GetError());
  impl->mInitialized = true;

  SDL_AudioSpec recordSpec{};
  recordSpec.format = SDL_AUDIO_F32;
  recordSpec.channels = 1;
  recordSpec.freq = SampleRate;

  impl->mRecordDevice = {};
  impl->mRecordDevice = SDL_OpenAudioDevice(SDL_AUDIO_DEVICE_DEFAULT_RECORDING, &recordSpec);
  if (!impl->mRecordDevice.value()) MAPLE_FATAL("Failed to open recording device: {}", SDL_GetError());
  SDL_AudioStream* recordStream = SDL_OpenAudioDeviceStream(impl->mRecordDevice.value(), &recordSpec, impl->sRecordCallback, this);
  SDL_ResumeAudioStreamDevice(recordStream);

  auto device = alcOpenDevice(nullptr);
  chkal();
  if (!device) MAPLE_FATAL("failed to initialize OpenAL Device");
  chkal();

  impl->mContext = alcCreateContext(device, nullptr);
  chkal();
  alcMakeContextCurrent(impl->mContext);
  chkal();
  if (!impl->mContext) MAPLE_FATAL("failed to make OpenAL Context current");

  UpdateListener(glm::vec3(0), glm::vec3(0), glm::identity<glm::quat>());
}

void Audio::Destroy() {
  if (!impl) return;

  alDeleteSources(impl->mSources.size(), impl->mSources.data());
  auto device = alcGetContextsDevice(impl->mContext);
  alcMakeContextCurrent(NULL);
  alcDestroyContext(impl->mContext);
  alcCloseDevice(device);
  if (impl->mRecordDevice.has_value()) SDL_CloseAudioDevice(impl->mRecordDevice.value());
  if (impl->mInitialized) SDL_Quit();
}

void Audio::UpdateListener(const glm::vec3& pos, const glm::vec3& velocity, const glm::quat& orientation) {
  glm::vec3 forward(0, 0, -1);
  glm::vec3 up(0, 1, 0);
  ALfloat listenerOri[] = {forward.x, forward.y, forward.z, up.x, up.y, up.z};

  alListener3f(AL_POSITION, pos.x, pos.y, pos.z);
  chkal();
  alListener3f(AL_VELOCITY, velocity.x, velocity.y, velocity.z);
  chkal();
  alListenerfv(AL_ORIENTATION, listenerOri);
  chkal();
}

ALenum toALFormat(bool isStereo, uint8_t bitsPerSample) {
  switch (bitsPerSample) {
    case 32:
      return isStereo ? AL_FORMAT_STEREO_FLOAT32 : AL_FORMAT_MONO_FLOAT32;
    case 16:
      return isStereo ? AL_FORMAT_STEREO16 : AL_FORMAT_MONO16;
    case 8:
      return isStereo ? AL_FORMAT_STEREO8 : AL_FORMAT_MONO8;
    default:
      MAPLE_FATAL("failed to get AL Format from provided data");
  }
}

Audio::ClipHndl Audio::CreateClip(const ClipCreateInfo& info) {
  ALuint buffer;
  alGenBuffers((ALuint)1, &buffer);
  chkal();
  auto format = toALFormat(info.isStereo, info.bitsPerSample);
  alBufferData(buffer, format, info.data.get(), info.bitsPerSample / 8 * info.sampleCount * (info.isStereo ? 2 : 1), info.sampleRate);
  chkal();

  return buffer;
}

void Audio::DestroyClip(ClipHndl clip) { alDeleteBuffers(1, &clip); }

Audio::SourceHndl Audio::PlayClip(ClipHndl clip, const SourceInfo& info) {
  uint32_t source = std::numeric_limits<uint32_t>::max();

  for (auto v : impl->mSources) {
    if (!IsPlaying(v)) {
      source = v;
      break;
    }
  }

  if (source == std::numeric_limits<uint32_t>::max()) {
    alGenSources((ALuint)1, &source);
    chkal();
    impl->mSources.push_back(source);
  }

  alSourcei(source, AL_BUFFER, clip);
  chkal();

  UpdateSource(source, info);

  alSourcePlay(source);
  chkal();

  return source;
}

void Audio::UpdateSource(SourceHndl source, const SourceInfo& info) {
  if (info.posRelativeToListener) {
    alSourcei(source, AL_SOURCE_RELATIVE, AL_TRUE);
    chkal();
  }
  alSource3f(source, AL_POSITION, info.position.x, info.position.y, info.position.z);
  chkal();
  alSource3f(source, AL_VELOCITY, info.velocity.x, info.velocity.y, info.velocity.z);
  chkal();
  alSourcef(source, AL_PITCH, info.pitch);
  chkal();
  alSourcef(source, AL_GAIN, info.gain);
  chkal();
  alSourcei(source, AL_LOOPING, info.loop);
  chkal();
}

bool Audio::IsPlaying(SourceHndl source) {
  ALint v;
  alGetSourcei(source, AL_SOURCE_STATE, &v);
  return v == AL_PLAYING;
}

void Audio::StopSource(SourceHndl source) {
  alSourceStop(source);
  chkal();
}

}  // namespace maple