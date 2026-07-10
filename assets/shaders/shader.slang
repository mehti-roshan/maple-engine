struct UniformBuffer {
  float4x4 view;
  float4x4 proj;
  float time;
};

struct PushConstants {
  uint64_t vertexBufferAddress;
  uint32_t indexBufferOffset;
  uint32_t materialBufferOffset;
  uint32_t instanceBufferIndex;
};

[[vk::push_constant]] PushConstants push;

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBuffer> ubo;

[[vk::binding(1, 0)]]
StructuredBuffer<float4x4> instanceModels;

[[vk::binding(2, 0)]]
StructuredBuffer<uint32_t> materialBuffer;

struct VSOutput {
  float4 pos : SV_Position;
  float2 uv;
};

struct Vertex {
  float3 pos;
  float2 uv;
};

[shader("vertex")]
VSOutput vertMain(uint vertexIndex : SV_VertexID, uint instanceID : SV_InstanceID) {
  Vertex* vertBuffer = reinterpret<Vertex*>(push.vertexBufferAddress);
  uint* indexBuffer = reinterpret<uint*>(push.vertexBufferAddress + push.indexBufferOffset);

  uint index = indexBuffer[vertexIndex];
  Vertex vert = vertBuffer[index];

  float4x4 model = instanceModels[push.instanceBufferIndex + instanceID];

  VSOutput output;

  output.pos = mul(ubo.proj, mul(ubo.view, mul(model, float4(vert.pos, 1.0))));
  output.uv = vert.uv;
  return output;
}

[[vk::binding(3, 0)]]
Sampler2D textures[];

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
  float dist = distance(vertIn.uv, float2(0.5, 0.5));
  // if (dist > 0.5) discard;
  dist *= 2.0;
  dist = 1.0 - dist;
  float4 tint = float4(sin(ubo.time) * 0.5 + 0.5, cos(ubo.time) * 0.5 + 0.5, 0.0, 1.0);
  uint32_t textureIdx = materialBuffer[push.materialBufferOffset];
  return textures[textureIdx].Sample(vertIn.uv) * dist;
}
