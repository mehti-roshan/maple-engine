struct VSInput {
  float3 inPosition;
  float2 inTexCoord;
};

struct UniformBuffer {
  // float4x4 model;
  float4x4 view;
  float4x4 proj;
};

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBuffer> ubo;

[[vk::binding(2, 0)]]
StructuredBuffer<float4x4> instanceModels;

struct VSOutput {
  float4 pos : SV_Position;
  float2 texCoord;
};

[shader("vertex")]
VSOutput vertMain(VSInput input, uint instanceID: SV_InstanceID) {
  VSOutput output;
  float4x4 model = instanceModels[instanceID];
  output.pos = mul(ubo.proj, mul(ubo.view, mul(model, float4(input.inPosition, 1.0))));
  output.texCoord = input.inTexCoord;
  return output;
}

[[vk::binding(1, 0)]]
Sampler2D texture;

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
  return texture.Sample(vertIn.texCoord);
}
