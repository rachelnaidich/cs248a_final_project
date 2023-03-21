//
//  SceneKitShaders.metal
//  cs248a_final_project
//
//  Created by Rachel Naidich on 3/21/23.
//

#include <metal_stdlib>
using namespace metal;

#include <SceneKit/scn_metal>

struct VertexInput {
    float3 position  [[attribute(SCNVertexSemanticPosition)]];
    float3 normal [[attribute(SCNVertexSemanticNormal)]];
    float2 texCoords [[attribute(SCNVertexSemanticTexcoord0)]];
};

struct NodeBuffer {
    float4x4 modelViewProjectionTransform;
    float4x4 modelViewTransform;
};

float2 getBackgroundCoordinate(
                      constant float4x4& displayTransform,
                      constant float4x4& modelViewTransform,
                      constant float4x4& projectionTransform,
                      float4 position) {
    // Transform the vertex to the camera coordinate system.
    float4 vertexCamera = modelViewTransform * position;
    
    // Camera projection and perspective divide to get normalized viewport coordinates (clip space).
    float4 vertexClipSpace = projectionTransform * vertexCamera;
    vertexClipSpace /= vertexClipSpace.w;
    
    // XY in clip space is [-1,1]x[-1,1], so adjust to UV texture coordinates: [0,1]x[0,1].
    // Image coordinates are Y-flipped (upper-left origin).
    float4 vertexImageSpace = float4(vertexClipSpace.xy * 0.5 + 0.5, 0.0, 1.0);
    vertexImageSpace.y = 1.0 - vertexImageSpace.y;
    
    // Apply ARKit's display transform (device orientation * front-facing camera flip).
    return (displayTransform * vertexImageSpace).xy;
}

struct GeometryEffectInOut {
    float4 position [[ position ]];
    float2 backgroundTextureCoords;
    float2 overlayTextureCoords;
    float time;
    float len;
};

float rand(int x, int y, int z)
{
    int seed = x + y * 57 + z * 241;
    seed= (seed<< 13) ^ seed;
    return (( 1.0 - ( (seed * (seed * seed * 15731 + 789221) + 1376312589) & 2147483647) / 1073741824.0f) + 1.0f) / 2.0f;
}

vertex GeometryEffectInOut geometryEffectVertexShader(VertexInput in [[ stage_in ]],
                                  constant SCNSceneBuffer& scn_frame [[buffer(0)]],
                                  constant NodeBuffer& scn_node [[ buffer(1) ]],
                                  constant float4x4& u_displayTransform [[buffer(2)]],
                                  constant float& u_time [[buffer(3)]])
{
    GeometryEffectInOut out;

    out.backgroundTextureCoords = getBackgroundCoordinate(
                                   u_displayTransform,
                                   scn_node.modelViewTransform,
                                   scn_frame.projectionTransform,
                                   float4(in.position, 1.0));

    float waveHeight = 0.2;
    float waveFrequency = 20.0;

    float len = length(in.position.xy);

    float blending = max(0.0, 0.5 - len);
    in.position.z += sin(len * waveFrequency + u_time * 5) * waveHeight * blending;
    //in.position.z -= blending;
    
    out.position = scn_node.modelViewProjectionTransform * float4(in.position, 1.0);
    out.overlayTextureCoords = in.texCoords;
    out.time = u_time;
    out.len = len;
      
    return out;
}

float4 waterColor(float time, float2 sp) {
    float2 p = sp * 15.0 - float2(20.0);
    float2 i = p;
    float c = 0.0;
    float inten = 0.025;
    float speed = 1.5;
    float speed2 = 3.0;
    float freq = 0.8;
    float xflow = 1.5;
    float yflow = 0.0;

    for (int n = 0; n < 8; n++) {
        float t = time * (1.0 - (3.0 / (float(n) + speed)));
        i = p + float2(cos(t - i.x * freq) + sin(t + i.y * freq) + (time * xflow), sin(t - i.y * freq) + cos(t + i.x * freq) + (time * yflow));
        c += 1.0 / length(float2(p.x / (sin(i.x + t * speed2) / inten), p.y / (cos(i.y + t * speed2) / inten)));
    }
    
    c /= float(8);
    c = 1.5 - sqrt(c);
    return float4(float3(c * c * c * c), 0.0) + float4(0.0, 0.4, 0.55, 1);
}

float3 hsv2rgb(  float3 c )
{
    float3 rgb = clamp( abs(fmod(c.x*6.0+float3(0.0,4.0,2.0),6.0)-3.0)-1.0, 0.0, 1.0 );
    rgb = rgb*rgb*(3.0-2.0*rgb);
    return c.z * mix( float3(1.0), rgb, c.y);
}

fragment float4 waterFragmentShader(GeometryEffectInOut in [[ stage_in] ],
                                 texture2d<float, access::sample> diffuseTexture [[texture(0)]])
{
    constexpr sampler textureSampler( mip_filter::linear, mag_filter::linear,  min_filter::linear );

    float4 backgroundColor = diffuseTexture.sample(textureSampler, in.backgroundTextureCoords);
    float4 water = waterColor(in.time, in.overlayTextureCoords);

    float3 rainbow = hsv2rgb(float3(in.time * 0.3 + in.overlayTextureCoords.x - in.overlayTextureCoords.y,0.5,1.0));
    float4 rainbowColor = float4(rainbow, 1.0);
    //return backgroundColor;
    return mix(backgroundColor, water, 2 * max(0.0, 0.5 - in.len));
//    return mix(backgroundColor, rainbowColor, 2 * max(0.0, 0.5 - in.len));
   // return mix(backgroundColor, rainbowColor, pow(max(0.0, 0.5 - in.len), 2));
    
}

fragment float4 rainbowFragmentShader(GeometryEffectInOut in [[ stage_in] ],
                                 texture2d<float, access::sample> diffuseTexture [[texture(0)]])
{
    constexpr sampler textureSampler( mip_filter::linear, mag_filter::linear,  min_filter::linear );

    float4 backgroundColor = diffuseTexture.sample(textureSampler, in.backgroundTextureCoords);
    float4 water = waterColor(in.time, in.overlayTextureCoords);

    float3 rainbow = hsv2rgb(float3(in.time * 0.3 + in.overlayTextureCoords.x - in.overlayTextureCoords.y,0.5,1.0));
    float4 rainbowColor = float4(rainbow, 1.0);
    //return backgroundColor;
    //return mix(backgroundColor, water, 2 * max(0.0, 0.5 - in.len));
//    return mix(backgroundColor, rainbowColor, 2 * max(0.0, 0.5 - in.len));
    return mix(backgroundColor, rainbowColor, pow(max(0.0, 0.5 - in.len), 2));
    
}
