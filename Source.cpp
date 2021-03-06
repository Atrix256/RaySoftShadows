#include <atomic>
#include <thread>
#include <vector>
#include <array>
#include <stdint.h>
#include <stdlib.h>
#include <random>
#include <algorithm>

#include "VectorMath.h"
#include "SImageData.h"
#include "RayIntersect.h"

// stb_image is an amazing header only image library (aka no linking, just include the headers!).  http://nothings.org/stb
#pragma warning( disable : 4996 ) 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#pragma warning( default : 4996 ) 



#define IMAGE_WIDTH() 400
#define IMAGE_HEIGHT() 400

#define STRATIFIED_SAMPLE_COUNT_ONE_AXIS() 2  // it does this many samples squared per pixel for AA

#define STRATIFIED_SAMPLE_COUNT_ONE_AXIS_PATHTRACING() 256 // it does this many samples squared per pixel while path tracing
#define PATHTRACING_RAY_BOUNCE() 4 // set to 1 to make the pathtracing be like the raytracing

#define DO_PATHTRACE() 0 // takes about 10 minutes on my machine

#define FORCE_SINGLE_THREADED() 0 // useful for debugging



typedef uint8_t uint8;

static const float c_rayEpsilon = 0.01f; // value used to push the ray a little bit away from surfaces before doing shadow rays
static const float c_pi = 3.14159265359f;
static const float c_goldenRatioConjugate = 0.61803398875f;

#define STRATIFIED_SAMPLE_COUNT() (STRATIFIED_SAMPLE_COUNT_ONE_AXIS()*STRATIFIED_SAMPLE_COUNT_ONE_AXIS())
#define STRATIFIED_SAMPLE_COUNT_PATHTRACING() (STRATIFIED_SAMPLE_COUNT_ONE_AXIS_PATHTRACING()*STRATIFIED_SAMPLE_COUNT_ONE_AXIS_PATHTRACING())

enum class EProcess
{
    No,
    Yes,
    Exhaustive
};

//-------------------------------------------------------------------------------------------------------------------
float3 RandomVectorTowardsLight (float3 lightDir, float radius, float rngX, float rngY)
{
    // make basis vectors for the light quad
    lightDir = Normalize(lightDir);
    float3 scaledUAxis = Normalize(Cross(float3{ 0.0f, 1.0f, 0.0f }, lightDir)) * radius;
    float3 scaledVAxis = Normalize(Cross(lightDir, scaledUAxis)) * radius;

    float r1 = 2.0f * c_pi *rngX;
    float r2 = rngY;
    float r2s = sqrt(r2);

    return Normalize(lightDir + scaledUAxis * cos(r1)*r2s + scaledVAxis * sin(r1)*r2s);
}

//-------------------------------------------------------------------------------------------------------------------
inline float3 CosineSampleHemisphere (const float3& normal, float rngX, float rngY)
{
    // from smallpt: http://www.kevinbeason.com/smallpt/

    float r1 = 2.0f * c_pi * rngX;
    float r2 = rngY;
    float r2s = sqrt(r2);

    float3 w = normal;
    float3 u;
    if (fabs(w[0]) > 0.1f)
        u = Cross({ 0.0f, 1.0f, 0.0f }, w);
    else
        u = Cross({ 1.0f, 0.0f, 0.0f }, w);

    u = Normalize(u);
    float3 v = Cross(w, u);
    float3 d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1 - r2));
    d = Normalize(d);

    return d;
}

//=================================================================================
inline float3 UniformSampleHemisphere (const float3& N, float rngX, float rngY)
{
    // Uniform point on sphere
    // from http://mathworld.wolfram.com/SpherePointPicking.html
    float u = rngX;
    float v = rngY;

    float theta = 2.0f * c_pi * u;
    float phi = acos(2.0f * v - 1.0f);

    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    float cosPhi = cos(phi);
    float sinPhi = sin(phi);

    float3 dir;
    dir[0] = cosTheta * sinPhi;
    dir[1] = sinTheta * sinPhi;
    dir[2] = cosPhi;

    // if our vector is facing the wrong way vs the normal, flip it!
    if (Dot(dir, N) <= 0.0f)
        dir = dir * -1.0f;

    return dir;
}

//-------------------------------------------------------------------------------------------------------------------
static SQuad g_quads[] =
{
    { { -15.0f, 0.0f, 15.0f },{ 15.0f, 0.0f, 15.0f },{ 15.0f, 0.0f, -15.0f },{ -15.0f, 0.0f, -15.0f }, {1.0f, 1.0f, 1.0f}},
};

static const SSphere g_spheres[] =
{
    { { -2.0f, 1.0f, 4.0f }, 1.0f,{ 1.0f, 0.1f, 0.1f } },
    { {  0.0f, 1.0f, 4.0f }, 0.6f,{ 0.1f, 1.0f, 0.1f } },
    { {  2.0f, 1.0f, 4.0f }, 0.3f,{ 0.1f, 0.1f, 1.0f } },
};

static const SPositionalLight g_positionalLights[] =
{
    { {-1.0f, 5.0f, 6.0f}, 0.5f,{ 10.0f, 10.0f, 10.0f }},

    { { 1.0f, 3.0f, 6.0f}, 0.75f,{ 10.0f, 10.0f, 10.0f }},
};

static const float  g_cameraDistance = 2.0f;
static const float3 g_cameraPos = { 0.0f, 2.0f, -6.0f };
static const float3 g_cameraX = { 1.0f, 0.0f, 0.0f };
static const float3 g_cameraY = { 0.0f, 1.0f, 0.0f };
static const float3 g_cameraZ = { 0.0f, 0.0f, 1.0f };

static const float3 g_skyColor = float3{ 135.0f / 255.0f, 206.0f / 255.0f, 235.0f / 255.0f } / 32.0f;

SImageData<4> g_blueNoiseTexture;

static const size_t c_numLights = sizeof(g_positionalLights) / sizeof(g_positionalLights[0]);

//-------------------------------------------------------------------------------------------------------------------
template <typename L>
void RunMultiThreaded (const char* label, const L& lambda)
{
    std::atomic<size_t> atomicCounter(0);
    auto wrapper = [&] ()
    {
        lambda(atomicCounter);
    };

    size_t numThreads = FORCE_SINGLE_THREADED() ? 1 : std::thread::hardware_concurrency();
    printf("%s (%zu threads)\n", label, numThreads);
    if (numThreads > 1)
    {
        std::vector<std::thread> threads;
        threads.resize(numThreads);
        size_t faceIndex = 0;
        for (std::thread& t : threads)
            t = std::thread(wrapper);
        for (std::thread& t : threads)
            t.join();
    }
    else
    {
        wrapper();
    }
}

//-------------------------------------------------------------------------------------------------------------------

struct SGbufferPixel
{
    SHitInfo hitInfo;
    float u, v;
    float shadowMultipliersPositional[c_numLights];
    bool isLight = false;
    float3 worldPos;
};

//-------------------------------------------------------------------------------------------------------------------

template <int RADIUS, size_t CHANNELS>
void Convolve (const SImageData<CHANNELS>& src, SImageData<CHANNELS>& dest, std::array<float, (1 + RADIUS * 2) * (1 + RADIUS * 2)>& kernel)
{
    static const size_t numPixelsOneSide = (1 + RADIUS * 2);

    float totalKernelWeight = 0.0f;
    for (float f : kernel)
        totalKernelWeight += f;

    dest = src;

    float* outPixel = &dest.m_pixels[0];

    for (int y = 0; y < dest.m_height; ++y)
    {
        for (int x = 0; x < dest.m_width; ++x)
        {
            for (size_t channel = 0; channel < CHANNELS; ++channel)
            {
                *outPixel = 0.0f;

                for (int offsetY = -RADIUS; offsetY <= RADIUS; ++offsetY)
                {
                    int sampleY = y + offsetY;
                    sampleY = clamp(sampleY, 0, (int)dest.m_height - 1);

                    for (int offsetX = -RADIUS; offsetX <= RADIUS; ++offsetX)
                    {
                        int sampleX = x + offsetX;
                        sampleX = clamp(sampleX, 0, (int)dest.m_width - 1);
                        
                        *outPixel += src.GetPixel(sampleX, sampleY)[channel] * kernel[(offsetY + RADIUS) * numPixelsOneSide + (offsetX + RADIUS)];
                    }
                }

                *outPixel /= totalKernelWeight;

                ++outPixel;
            }
        }
    }
}

//-------------------------------------------------------------------------------------------------------------------

template <int RADIUS>
std::array<float, (1 + RADIUS * 2)*(1 + RADIUS * 2)> MakeBoxBlurKernel()
{
    std::array<float, (1 + RADIUS * 2)*(1 + RADIUS * 2)> ret;

    for (float& f : ret)
        f = 1.0f;

    return ret;
}

//-------------------------------------------------------------------------------------------------------------------

template <int RADIUS>
std::array<float, (1 + RADIUS * 2)*(1 + RADIUS * 2)> MakeTriangleBlurKernel()
{
    std::array<float, (1 + RADIUS * 2)*(1 + RADIUS * 2)> ret;

    static const size_t numPixelsOneSide = (1 + RADIUS * 2);
    static const size_t numPixelsInFilter = numPixelsOneSide * numPixelsOneSide;

    for (size_t i = 0; i < numPixelsInFilter; ++i)
    {
        size_t x = i % numPixelsOneSide;
        size_t y = i / numPixelsOneSide;

        float triangleX = (x <= RADIUS) ? float(x + 1) : float(numPixelsOneSide - x);
        float triangleY = (y <= RADIUS) ? float(y + 1) : float(numPixelsOneSide - y);

        ret[i] = triangleX * triangleY;
    }

    return ret;
}

//-------------------------------------------------------------------------------------------------------------------

template <int RADIUS, size_t CHANNELS>
void BoxBlur(const SImageData<CHANNELS>& src, SImageData<CHANNELS>& dest)
{
    auto kernel = MakeBoxBlurKernel<RADIUS>();
    Convolve<RADIUS, CHANNELS>(src, dest, kernel);
}

//-------------------------------------------------------------------------------------------------------------------

template <int RADIUS, size_t CHANNELS>
void TriangleBlur(const SImageData<CHANNELS>& src, SImageData<CHANNELS>& dest)
{
    auto kernel = MakeTriangleBlurKernel<RADIUS>();
    Convolve<RADIUS, CHANNELS>(src, dest, kernel);
}

//-------------------------------------------------------------------------------------------------------------------

template <int RADIUS, size_t CHANNELS>
void MedianFilter(const SImageData<CHANNELS>& src, SImageData<CHANNELS>& dest)
{
    const size_t numPixelsOneSide = (1 + RADIUS * 2);
    const size_t numPixelsInFilter = numPixelsOneSide * numPixelsOneSide;

    std::array<float, numPixelsInFilter> samples;

    dest = src;

    float* outPixel = &dest.m_pixels[0];
    for (int y = 0; y < dest.m_height; ++y)
    {
        for (int x = 0; x < dest.m_width; ++x)
        {
            for (size_t channel = 0; channel < CHANNELS; ++channel)
            {
                for (int offsetY = -RADIUS; offsetY <= RADIUS; ++offsetY)
                {
                    int sampleY = y + offsetY;
                    sampleY = clamp(sampleY, 0, (int)dest.m_height - 1);

                    for (int offsetX = -RADIUS; offsetX <= RADIUS; ++offsetX)
                    {
                        int sampleX = x + offsetX;
                        sampleX = clamp(sampleX, 0, (int)dest.m_width - 1);

                        samples[(offsetX + RADIUS)*numPixelsOneSide + (offsetY+RADIUS)] = src.GetPixel(sampleX, sampleY)[channel];
                    }
                }

                std::sort(samples.begin(), samples.end());

                *outPixel = samples[numPixelsInFilter / 2];
                ++outPixel;
            }
        }
    }
}

//-------------------------------------------------------------------------------------------------------------------

template <int RADIUS, size_t SAMPLES_PER_PIXEL>
void Convolve (const std::vector<SGbufferPixel>& src, std::vector<SGbufferPixel>& dest, size_t width, size_t height, std::array<float, (1 + RADIUS * 2) * (1 + RADIUS * 2)>& kernel)
{
    static const size_t numPixelsOneSide = (1 + RADIUS * 2);

    float totalKernelWeight = 0.0f;
    for (float f : kernel)
        totalKernelWeight += f;

    dest = src;

    SGbufferPixel* outPixel = &dest[0];

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            for (size_t sampleIndex = 0; sampleIndex < SAMPLES_PER_PIXEL; ++sampleIndex)
            {
                for (float& f : outPixel->shadowMultipliersPositional)
                    f = 0.0f;

                for (int offsetY = -RADIUS; offsetY <= RADIUS; ++offsetY)
                {
                    int sampleY = y + offsetY;
                    sampleY = clamp(sampleY, 0, (int)height - 1);

                    for (int offsetX = -RADIUS; offsetX <= RADIUS; ++offsetX)
                    {
                        int sampleX = x + offsetX;
                        sampleX = clamp(sampleX, 0, (int)width - 1);

                        const SGbufferPixel& inPixel = src[(sampleY * width + sampleX) * SAMPLES_PER_PIXEL + sampleIndex];

                        for (size_t index = 0; index < c_numLights; ++index)
                            outPixel->shadowMultipliersPositional[index] += inPixel.shadowMultipliersPositional[index] * kernel[(offsetY + RADIUS) * numPixelsOneSide + (offsetX + RADIUS)];
                    }
                }

                for (float& f : outPixel->shadowMultipliersPositional)
                    f /= totalKernelWeight;

                ++outPixel;
            }
        }
    }
}

//-------------------------------------------------------------------------------------------------------------------

template <int RADIUS, size_t SAMPLES_PER_PIXEL>
void BoxBlurShadowMultipliers (const std::vector<SGbufferPixel>& src, std::vector<SGbufferPixel>& dest, size_t width, size_t height)
{
    auto kernel = MakeBoxBlurKernel<RADIUS>();
    Convolve<RADIUS, SAMPLES_PER_PIXEL>(src, dest, width, height, kernel);
}

//-------------------------------------------------------------------------------------------------------------------

template <int RADIUS, size_t SAMPLES_PER_PIXEL>
void TriangleBlurShadowMultipliers (const std::vector<SGbufferPixel>& src, std::vector<SGbufferPixel>& dest, size_t width, size_t height)
{
    auto kernel = MakeTriangleBlurKernel<RADIUS>();
    Convolve<RADIUS, SAMPLES_PER_PIXEL>(src, dest, width, height, kernel);
}

//-------------------------------------------------------------------------------------------------------------------

enum class RayPattern
{
    None,
    Grid,
    Stratified
};

enum class RNGSource
{
    WhiteNoise,
    BlueNoiseGR
};

//-------------------------------------------------------------------------------------------------------------------

template <size_t SHADOW_RAY_COUNT, size_t SHADOW_RAY_COUNT_GRID_SIZE, RayPattern RAY_PATTERN, RNGSource RNG_SOURCE, bool STOCHASTIC_LIGHT_SOURCE>
SGbufferPixel PixelFunctionGBuffer(float u, float v, size_t pixelX, size_t pixelY, size_t SSSampleIndex, std::mt19937& rng)
{
    // raytrace to find primary ray intersection
    float3 rayPos = g_cameraPos;
    float3 rayDir = Normalize(g_cameraX * u + g_cameraY * v + g_cameraZ * g_cameraDistance);
    SGbufferPixel ret;
    for (size_t i = 0; i < sizeof(g_spheres) / sizeof(g_spheres[0]); ++i)
        RayIntersect(rayPos, rayDir, g_spheres[i], ret.hitInfo);
    for (size_t i = 0; i < sizeof(g_quads) / sizeof(g_quads[0]); ++i)
        RayIntersect(rayPos, rayDir, g_quads[i], ret.hitInfo);

    // see if a positional light was hit, and if so, don't do any shadow rays
    for (size_t lightIndex = 0; lightIndex < c_numLights; ++lightIndex)
    {
        SSphere lightSphere;
        lightSphere.position = g_positionalLights[lightIndex].position;
        lightSphere.radius = g_positionalLights[lightIndex].radius;

        if (RayIntersect(rayPos, rayDir, lightSphere, ret.hitInfo))
        {
            ret.isLight = true;
            ret.hitInfo.emissive = g_positionalLights[lightIndex].color;
        }
    }
    if (ret.isLight)
        return ret;

    if (ret.hitInfo.collisionTime == -1.0f)
        return ret;

    ret.worldPos = rayPos + rayDir * ret.hitInfo.collisionTime;

    size_t selectedLightIndex = c_numLights;
    if (STOCHASTIC_LIGHT_SOURCE)
    {
        float totalSize = 0.0f;

        size_t lightIndex = 0;
        for (lightIndex = 0; lightIndex < c_numLights; ++lightIndex)
        {
            float3 lightDir = Normalize(g_positionalLights[lightIndex].position - ret.worldPos);
            float lightDistance = Length(g_positionalLights[lightIndex].position - ret.worldPos);
            float lightDiscRadius = g_positionalLights[lightIndex].radius / lightDistance;
            totalSize += lightDiscRadius;
        }

        float rngN;
        switch (RNG_SOURCE)
        {
            case RNGSource::WhiteNoise:
            {
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                rngN = dist(rng);
                break;
            }
            case RNGSource::BlueNoiseGR:
            {
                size_t texX = pixelX % g_blueNoiseTexture.m_width;
                size_t texY = pixelY % g_blueNoiseTexture.m_height;

                float* blueNoise = g_blueNoiseTexture.GetPixel(texX, texY);
                rngN = std::fmodf(blueNoise[2] + c_goldenRatioConjugate * float(SSSampleIndex), 1.0f);
                break;
            }
            static_assert(RNG_SOURCE >= RNGSource::WhiteNoise && RNG_SOURCE <= RNGSource::BlueNoiseGR, "RNG_SOURCE invalid");
        }

        float chosenSize = totalSize * rngN;
        for (lightIndex = 0; lightIndex < c_numLights; ++lightIndex)
        {
            float3 lightDir = Normalize(g_positionalLights[lightIndex].position - ret.worldPos);
            float lightDistance = Length(g_positionalLights[lightIndex].position - ret.worldPos);
            float lightDiscRadius = g_positionalLights[lightIndex].radius / lightDistance;

            if (lightDiscRadius > chosenSize)
                break;

            chosenSize -= lightDiscRadius;
        }
        selectedLightIndex = clamp(lightIndex, (size_t)0, c_numLights-1);
    }

    // shadow rays
    float3 pixelPos = rayPos + rayDir * ret.hitInfo.collisionTime;
    float3 shadowPos = pixelPos + ret.hitInfo.normal * c_rayEpsilon;
    for (size_t lightIndex = 0; lightIndex < c_numLights; ++lightIndex)
    {
        ret.shadowMultipliersPositional[lightIndex] = 0.0f;

        if (selectedLightIndex != c_numLights && lightIndex != selectedLightIndex)
            continue;

        float3 lightDir = Normalize(g_positionalLights[lightIndex].position - ret.worldPos);
        float lightDistance = Length(g_positionalLights[lightIndex].position - ret.worldPos);
        float lightDiscRadius = g_positionalLights[lightIndex].radius / lightDistance;

        for (size_t sampleIndex = 0; sampleIndex < SHADOW_RAY_COUNT; ++sampleIndex)
        {
            float rngX, rngY;

            switch (RNG_SOURCE)
            {
                case RNGSource::WhiteNoise:
                {
                    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                    rngX = dist(rng);
                    rngY = dist(rng);
                    break;
                }
                case RNGSource::BlueNoiseGR:
                {
                    size_t texX = pixelX % g_blueNoiseTexture.m_width;
                    size_t texY = pixelY % g_blueNoiseTexture.m_height;

                    float* blueNoise = g_blueNoiseTexture.GetPixel(texX, texY);
                    rngX = std::fmodf(blueNoise[0] + c_goldenRatioConjugate * float(SSSampleIndex+sampleIndex), 1.0f);
                    rngY = std::fmodf(blueNoise[1] + c_goldenRatioConjugate * float(SSSampleIndex+sampleIndex), 1.0f);

                    break;
                }
                static_assert(RNG_SOURCE >= RNGSource::WhiteNoise && RNG_SOURCE <= RNGSource::BlueNoiseGR, "RNG_SOURCE invalid");
            }

            // calculate sample position
            float sampleX, sampleY;
            switch (RAY_PATTERN)
            {
                case RayPattern::None:
                {
                    sampleX = rngX;
                    sampleY = rngY;
                    break;
                }
                case RayPattern::Grid:
                {
                    if (SHADOW_RAY_COUNT == 1)
                    {
                        sampleX = sampleY = 0.0f;
                    }
                    else
                    {
                        sampleX = (float(sampleIndex % SHADOW_RAY_COUNT_GRID_SIZE) + 1.0f) / float(SHADOW_RAY_COUNT_GRID_SIZE + 1);
                        sampleY = (float(sampleIndex / SHADOW_RAY_COUNT_GRID_SIZE) + 1.0f) / float(SHADOW_RAY_COUNT_GRID_SIZE + 1);
                    }
                    break;
                }
                case RayPattern::Stratified:
                {
                    sampleX = (float(sampleIndex % SHADOW_RAY_COUNT_GRID_SIZE) + rngX) / float(SHADOW_RAY_COUNT_GRID_SIZE);
                    sampleY = (float(sampleIndex / SHADOW_RAY_COUNT_GRID_SIZE) + rngY) / float(SHADOW_RAY_COUNT_GRID_SIZE);
                    break;
                }
                static_assert(RAY_PATTERN >= RayPattern::None && RAY_PATTERN <= RayPattern::Stratified,"RAY_PATTERN invalid");
            }

            float3 randomDir = RandomVectorTowardsLight(lightDir, lightDiscRadius, sampleX, sampleY);

            SHitInfo shadowHitInfo;
            shadowHitInfo.collisionTime = lightDistance;

            bool intersectionFound = false;
            for (size_t i = 0; i < sizeof(g_spheres) / sizeof(g_spheres[0]) && !intersectionFound; ++i)
                intersectionFound |= RayIntersect(shadowPos, randomDir, g_spheres[i], shadowHitInfo);
            for (size_t i = 0; i < sizeof(g_quads) / sizeof(g_quads[0]) && !intersectionFound; ++i)
                intersectionFound |= RayIntersect(shadowPos, randomDir, g_quads[i], shadowHitInfo);

            if (intersectionFound)
                ret.shadowMultipliersPositional[lightIndex] = Lerp(ret.shadowMultipliersPositional[lightIndex], 0.0f, 1.0f / float(1+sampleIndex));
            else
                ret.shadowMultipliersPositional[lightIndex] = Lerp(ret.shadowMultipliersPositional[lightIndex], 1.0f, 1.0f / float(1+sampleIndex));
        }
    }

    return ret;
}

//-------------------------------------------------------------------------------------------------------------------

float3 PixelFunctionShade (const SGbufferPixel& gbufferData)
{
    if (gbufferData.hitInfo.collisionTime <= 0.0f)
        return g_skyColor;

    if (gbufferData.isLight)
        return gbufferData.hitInfo.emissive;

    float3 ret = g_skyColor * gbufferData.hitInfo.albedo;

    for (size_t lightIndex = 0; lightIndex < c_numLights; ++lightIndex)
    {
        float3 lightDir = Normalize(g_positionalLights[lightIndex].position - gbufferData.worldPos);
        float lightDistance = Length(g_positionalLights[lightIndex].position - gbufferData.worldPos);

        float falloff = 1.0f / (lightDistance * lightDistance);

        float NdotL = Dot(lightDir, gbufferData.hitInfo.normal);
        if (NdotL > 0.0f)
            ret = ret + g_positionalLights[lightIndex].color * gbufferData.hitInfo.albedo * NdotL * gbufferData.shadowMultipliersPositional[lightIndex] * falloff / c_pi;
    }

    return ret;
}

//-------------------------------------------------------------------------------------------------------------------

template <size_t GBUFFER_SAMPLES>
void ShadePixels(const std::vector<SGbufferPixel>& gbuffer, SImageData<3>& output)
{
    float3* pixel = (float3*)&output.m_pixels[0];
    const SGbufferPixel* gbufferPixel = &gbuffer[0];
    const size_t numPixels = gbuffer.size() / GBUFFER_SAMPLES;
    for (size_t pixelIndex = 0; pixelIndex < numPixels; ++pixelIndex)
    {
        *pixel = { 0.0f, 0.0f, 0.0f };
        for (size_t sampleIndex = 0; sampleIndex < GBUFFER_SAMPLES; ++sampleIndex)
            *pixel = *pixel + PixelFunctionShade(gbufferPixel[sampleIndex]);
        gbufferPixel += GBUFFER_SAMPLES;
        *pixel = *pixel / float(GBUFFER_SAMPLES);
        pixel++;
    }
}

//-------------------------------------------------------------------------------------------------------------------

template <size_t SHADOW_RAY_COUNT, size_t SHADOW_RAY_COUNT_GRID_SIZE, RayPattern RAY_PATTERN, RNGSource RNG_SOURCE, bool STOCHASTIC_LIGHT_SOURCE, size_t SHADOW_RESOLUTION_DIVIDER>
void GeneratePixels(const char* task, const char* baseFileName, EProcess preProcess, EProcess postProcess)
{
    char fileName[256];

    float aspectRatio = float(IMAGE_WIDTH()) / float(IMAGE_HEIGHT());

    SImageData<3> output(IMAGE_WIDTH()/ SHADOW_RESOLUTION_DIVIDER, IMAGE_HEIGHT()/ SHADOW_RESOLUTION_DIVIDER);
    size_t numPixels = output.m_width * output.m_height;
    std::vector<SGbufferPixel> gbuffer;
    gbuffer.resize(numPixels*STRATIFIED_SAMPLE_COUNT());

    // Generate gbuffer data
    RunMultiThreaded(task,
        [&] (std::atomic<size_t>& atomicRowIndex)
        {
            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);

            // pixel size is 2x because uv goes from -1 to 1, so is twice as big.
            float pixelSizeU = 2.0f / float(output.m_width);
            float pixelSizeV = 2.0f / float(output.m_height);

            size_t rowIndex = atomicRowIndex.fetch_add(1);
            bool reportProgress = (rowIndex == 0);
            int oldPercent = -1;
            while (rowIndex < output.m_height)
            {
                // do a row of work at a time
                size_t pixelBase = rowIndex * output.m_width;
                for (size_t pixelOffset = 0; pixelOffset < output.m_width; ++pixelOffset)
                {
                    size_t pixelIndex = pixelBase + pixelOffset;

                    // calculate uv coordinate of pixel in [-1,1], also correcting for aspect ratio and flipping the vertical axis
                    float u = ((float(pixelIndex % output.m_width) / float(output.m_width)) * 2.0f - 1.0f) * aspectRatio;
                    float v = ((float(pixelIndex / output.m_width) / float(output.m_height)) * 2.0f - 1.0f) * -1.0f;

                    // do multiple samples per pixel via stratified sampling and combine with a box filter
                    float3 sampleSum = { 0.0f, 0.0f, 0.0f };
                    for (size_t sampleIndex = 0; sampleIndex < STRATIFIED_SAMPLE_COUNT(); ++sampleIndex)
                    {
                        SGbufferPixel& pixel = gbuffer[pixelIndex * 4 + sampleIndex];

                        float stratU = float((sampleIndex) % STRATIFIED_SAMPLE_COUNT_ONE_AXIS()) / float(STRATIFIED_SAMPLE_COUNT_ONE_AXIS());
                        float stratV = float(((sampleIndex) / STRATIFIED_SAMPLE_COUNT_ONE_AXIS()) % STRATIFIED_SAMPLE_COUNT_ONE_AXIS()) / float(STRATIFIED_SAMPLE_COUNT_ONE_AXIS());

                        stratU += dist(rng) / float(STRATIFIED_SAMPLE_COUNT_ONE_AXIS());
                        stratV += dist(rng) / float(STRATIFIED_SAMPLE_COUNT_ONE_AXIS());

                        stratU *= pixelSizeU;
                        stratV *= pixelSizeV;

                        pixel.u = u + stratU;
                        pixel.v = v + stratV;

                        pixel = PixelFunctionGBuffer<SHADOW_RAY_COUNT, SHADOW_RAY_COUNT_GRID_SIZE, RAY_PATTERN, RNG_SOURCE, STOCHASTIC_LIGHT_SOURCE>(pixel.u, pixel.v, pixelIndex % output.m_width, pixelIndex / output.m_width, sampleIndex, rng);
                    }

                    // report progress
                    if (reportProgress)
                    {
                        int newPercent = (int)(100.0f * float(pixelIndex) / float(numPixels));
                        if (oldPercent != newPercent)
                        {
                            printf("\rGbuffer: %i%%", newPercent);
                            oldPercent = newPercent;
                        }
                    }
                }

                rowIndex = atomicRowIndex.fetch_add(1);
            }

            if (reportProgress)
                printf("\rGbuffer: 100%%\n");
        }
    );

    // upsize the gbuffer and output if we should
    // TODO: this isn't quite right. need to shrink the shadow gbuffer data. can't upsize the entire gbuffer.
    if (SHADOW_RESOLUTION_DIVIDER != 1)
    {
        SImageData<3> newOutput(IMAGE_WIDTH(), IMAGE_HEIGHT());

        size_t newNumPixels = newOutput.m_width * newOutput.m_height;
        
        std::vector<SGbufferPixel> newGbuffer;
        newGbuffer.resize(newNumPixels*STRATIFIED_SAMPLE_COUNT());

        for (size_t destY = 0; destY < IMAGE_HEIGHT(); ++destY)
        {
            size_t srcY = clamp(destY / SHADOW_RESOLUTION_DIVIDER, (size_t)0, output.m_height - 1);
            for (size_t destX = 0; destX < IMAGE_WIDTH(); ++destX)
            {
                size_t srcX = clamp(destX / SHADOW_RESOLUTION_DIVIDER, (size_t)0, output.m_width - 1);

                const SGbufferPixel* src = &gbuffer[(srcY*output.m_width + srcX)* STRATIFIED_SAMPLE_COUNT()];
                SGbufferPixel* dest = &newGbuffer[(destY*newOutput.m_width + destX)* STRATIFIED_SAMPLE_COUNT()];

                for (size_t sampleIndex = 0; sampleIndex < STRATIFIED_SAMPLE_COUNT(); ++sampleIndex)
                {
                    for (size_t lightIndex = 0; lightIndex < c_numLights; ++lightIndex)
                    {
                        dest[sampleIndex].shadowMultipliersPositional[lightIndex] = src[sampleIndex].shadowMultipliersPositional[lightIndex];
                    }
                }
            }
        }

        output = newOutput;
        gbuffer = newGbuffer;
        numPixels = newNumPixels;
    }

    // do gbuffer processing if we should
    if (preProcess != EProcess::No)
    {
        // for pre-processing the gbuffer, we want it to be as if we only took 1 shadow sample per pixel before filtering.
        std::vector<SGbufferPixel> gbuffer1ShadowSample = gbuffer;
        size_t pixelIndex = 0;
        while (pixelIndex < gbuffer1ShadowSample.size())
        {
            for (size_t sampleIndex = 1; sampleIndex < STRATIFIED_SAMPLE_COUNT(); ++sampleIndex)
                for (size_t lightIndex = 0; lightIndex < c_numLights; ++lightIndex)
                    gbuffer1ShadowSample[pixelIndex+sampleIndex].shadowMultipliersPositional[lightIndex] = gbuffer1ShadowSample[pixelIndex].shadowMultipliersPositional[lightIndex];

            pixelIndex += STRATIFIED_SAMPLE_COUNT();
        }

        std::vector<SGbufferPixel> gbufferFiltered;

        BoxBlurShadowMultipliers<2, STRATIFIED_SAMPLE_COUNT()>(gbuffer1ShadowSample, gbufferFiltered, output.m_width, output.m_height);
        ShadePixels<STRATIFIED_SAMPLE_COUNT()>(gbufferFiltered, output);
        sprintf_s(fileName, baseFileName, "_prebox5");
        output.Save(fileName);

        BoxBlurShadowMultipliers<3, STRATIFIED_SAMPLE_COUNT()>(gbuffer1ShadowSample, gbufferFiltered, output.m_width, output.m_height);
        ShadePixels<STRATIFIED_SAMPLE_COUNT()>(gbufferFiltered, output);
        sprintf_s(fileName, baseFileName, "_prebox7");
        output.Save(fileName);

        TriangleBlurShadowMultipliers<2, STRATIFIED_SAMPLE_COUNT()>(gbuffer1ShadowSample, gbufferFiltered, output.m_width, output.m_height);
        ShadePixels<STRATIFIED_SAMPLE_COUNT()>(gbufferFiltered, output);
        sprintf_s(fileName, baseFileName, "_pretriangle5");
        output.Save(fileName);

        TriangleBlurShadowMultipliers<3, STRATIFIED_SAMPLE_COUNT()>(gbuffer1ShadowSample, gbufferFiltered, output.m_width, output.m_height);
        ShadePixels<STRATIFIED_SAMPLE_COUNT()>(gbufferFiltered, output);
        sprintf_s(fileName, baseFileName, "_pretriangle7");
        output.Save(fileName);

        if (preProcess == EProcess::Exhaustive)
        {
            BoxBlurShadowMultipliers<1, STRATIFIED_SAMPLE_COUNT()>(gbuffer1ShadowSample, gbufferFiltered, output.m_width, output.m_height);
            ShadePixels<STRATIFIED_SAMPLE_COUNT()>(gbufferFiltered, output);
            sprintf_s(fileName, baseFileName, "_prebox3");
            output.Save(fileName);

            BoxBlurShadowMultipliers<7, STRATIFIED_SAMPLE_COUNT()>(gbuffer1ShadowSample, gbufferFiltered, output.m_width, output.m_height);
            ShadePixels<STRATIFIED_SAMPLE_COUNT()>(gbufferFiltered, output);
            sprintf_s(fileName, baseFileName, "_prebox15");
            output.Save(fileName);

            TriangleBlurShadowMultipliers<1, STRATIFIED_SAMPLE_COUNT()>(gbuffer1ShadowSample, gbufferFiltered, output.m_width, output.m_height);
            ShadePixels<STRATIFIED_SAMPLE_COUNT()>(gbufferFiltered, output);
            sprintf_s(fileName, baseFileName, "_pretriangle3");
            output.Save(fileName);

            TriangleBlurShadowMultipliers<7, STRATIFIED_SAMPLE_COUNT()>(gbuffer1ShadowSample, gbufferFiltered, output.m_width, output.m_height);
            ShadePixels<STRATIFIED_SAMPLE_COUNT()>(gbufferFiltered, output);
            sprintf_s(fileName, baseFileName, "_pretriangle15");
            output.Save(fileName);
        }
    }

    // make and save the unprocessed image
    {
        ShadePixels<STRATIFIED_SAMPLE_COUNT()>(gbuffer, output);
        sprintf_s(fileName, baseFileName, "");
        output.Save(fileName);
    }

    // do image post processing if we should
    if (postProcess != EProcess::No)
    {
        SImageData<3> filtered;

        BoxBlur<1>(output, filtered);
        sprintf_s(fileName, baseFileName, "_postbox3");
        filtered.Save(fileName);

        BoxBlur<2>(output, filtered);
        sprintf_s(fileName, baseFileName, "_postbox5");
        filtered.Save(fileName);

        TriangleBlur<1>(output, filtered);
        sprintf_s(fileName, baseFileName, "_posttriangle3");
        filtered.Save(fileName);

        TriangleBlur<2>(output, filtered);
        sprintf_s(fileName, baseFileName, "_posttriangle5");
        filtered.Save(fileName);

        MedianFilter<1>(output, filtered);
        sprintf_s(fileName, baseFileName, "_postmedian3");
        filtered.Save(fileName);

        MedianFilter<2>(output, filtered);
        sprintf_s(fileName, baseFileName, "_postmedian5");
        filtered.Save(fileName);

        if (preProcess == EProcess::Exhaustive)
        {
            BoxBlur<7>(output, filtered);
            sprintf_s(fileName, baseFileName, "_postbox15");
            filtered.Save(fileName);

            TriangleBlur<7>(output, filtered);
            sprintf_s(fileName, baseFileName, "_posttriangle15");
            filtered.Save(fileName);

            MedianFilter<7>(output, filtered);
            sprintf_s(fileName, baseFileName, "_postmedian15");
            filtered.Save(fileName);
        }
    }
}

//-------------------------------------------------------------------------------------------------------------------

float3 PixelFunctionPathTrace(const float3& rayPos, const float3& rayDir, std::mt19937& rng, int depth = 0)
{
    // check for intersections with geometry
    SHitInfo hitInfo;
    for (size_t i = 0; i < sizeof(g_spheres) / sizeof(g_spheres[0]); ++i)
        RayIntersect(rayPos, rayDir, g_spheres[i], hitInfo);
    for (size_t i = 0; i < sizeof(g_quads) / sizeof(g_quads[0]); ++i)
        RayIntersect(rayPos, rayDir, g_quads[i], hitInfo);

    // check for intersections with spherical lights
    for (size_t lightIndex = 0; lightIndex < c_numLights; ++lightIndex)
    {
        SSphere lightSphere;
        lightSphere.position = g_positionalLights[lightIndex].position;
        lightSphere.radius = g_positionalLights[lightIndex].radius;
        lightSphere.albedo = { 0.0f, 0.0f, 0.0f };

        if (RayIntersect(rayPos, rayDir, lightSphere, hitInfo))
            hitInfo.emissive = g_positionalLights[lightIndex].color;
    }

    // if nothing hit, return the sky color
    if (hitInfo.collisionTime < 0.0f)
        return g_skyColor;

    // recurse
    float3 incomingLight = { 0.0f, 0.0f, 0.0f };
    if (depth < PATHTRACING_RAY_BOUNCE())
    {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float3 recursiveRayPos = (rayPos + rayDir * hitInfo.collisionTime) + hitInfo.normal * c_rayEpsilon;
        float3 recursiveRayDir = CosineSampleHemisphere(hitInfo.normal, dist(rng), dist(rng));
        incomingLight = PixelFunctionPathTrace(recursiveRayPos, recursiveRayDir, rng, depth+1);
    }

    // return shaded shaded surface color.
    // No need to multiply by N dot L because we cosine sampled the hemisphere, aka it's importance sampled for the NdotL multiplication.
    return hitInfo.emissive + incomingLight * hitInfo.albedo;
}

//-------------------------------------------------------------------------------------------------------------------

void Pathtrace(const char* fileName)
{
    SImageData<3> output(IMAGE_WIDTH(), IMAGE_HEIGHT());
    const size_t numPixels = output.m_width * output.m_height;

    float aspectRatio = float(output.m_width) / float(output.m_height);

    // Generate gbuffer data
    RunMultiThreaded("Pathtrace",
        [&] (std::atomic<size_t>& atomicRowIndex)
        {
            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);

            size_t rowIndex = atomicRowIndex.fetch_add(1);
            bool reportProgress = (rowIndex == 0);
            int oldPercent = -1;
            while (rowIndex < output.m_height)
            {
                // do a row of work at a time
                size_t pixelBase = rowIndex * output.m_width;
                float3* outPixel = (float3*)output.GetPixel(0, rowIndex);
                for (size_t pixelOffset = 0; pixelOffset < output.m_width; ++pixelOffset, ++outPixel)
                {
                    size_t pixelIndex = pixelBase + pixelOffset;

                    // do multiple samples per pixel via stratified sampling and combine with a box filter
                    float3 sampleSum = { 0.0f, 0.0f, 0.0f };
                    for (size_t sampleIndex = 0; sampleIndex < STRATIFIED_SAMPLE_COUNT_PATHTRACING(); ++sampleIndex)
                    {
                        size_t sampleX = sampleIndex % STRATIFIED_SAMPLE_COUNT_ONE_AXIS_PATHTRACING();
                        size_t sampleY = sampleIndex / STRATIFIED_SAMPLE_COUNT_ONE_AXIS_PATHTRACING();

                        float stratU = (float(sampleX)+dist(rng)) / float(STRATIFIED_SAMPLE_COUNT_ONE_AXIS_PATHTRACING());
                        float stratV = -(float(sampleY)+dist(rng)) / float(STRATIFIED_SAMPLE_COUNT_ONE_AXIS_PATHTRACING());

                        // calculate uv coordinate of pixel in [-1,1], also correcting for aspect ratio and flipping the vertical axis
                        float u = (((float(pixelIndex % output.m_width)+stratU) / float(output.m_width)) * 2.0f - 1.0f) * aspectRatio;
                        float v = (((float(pixelIndex / output.m_width)+stratV) / float(output.m_height)) * 2.0f - 1.0f) * -1.0f;

                        float3 rayPos = g_cameraPos;
                        float3 rayDir = Normalize(g_cameraX * u + g_cameraY * v + g_cameraZ * g_cameraDistance);

                        float3 sample = PixelFunctionPathTrace(rayPos, rayDir, rng);
                        *outPixel = Lerp(*outPixel, sample, 1.0f / float(sampleIndex+1));
                    }

                    // report progress
                    if (reportProgress)
                    {
                        int newPercent = (int)(100.0f * float(pixelIndex) / float(numPixels));
                        if (oldPercent != newPercent)
                        {
                            printf("\r%i%%", newPercent);
                            oldPercent = newPercent;
                        }
                    }
                }

                rowIndex = atomicRowIndex.fetch_add(1);
            }

            if (reportProgress)
                printf("\r100%%\n");
        }
    );

    // save the image
    output.Save(fileName);
}

//-------------------------------------------------------------------------------------------------------------------

int main (int argc, char** argv)
{
    // make a 4 channel blue noise tetxure
    {
        SImageData<3> blueNoiseTextures[4];
        if (!blueNoiseTextures[0].Load("LDR_RGB1_0.png", false) ||
            !blueNoiseTextures[1].Load("LDR_RGB1_1.png", false) ||
            !blueNoiseTextures[2].Load("LDR_RGB1_2.png", false) ||
            !blueNoiseTextures[3].Load("LDR_RGB1_3.png", false))
        {
            printf("Could not load the blue noise textures!\n");
            return 1;
        }

        g_blueNoiseTexture.m_width = blueNoiseTextures[0].m_width;
        g_blueNoiseTexture.m_height = blueNoiseTextures[0].m_height;
        g_blueNoiseTexture.m_pixels.resize(g_blueNoiseTexture.m_height*g_blueNoiseTexture.Pitch());
        for (size_t i = 0, c = blueNoiseTextures[0].m_width*blueNoiseTextures[0].m_height; i < c; ++i)
        {
            g_blueNoiseTexture.m_pixels[i * 4 + 0] = blueNoiseTextures[0].m_pixels[i * 3];
            g_blueNoiseTexture.m_pixels[i * 4 + 1] = blueNoiseTextures[1].m_pixels[i * 3];
            g_blueNoiseTexture.m_pixels[i * 4 + 2] = blueNoiseTextures[2].m_pixels[i * 3];
            g_blueNoiseTexture.m_pixels[i * 4 + 3] = blueNoiseTextures[3].m_pixels[i * 3];
        }
    }

    // init geometry
    for (size_t i = 0; i < sizeof(g_quads) / sizeof(g_quads[0]); ++i)
        g_quads[i].CalculateNormal();

    // path traced version
    #if DO_PATHTRACE()
        Pathtrace("out/A_Pathtrace.png");
    #endif

        // TODO: move to end
        GeneratePixels<1, 3, RayPattern::None, RNGSource::BlueNoiseGR, false, 2>("Blue 1 Half", "out/I_blue_1_half%s.png", EProcess::Yes, EProcess::No);
        GeneratePixels<1, 3, RayPattern::None, RNGSource::BlueNoiseGR, false, 3>("Blue 1 Third", "out/I_blue_1_thirds.png", EProcess::Yes, EProcess::No);

    // make sampled images
    GeneratePixels<1, 1, RayPattern::Grid, RNGSource::WhiteNoise, false, 1>("Hard", "out/B_hard_1%s.png", EProcess::No, EProcess::No);

    GeneratePixels<1, 3, RayPattern::None, RNGSource::WhiteNoise, false, 1>("White 1", "out/C_white_1%s.png", EProcess::Exhaustive, EProcess::Exhaustive);
    GeneratePixels<2, 3, RayPattern::None, RNGSource::WhiteNoise, false, 1>("White 2", "out/C_white_2%s.png", EProcess::No, EProcess::No);
    GeneratePixels<4, 3, RayPattern::None, RNGSource::WhiteNoise, false, 1>("White 4", "out/C_white_4%s.png", EProcess::No, EProcess::No);
    GeneratePixels<8, 3, RayPattern::None, RNGSource::WhiteNoise, false, 1>("White 8", "out/C_white_8%s.png", EProcess::Yes, EProcess::No);
    GeneratePixels<256, 3, RayPattern::None, RNGSource::WhiteNoise, false, 1>("White 256", "out/C_white_256%s.png", EProcess::No, EProcess::No);

    GeneratePixels<9, 3, RayPattern::Grid, RNGSource::WhiteNoise, false, 1>("Grid 9", "out/D_grid_9%s.png", EProcess::Yes, EProcess::No);
    GeneratePixels<256, 16, RayPattern::Grid, RNGSource::WhiteNoise, false, 1>("Grid 256", "out/D_grid_256%s.png", EProcess::No, EProcess::No);

    GeneratePixels<1, 3, RayPattern::None, RNGSource::BlueNoiseGR, false, 1>("Blue 1", "out/E_blue_1%s.png", EProcess::Yes, EProcess::No);
    GeneratePixels<2, 3, RayPattern::None, RNGSource::BlueNoiseGR, false, 1>("Blue 2", "out/E_blue_2%s.png", EProcess::No, EProcess::No);
    GeneratePixels<4, 3, RayPattern::None, RNGSource::BlueNoiseGR, false, 1>("Blue 4", "out/E_blue_4%s.png", EProcess::No, EProcess::No);
    GeneratePixels<8, 3, RayPattern::None, RNGSource::BlueNoiseGR, false, 1>("Blue 8", "out/E_blue_8%s.png", EProcess::Yes, EProcess::No);
    GeneratePixels<256, 3, RayPattern::None, RNGSource::BlueNoiseGR, false, 1>("Blue 256", "out/E_blue_256%s.png", EProcess::No, EProcess::No);

    GeneratePixels<4, 2, RayPattern::Stratified, RNGSource::WhiteNoise, false, 1>("Stratified White 4", "out/F_stratified_white_4%s.png", EProcess::Yes, EProcess::No);
    GeneratePixels<9, 3, RayPattern::Stratified, RNGSource::WhiteNoise, false, 1>("Stratified White 9", "out/F_stratified_white_9%s.png", EProcess::Yes, EProcess::No);

    GeneratePixels<4, 2, RayPattern::Stratified, RNGSource::BlueNoiseGR, false, 1>("Stratified Blue 4", "out/G_stratified_blue_4%s.png", EProcess::Yes, EProcess::No);
    GeneratePixels<9, 3, RayPattern::Stratified, RNGSource::BlueNoiseGR, false, 1>("Stratified Blue 9", "out/G_stratified_blue_9%s.png", EProcess::Yes, EProcess::No);

    GeneratePixels<1, 3, RayPattern::None, RNGSource::WhiteNoise, true, 1>("White 1 Stochastic Light", "out/H_white_1_stochastic%s.png", EProcess::Yes, EProcess::No);
    GeneratePixels<1, 3, RayPattern::None, RNGSource::BlueNoiseGR, true, 1>("Blue 1 Stochastic Light", "out/H_blue_1_stochastic%s.png", EProcess::Yes, EProcess::No);

    

    system("pause");
    return 0;
}

/*

TODO:

* trim the number of files actually generated

* median filter gbuffer

? why is the edges of the light so rough looking? we do a lot of samples per pixel in PT and RT, it should smooth out.
 * maybe because the light is so bright? maybe we need to fully shade each pixel, and combine that, instead of combining when shading
 * didn't really help... maybe there is a secret to supersampling HDR?

* subtract the radius of the light from the light distance used for interesection test AND falloff

* try shining a light from behind the camera to a quad perpendicular to the camera. no distance falloff.   Make sure the quad is the correct color, and same on both pathtraced and raytraced

* blue64 and blue 256 don't look any different
 * white 64 and 256 don't look very different either

* make a better scene?

* do a blur where you blur NxN sections instead of each pixel reading every other pixel. This simulates something more quickly / easily done in a ray dispatch shader
* try doing a blur where every 2x2 block is averaged. (a gbuffer blur! maybe depth aware)

* try gaussian blur too? at least for white 1 aka exhaustive

* try fitting the shadow data to a function? piecewise least squares?

* ray marched shadows (single ray that keeps track of closest point)
 * and this way https://www.shadertoy.com/view/lsKcDD

* animated & make an animated gif of results? (gif is low quality though... maybe ffmpeg?)

* todos

BLOG Notes:
* Using gamma 2.2
* box, triangle, gaussian are separable which is faster

*/