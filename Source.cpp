#include <atomic>
#include <thread>
#include <vector>
#include <array>
#include <stdint.h>
#include <stdlib.h>
#include <random>
#include <algorithm>

#define IMAGE_WIDTH() 400
#define IMAGE_HEIGHT() 400

#define STRATIFIED_SAMPLE_COUNT_ONE_AXIS() 2  // it does this many samples squared per pixel for AA

#define FORCE_SINGLE_THREADED() 0  // useful for debugging

// stb_image is an amazing header only image library (aka no linking, just include the headers!).  http://nothings.org/stb
#pragma warning( disable : 4996 ) 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#pragma warning( default : 4996 ) 

typedef uint8_t uint8;
typedef std::array<float, 3> float3;
typedef std::array<std::array<float, 3>, 3> float3x3;

float LinearTosRGB(float value);
float sRGBToLinear(float value);

static const float c_rayEpsilon = 0.01f; // value used to push the ray a little bit away from surfaces before doing shadow rays
static const float c_pi = 3.14159265359f;
static const float c_goldenRatioConjugate = 0.61803398875f;

#define STRATIFIED_SAMPLE_COUNT() (STRATIFIED_SAMPLE_COUNT_ONE_AXIS()*STRATIFIED_SAMPLE_COUNT_ONE_AXIS())

//-------------------------------------------------------------------------------------------------------------------

float3 operator* (const float3& a, const float3& b)
{
    float3 ret;
    ret[0] = a[0] * b[0];
    ret[1] = a[1] * b[1];
    ret[2] = a[2] * b[2];
    return ret;
}

float3 operator+ (const float3& a, const float3& b)
{
    float3 ret;
    ret[0] = a[0] + b[0];
    ret[1] = a[1] + b[1];
    ret[2] = a[2] + b[2];
    return ret;
}

float3 operator- (const float3& a, const float3& b)
{
    float3 ret;
    ret[0] = a[0] - b[0];
    ret[1] = a[1] - b[1];
    ret[2] = a[2] - b[2];
    return ret;
}

float3 operator* (const float3& a, float b)
{
    float3 ret;
    ret[0] = a[0] * b;
    ret[1] = a[1] * b;
    ret[2] = a[2] * b;
    return ret;
}

float3 operator+ (const float3& a, float b)
{
    float3 ret;
    ret[0] = a[0] + b;
    ret[1] = a[1] + b;
    ret[2] = a[2] + b;
    return ret;
}

float3 operator/ (const float3& a, float b)
{
    return a * (1.0f / b);
}

float3x3 operator* (const float3x3& a, float b)
{
    float3x3 ret;
    for (size_t y = 0; y < 3; ++y)
    {
        for (size_t x = 0; x < 3; ++x)
        {
            ret[y][x] = a[y][x] * b;
        }
    }
    return ret;
}

float3 operator* (const float3x3& a, const float3& b)
{
    float3 ret = { 0.0f, 0.0f, 0.0f };
    for (size_t y = 0; y < 3; ++y)
    {
        for (size_t x = 0; x < 3; ++x)
        {
            ret[y] = a[y][x] *  b[x];
        }
    }
    return ret;
}

float3x3 operator+ (const float3x3& a, const float3x3& b)
{
    float3x3 ret;
    for (size_t y = 0; y < 3; ++y)
    {
        for (size_t x = 0; x < 3; ++x)
        {
            ret[y][x] = a[y][x] + b[y][x];
        }
    }
    return ret;
}

float Length(const float3& a)
{
    return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

float3 Normalize(const float3& a)
{
    return a / Length(a);
}

float Dot(const float3& a, const float3& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline float3 Cross (const float3& a, const float3& b)
{
    return
    {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

inline float ScalarTriple(const float3& a, const float3& b, const float3& c)
{
    return Dot(Cross(a, b), c);
}

inline float3x3 Transpose(const float3x3& a)
{
    float3x3 ret;
    ret[0] = { a[0][0], a[1][0], a[2][0] };
    ret[1] = { a[0][1], a[1][1], a[2][1] };
    ret[2] = { a[0][2], a[1][2], a[2][2] };
    return ret;
}

inline float3x3 MultiplyVectorByTranspose(const float3& a)
{
    float3x3 ret;
    for (size_t y = 0; y < 3; ++y)
    {
        for (size_t x = 0; x < 3; ++x)
        {
            ret[y][x] = a[y] * a[x];
        }
    }
    return ret;
}

inline float Lerp (float a, float b, float t)
{
    return a * (1.0f - t) + b * t;
}

template <typename T>
T clamp(T value, T min, T max)
{
    if (value <= min)
        return min;
    else if (value >= max)
        return max;
    else
        return value;
}

//-------------------------------------------------------------------------------------------------------------------
float3 RandomVectorTowardsLight (float3 lightDir, float lightSolidAngleRadius, float rngX, float rngY)
{
    // TODO: should directional lights always store this value?
    float radius = sinf(lightSolidAngleRadius);

    // make basis vectors for the light quad
    lightDir = Normalize(lightDir);
    float3 lightQuadUAxis = Normalize(Cross(float3{ 0.0f, 1.0f, 0.0f }, lightDir));
    float3 lightQuadVAxis = Normalize(Cross(lightDir, lightQuadUAxis));

    // make rng values from [0,1] to [-radius,radius]
    rngX = (rngX * 2.0f - 1.0f) * radius;
    rngY = (rngY * 2.0f - 1.0f) * radius;

    // return the ray
    return Normalize(lightDir + lightQuadUAxis * rngX + lightQuadVAxis * rngY);

    // TODO: this is a quad, should switch to disc
}

//-------------------------------------------------------------------------------------------------------------------
float3 RandomVectorInsideCone (float3 coneDir, float coneAngle, float rngX, float rngY)
{
    // Translated from: https://stackoverflow.com/questions/38997302/create-random-unit-vector-inside-a-defined-conical-region


    // Generate points on the spherical cap around the north pole [1].
    // [1] See https://math.stackexchange.com/a/205589/81266

    float z = rngX * (1.0f - std::cosf(coneAngle)) + std::cosf(coneAngle);
    float phi = rngY * 2.0f * c_pi;
    float x = std::sqrtf(1.0f - z * z) * std::cosf(phi);
    float y = std::sqrtf(1.0f - z * z) * std::sinf(phi);

    // Find the rotation axis `u` and rotation angle `rot`[1]
    float3 u = Normalize(Cross({ 0.0f, 0.0f, 1.0f }, Normalize(coneDir)));
    float rot = std::acosf(Dot(Normalize(coneDir), { 0.0f, 0.0f, 1.0f }));

    // Convert rotation axis and angle to 3x3 rotation matrix [2]
    // [2] See https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    float3x3 crossMatrix;
    crossMatrix[0] = { 0.0f, -u[2], u[1] };
    crossMatrix[1] = { u[2], 0.0f, -u[0] };
    crossMatrix[2] = { -u[1], u[0], 0.0f };

    float3x3 identity;
    identity[0] = { 1.0f, 0.0f, 0.0f };
    identity[1] = { 0.0f, 1.0f, 0.0f };
    identity[2] = { 0.0f, 0.0f, 1.0f };

    float3x3 R = identity * std::cosf(rot) + crossMatrix * std::sinf(rot) + MultiplyVectorByTranspose(u) * (1.0f - std::cosf(rot));

    // Rotate[x; y; z] from north pole to `coneDir`.
    float3 r = { x, y, z };
    return R * r;


    /*
    coneAngle = coneAngleDegree * pi/180;

    % Generate points on the spherical cap around the north pole [1].
    % [1] See https://math.stackexchange.com/a/205589/81266
    z = RNG.rand(1, N) * (1 - cos(coneAngle)) + cos(coneAngle);
    phi = RNG.rand(1, N) * 2 * pi;
    x = sqrt(1-z.^2).*cos(phi);
    y = sqrt(1-z.^2).*sin(phi);

    % If the spherical cap is centered around the north pole, we're done.
    if all(coneDir(:) == [0;0;1])
    r = [x; y; z];
    return;
    end

    % Find the rotation axis `u` and rotation angle `rot` [1]
    u = normc(cross([0;0;1], normc(coneDir)));
    rot = acos(dot(normc(coneDir), [0;0;1]));

    % Convert rotation axis and angle to 3x3 rotation matrix [2]
    % [2] See https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    crossMatrix = @(x,y,z) [0 -z y; z 0 -x; -y x 0];
    R = cos(rot) * eye(3) + sin(rot) * crossMatrix(u(1), u(2), u(3)) + (1-cos(rot))*(u * u');

    % Rotate [x; y; z] from north pole to `coneDir`.
    r = R * [x; y; z];
    */
}

//-------------------------------------------------------------------------------------------------------------------
template <size_t NUM_CHANNELS>
struct SImageData
{
    SImageData (size_t width=0, size_t height=0)
        : m_width(width)
        , m_height(height)
    {
        m_pixels.resize(m_width*m_height * NUM_CHANNELS);
    }

    bool Load (const char* fileName, bool isSRGB)
    {
        int width, height, channels;
        unsigned char* pixels = stbi_load(fileName, &width, &height, &channels, NUM_CHANNELS);

        if (!pixels)
            return false;

        m_width = width;
        m_height = height;
        m_pixels.resize(m_width*m_height * NUM_CHANNELS);

        for (size_t i = 0; i < m_pixels.size(); ++i)
        {
            m_pixels[i] = float(pixels[i]) / 255.0f;
            if (isSRGB)
                m_pixels[i] = sRGBToLinear(m_pixels[i]);
        }

        stbi_image_free(pixels);

        return true;
    }

    bool Save (const char* fileName, bool toSRGB = true)
    {
        // convert from linear f32 to sRGB u8
        std::vector<uint8> pixelsU8;
        pixelsU8.resize(m_pixels.size());
        for (size_t i = 0; i < m_pixels.size(); ++i)
        {
            if (toSRGB)
            {
                pixelsU8[i] = uint8(LinearTosRGB(m_pixels[i])*255.0f);
            }
            else
            {
                pixelsU8[i] = uint8(m_pixels[i]*255.0f);
            }
        }

        // save the image
        return (stbi_write_png(fileName, (int)m_width, (int)m_height, NUM_CHANNELS, &pixelsU8[0], (int)Pitch()) == 1);
    }

    size_t NumChannels () const { return NUM_CHANNELS; }

    size_t Pitch () const { return m_width * NUM_CHANNELS; }

    float* GetPixel(size_t x, size_t y)
    {
        return &m_pixels[y*Pitch() + x* NUM_CHANNELS];
    }

    const float* GetPixel(size_t x, size_t y) const
    {
        return &m_pixels[y*Pitch() + x * NUM_CHANNELS];
    }

    size_t m_width;
    size_t m_height;
    std::vector<float> m_pixels;
};

struct SPositionalLight
{
    float3 position;
    float radius;

    float3 color;
};

struct SDirectionalLight
{
    float3 direction;
    float solidAngleRadius;  

    float3 color;
};

struct SSphere
{
    float3 position;
    float radius;

    float3 albedo;
};

struct SQuad
{
    void CalculateNormal ()
    {
        float3 e1 = b - a;
        float3 e2 = c - a;
        normal = Normalize(Cross(e1, e2));
    }

    float3  a, b, c, d;
    float3  albedo;

    // calculated!
    float3  normal;
};

struct SHitInfo
{
    float collisionTime = -1.0f;
    float3 normal;
    float3 albedo = { 0.0f, 0.0f, 0.0f };
};

//-------------------------------------------------------------------------------------------------------------------
inline bool RayIntersect (const float3& rayPos, const float3& rayDir, const SQuad& quad, SHitInfo& hitInfo)
{
    // This function adapted from "Real Time Collision Detection" 5.3.5 Intersecting Line Against Quadrilateral
    // IntersectLineQuad()
    float3 pa = quad.a - rayPos;
    float3 pb = quad.b - rayPos;
    float3 pc = quad.c - rayPos;
    // Determine which triangle to test against by testing against diagonal first
    float3 m = Cross(pc, rayDir);
    float3 r;
    float v = Dot(pa, m); // ScalarTriple(pq, pa, pc);
    if (v >= 0.0f) {
        // Test intersection against triangle abc
        float u = -Dot(pb, m); // ScalarTriple(pq, pc, pb);
        if (u < 0.0f) return false;
        float w = ScalarTriple(rayDir, pb, pa);
        if (w < 0.0f) return false;
        // Compute r, r = u*a + v*b + w*c, from barycentric coordinates (u, v, w)
        float denom = 1.0f / (u + v + w);
        u *= denom;
        v *= denom;
        w *= denom; // w = 1.0f - u - v;
        r = quad.a*u + quad.b*v + quad.c*w;
    }
    else {
        // Test intersection against triangle dac
        float3 pd = quad.d - rayPos;
        float u = Dot(pd, m); // ScalarTriple(pq, pd, pc);
        if (u < 0.0f) return false;
        float w = ScalarTriple(rayDir, pa, pd);
        if (w < 0.0f) return false;
        v = -v;
        // Compute r, r = u*a + v*d + w*c, from barycentric coordinates (u, v, w)
        float denom = 1.0f / (u + v + w);
        u *= denom;
        v *= denom;
        w *= denom; // w = 1.0f - u - v;
        r = quad.a*u + quad.d*v + quad.c*w;
    }

    // make sure normal is facing opposite of ray direction.
    // this is for if we are hitting the object from the inside / back side.
    float3 normal = quad.normal;
    if (Dot(quad.normal, rayDir) > 0.0f)
        normal = normal * -1.0f;

    // figure out the time t that we hit the plane (quad)
    float t;
    if (abs(rayDir[0]) > 0.0f)
        t = (r[0] - rayPos[0]) / rayDir[0];
    else if (abs(rayDir[1]) > 0.0f)
        t = (r[1] - rayPos[1]) / rayDir[1];
    else if (abs(rayDir[2]) > 0.0f)
        t = (r[2] - rayPos[2]) / rayDir[2];

    // only positive time hits allowed!
    if (t < 0.0f)
        return false;

    //enforce a max distance if we should
    if (hitInfo.collisionTime >= 0.0 && t > hitInfo.collisionTime)
        return false;

    hitInfo.collisionTime = t;
    hitInfo.albedo = quad.albedo;
    hitInfo.normal = normal;
    return true;
}

//-------------------------------------------------------------------------------------------------------------------
inline bool RayIntersect (const float3& rayPos, const float3& rayDir, const SSphere& sphere, SHitInfo& hitInfo)
{
    //get the vector from the center of this circle to where the ray begins.
    float3 m = rayPos - sphere.position;

    //get the dot product of the above vector and the ray's vector
    float b = Dot(m, rayDir);

    float c = Dot(m, m) - sphere.radius * sphere.radius;

    //exit if r's origin outside s (c > 0) and r pointing away from s (b > 0)
    if (c > 0.0 && b > 0.0)
        return false;

    //calculate discriminant
    float discr = b * b - c;

    //a negative discriminant corresponds to ray missing sphere
    if (discr <= 0.0)
        return false;

    //ray now found to intersect sphere, compute smallest t value of intersection
    float collisionTime = -b - sqrt(discr);

    //if t is negative, ray started inside sphere so clamp t to zero and remember that we hit from the inside
    if (collisionTime < 0.0)
        collisionTime = -b + sqrt(discr);

    //enforce a max distance if we should
    if (hitInfo.collisionTime >= 0.0 && collisionTime > hitInfo.collisionTime)
        return false;

    float3 normal = Normalize((rayPos + rayDir * collisionTime) - sphere.position);

    // make sure normal is facing opposite of ray direction.
    // this is for if we are hitting the object from the inside / back side.
    if (Dot(normal, rayDir) > 0.0f)
        normal = normal * -1.0f;

    hitInfo.collisionTime = collisionTime;
    hitInfo.normal = normal;
    hitInfo.albedo = sphere.albedo;
    return true;
}

//-------------------------------------------------------------------------------------------------------------------
static SQuad g_quads[] =
{
    { { -15.0f, 0.0f, 15.0f },{ 15.0f, 0.0f, 15.0f },{ 15.0f, 0.0f, -15.0f },{ -15.0f, 0.0f, -15.0f }, {1.0f, 1.0f, 1.0}},
};

static const SSphere g_spheres[] =
{
    {{0.0f, 2.0f, 5.0f}, 1.0f, {1.0f, 0.1f, 0.1f}},
    {{-2.0f, 2.3f, 7.0f}, 1.0f, {0.1f, 1.0f, 0.1f}},
    {{3.0f, 1.8f, 8.0f}, 1.0f, {0.1f, 0.1f, 1.0f}},
};

static const SDirectionalLight g_directionalLights[] =
{
    {{-0.3f, -1.0f, 0.0f}, 0.4f, {1.0f, 1.0f, 1.0f}},
};

static const SPositionalLight g_positionalLights[] =
{
#if 1
    { { -4.0f, 5.0f, 5.0f },1.0f,{ 0.0f, 0.0f, 0.0f } },
#else
    {{-4.0f, 5.0f, 5.0f},1.0f,{20.0f, 10.0f, 10.0f}},
    {{ 0.0f, 5.0f, 0.0f},1.0f,{10.0f, 20.0f, 10.0f}},
    {{ 4.0f, 5.0f, 5.0f},1.0f,{10.0f, 10.0f, 20.0f}},
#endif
};

static const float  g_cameraDistance = 2.0f;
static const float3 g_cameraPos = { 0.0f, 2.0f, 0.0f };
static const float3 g_cameraX = { 1.0f, 0.0f, 0.0f };
static const float3 g_cameraY = { 0.0f, 1.0f, 0.0f };
static const float3 g_cameraZ = { 0.0f, 0.0f, 1.0f };

static const float3 g_skyColor = { 135.0f / 255.0f, 206.0f / 255.0f, 235.0f / 255.0f };

SImageData<4> g_blueNoiseTexture;

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
    printf("Doing %s with %zu threads.\n", label, numThreads);
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
float sRGBToLinear (float value)
{
    if (value < 0.04045f)
        return value / 12.92f;
    else
        return std::powf(((value + 0.055f) / 1.055f), 2.4f);
}

float LinearTosRGB (float value)
{
    if (value < 0.0031308f)
        return value * 12.92f;
    else
        return std::powf(value, 1.0f / 2.4f) *  1.055f - 0.055f;
}

//-------------------------------------------------------------------------------------------------------------------

struct SGbufferPixel
{
    SHitInfo hitInfo;
    float u, v;
    float shadowMultipliersDirectional[sizeof(g_directionalLights) / sizeof(g_directionalLights[0])];
    float shadowMultipliersPositional[sizeof(g_positionalLights) / sizeof(g_positionalLights[0])];
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

template <int RADIUS, size_t CHANNELS>
void BoxBlur(const SImageData<CHANNELS>& src, SImageData<CHANNELS>& dest)
{
    static const size_t numPixelsInFilter = (1 + RADIUS * 2) * (1 + RADIUS * 2);

    std::array<float, numPixelsInFilter> kernel;

    for (float& f : kernel)
        f = 1.0f;

    Convolve<RADIUS, CHANNELS>(src, dest, kernel);
}

//-------------------------------------------------------------------------------------------------------------------

template <int RADIUS, size_t CHANNELS>
void TriangleBlur(const SImageData<CHANNELS>& src, SImageData<CHANNELS>& dest)
{
    static const size_t numPixelsOneSide = (1 + RADIUS * 2);
    static const size_t numPixelsInFilter = numPixelsOneSide * numPixelsOneSide;

    std::array<float, numPixelsInFilter> kernel;

    for (size_t i = 0; i < numPixelsInFilter; ++i)
    {
        size_t x = i % numPixelsOneSide;
        size_t y = i / numPixelsOneSide;

        float triangleX = (x <= RADIUS) ? float(x + 1) : float(numPixelsOneSide - x);
        float triangleY = (y <= RADIUS) ? float(y + 1) : float(numPixelsOneSide - y);

        kernel[i] = triangleX * triangleY;
    }

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
void BoxBlurShadowMultipliers (const std::vector<SGbufferPixel>& src, std::vector<SGbufferPixel>& dest, size_t width, size_t height)
{
    static const size_t numPixelsInFilter = (1 + RADIUS * 2) * (1 + RADIUS * 2);

    dest = src;

    SGbufferPixel* outPixel = &dest[0];

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            for (size_t sampleIndex = 0; sampleIndex < SAMPLES_PER_PIXEL; ++sampleIndex)
            {
                for (float& f : outPixel->shadowMultipliersDirectional)
                    f = 0.0f;

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
                        
                        for (size_t index = 0; index < sizeof(g_directionalLights) / sizeof(g_directionalLights[0]); ++index)
                            outPixel->shadowMultipliersDirectional[index] += inPixel.shadowMultipliersDirectional[index];

                        for (size_t index = 0; index < sizeof(g_positionalLights) / sizeof(g_positionalLights[0]); ++index)
                            outPixel->shadowMultipliersPositional[index] += inPixel.shadowMultipliersPositional[index];
                    }
                }

                for (float& f : outPixel->shadowMultipliersDirectional)
                    f /= float(numPixelsInFilter);

                for (float& f : outPixel->shadowMultipliersPositional)
                    f /= float(numPixelsInFilter);

                ++outPixel;
            }
        }
    }
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
    Hash,
    BlueNoiseGR
};

//-------------------------------------------------------------------------------------------------------------------

template <size_t SHADOW_RAY_COUNT, size_t SHADOW_RAY_COUNT_GRID_SIZE, RayPattern RAY_PATTERN, RNGSource RNG_SOURCE>
SGbufferPixel PixelFunctionGBuffer(float u, float v, size_t pixelX, size_t pixelY, std::mt19937& rng)
{
    // TODO: make all RNG_SOURCE types work!

    // raytrace to find primary ray intersection
    float3 rayPos = g_cameraPos;
    float3 rayDir = Normalize(g_cameraX * u + g_cameraY * v + g_cameraZ * g_cameraDistance);
    SGbufferPixel ret;
    for (size_t i = 0; i < sizeof(g_spheres) / sizeof(g_spheres[0]); ++i)
        RayIntersect(rayPos, rayDir, g_spheres[i], ret.hitInfo);
    for (size_t i = 0; i < sizeof(g_quads) / sizeof(g_quads[0]); ++i)
        RayIntersect(rayPos, rayDir, g_quads[i], ret.hitInfo);

    // apply directional lighting
    float3 pixelPos = rayPos + rayDir * ret.hitInfo.collisionTime;
    float3 shadowPos = pixelPos + ret.hitInfo.normal * c_rayEpsilon;
    for (size_t lightIndex = 0; lightIndex < sizeof(g_directionalLights) / sizeof(g_directionalLights[0]); ++lightIndex)
    {
        float3 lightDir = Normalize(g_directionalLights[lightIndex].direction * -1.0f);

        ret.shadowMultipliersDirectional[lightIndex] = 1.0f;

        for (size_t sampleIndex = 0; sampleIndex <= SHADOW_RAY_COUNT; ++sampleIndex)
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
                case RNGSource::Hash:
                {
                    break;
                }
                case RNGSource::BlueNoiseGR:
                {
                    size_t texX = pixelX % g_blueNoiseTexture.m_width;
                    size_t texY = pixelY % g_blueNoiseTexture.m_height;

                    float* blueNoise = g_blueNoiseTexture.GetPixel(texX, texY);
                    rngX = std::fmodf(blueNoise[0] + c_goldenRatioConjugate * float(sampleIndex), 1.0f);
                    rngY = std::fmodf(blueNoise[1] + c_goldenRatioConjugate * float(sampleIndex), 1.0f);

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
                    if (SHADOW_RAY_COUNT_GRID_SIZE == 1)
                    {
                        sampleX = 0.5f;
                        sampleY = 0.5f;
                    }
                    else
                    {
                        sampleX = float(sampleIndex % SHADOW_RAY_COUNT_GRID_SIZE) / float(SHADOW_RAY_COUNT_GRID_SIZE - 1);
                        sampleY = float(sampleIndex / SHADOW_RAY_COUNT_GRID_SIZE) / float(SHADOW_RAY_COUNT_GRID_SIZE - 1);
                    }

                    // TODO: should we add a half cell for grid? i think so!

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

            //float3 randomDir = RandomVectorInsideCone(lightDir, g_directionalLights[lightIndex].solidAngleRadius, sampleX, sampleY);
            float3 randomDir = RandomVectorTowardsLight(lightDir, g_directionalLights[lightIndex].solidAngleRadius, sampleX, sampleY);

            SHitInfo shadowHitInfo;

            bool intersectionFound = false;
            for (size_t i = 0; i < sizeof(g_spheres) / sizeof(g_spheres[0]) && !intersectionFound; ++i)
                intersectionFound |= RayIntersect(shadowPos, randomDir, g_spheres[i], shadowHitInfo);
            for (size_t i = 0; i < sizeof(g_quads) / sizeof(g_quads[0]) && !intersectionFound; ++i)
                intersectionFound |= RayIntersect(shadowPos, randomDir, g_quads[i], shadowHitInfo);

            if (intersectionFound)
                ret.shadowMultipliersDirectional[lightIndex] = Lerp(ret.shadowMultipliersDirectional[lightIndex], 0.0f, 1.0f / float(1+sampleIndex));
            else
                ret.shadowMultipliersDirectional[lightIndex] = Lerp(ret.shadowMultipliersDirectional[lightIndex], 1.0f, 1.0f / float(1+sampleIndex));
        }
    }

    // TODO: positional lights. Maybe make a common function or something

    return ret;

}

float3 PixelFunctionShade (const SGbufferPixel& gbufferData)
{
    float3 ret = { 0.0f, 0.0f, 0.0f };

    if (gbufferData.hitInfo.collisionTime <= 0.0f)
        return g_skyColor;

    for (size_t lightIndex = 0; lightIndex < sizeof(g_directionalLights) / sizeof(g_directionalLights[0]); ++lightIndex)
    {
        float3 lightDir = Normalize(g_directionalLights[lightIndex].direction * -1.0f);

        float NdotL = Dot(lightDir, gbufferData.hitInfo.normal);
        if (NdotL > 0.0f)
            ret = ret + g_directionalLights[lightIndex].color * gbufferData.hitInfo.albedo * NdotL * gbufferData.shadowMultipliersDirectional[lightIndex];
    }

    // TODO: positional lights too!

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

template <size_t SHADOW_RAY_COUNT, size_t SHADOW_RAY_COUNT_GRID_SIZE, RayPattern RAY_PATTERN, RNGSource RNG_SOURCE>
void GeneratePixels(const char* task, const char* baseFileName, bool doPreProcessing, bool doPostProcessing)
{
    char fileName[256];

    SImageData<3> output(IMAGE_WIDTH(), IMAGE_HEIGHT());
    const size_t numPixels = output.m_width * output.m_height;

    float aspectRatio = float(output.m_width) / float(output.m_height);

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

                        pixel = PixelFunctionGBuffer<SHADOW_RAY_COUNT, SHADOW_RAY_COUNT_GRID_SIZE, RAY_PATTERN, RNG_SOURCE>(pixel.u, pixel.v, pixelIndex % output.m_width, pixelIndex / output.m_width, rng);
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

    // do gbuffer processing if we should
    if (doPreProcessing)
    {
        // for pre-processing the gbuffer, we want it to be as if we only took 1 shadow sample per pixel before filtering.
        std::vector<SGbufferPixel> gbuffer1ShadowSample = gbuffer;
        for (SGbufferPixel& pixel : gbuffer1ShadowSample)
        {
            for (size_t lightIndex = 1; lightIndex < sizeof(g_directionalLights) / sizeof(g_directionalLights[0]); ++lightIndex)
                pixel.shadowMultipliersDirectional[lightIndex] = pixel.shadowMultipliersDirectional[0];

            for (size_t lightIndex = 1; lightIndex < sizeof(g_positionalLights) / sizeof(g_positionalLights[0]); ++lightIndex)
                pixel.shadowMultipliersPositional[lightIndex] = pixel.shadowMultipliersPositional[0];
        }

        std::vector<SGbufferPixel> gbufferFiltered;

        BoxBlurShadowMultipliers<1, STRATIFIED_SAMPLE_COUNT()>(gbuffer1ShadowSample, gbufferFiltered, output.m_width, output.m_height);
        ShadePixels<STRATIFIED_SAMPLE_COUNT()>(gbufferFiltered, output);
        sprintf_s(fileName, baseFileName, "_prebox3");
        output.Save(fileName);

        BoxBlurShadowMultipliers<2, STRATIFIED_SAMPLE_COUNT()>(gbuffer1ShadowSample, gbufferFiltered, output.m_width, output.m_height);
        ShadePixels<STRATIFIED_SAMPLE_COUNT()>(gbufferFiltered, output);
        sprintf_s(fileName, baseFileName, "_prebox5");
        output.Save(fileName);

        BoxBlurShadowMultipliers<3, STRATIFIED_SAMPLE_COUNT()>(gbuffer1ShadowSample, gbufferFiltered, output.m_width, output.m_height);
        ShadePixels<STRATIFIED_SAMPLE_COUNT()>(gbufferFiltered, output);
        sprintf_s(fileName, baseFileName, "_prebox7");
        output.Save(fileName);

        BoxBlurShadowMultipliers<7, STRATIFIED_SAMPLE_COUNT()>(gbuffer1ShadowSample, gbufferFiltered, output.m_width, output.m_height);
        ShadePixels<STRATIFIED_SAMPLE_COUNT()>(gbufferFiltered, output);
        sprintf_s(fileName, baseFileName, "_prebox15");
        output.Save(fileName);
    }

    // make and save the unprocessed image
    {
        ShadePixels<STRATIFIED_SAMPLE_COUNT()>(gbuffer, output);
        sprintf_s(fileName, baseFileName, "");
        output.Save(fileName);
    }

    // do image post processing if we should
    if (doPostProcessing)
    {
        SImageData<3> filtered;

        BoxBlur<1>(output, filtered);
        sprintf_s(fileName, baseFileName, "_box3");
        filtered.Save(fileName);

        BoxBlur<2>(output, filtered);
        sprintf_s(fileName, baseFileName, "_box5");
        filtered.Save(fileName);

        BoxBlur<7>(output, filtered);
        sprintf_s(fileName, baseFileName, "_box15");
        filtered.Save(fileName);

        TriangleBlur<1>(output, filtered);
        sprintf_s(fileName, baseFileName, "_triangle3");
        filtered.Save(fileName);

        TriangleBlur<2>(output, filtered);
        sprintf_s(fileName, baseFileName, "_triangle5");
        filtered.Save(fileName);

        TriangleBlur<7>(output, filtered);
        sprintf_s(fileName, baseFileName, "_triangle15");
        filtered.Save(fileName);

        MedianFilter<1>(output, filtered);
        sprintf_s(fileName, baseFileName, "_median3");
        filtered.Save(fileName);

        MedianFilter<2>(output, filtered);
        sprintf_s(fileName, baseFileName, "_median5");
        filtered.Save(fileName);

        MedianFilter<7>(output, filtered);
        sprintf_s(fileName, baseFileName, "_median15");
        filtered.Save(fileName);
    }
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

    // make images

    GeneratePixels<1, 3, RayPattern::None, RNGSource::WhiteNoise>("White 1", "out/white_1%s.png", true, true);
    GeneratePixels<2, 3, RayPattern::None, RNGSource::WhiteNoise>("White 2", "out/white_2%s.png", true, false);
    GeneratePixels<4, 3, RayPattern::None, RNGSource::WhiteNoise>("White 4", "out/white_4%s.png", true, false);
    GeneratePixels<8, 3, RayPattern::None, RNGSource::WhiteNoise>("White 8", "out/white_8%s.png", true, false);
    GeneratePixels<64, 3, RayPattern::None, RNGSource::WhiteNoise>("White 64", "out/white_64%s.png", false, false);
    GeneratePixels<256, 3, RayPattern::None, RNGSource::WhiteNoise>("White 256", "out/white_256%s.png", false, false);

    GeneratePixels<8, 3, RayPattern::Grid, RNGSource::WhiteNoise>("Grid 8", "out/grid_8%s.png", true, false);

    GeneratePixels<1, 3, RayPattern::None, RNGSource::BlueNoiseGR>("Blue 1", "out/blue_1%s.png", true, false);
    GeneratePixels<2, 3, RayPattern::None, RNGSource::BlueNoiseGR>("Blue 2", "out/blue_2%s.png", true, false);
    GeneratePixels<4, 3, RayPattern::None, RNGSource::BlueNoiseGR>("Blue 4", "out/blue_4%s.png", true, false);
    GeneratePixels<8, 3, RayPattern::None, RNGSource::BlueNoiseGR>("Blue 8", "out/blue_8%s.png", true, false);
    GeneratePixels<64, 3, RayPattern::None, RNGSource::BlueNoiseGR>("Blue 64", "out/blue_64%s.png", false, false);
    GeneratePixels<256, 3, RayPattern::None, RNGSource::BlueNoiseGR>("Blue 256", "out/blue_256%s.png", false, false);

    GeneratePixels<8, 3, RayPattern::Stratified, RNGSource::WhiteNoise>("Stratified White 8", "out/stratified_white_8%s.png", true, false);
    GeneratePixels<8, 3, RayPattern::Stratified, RNGSource::BlueNoiseGR>("Stratified Blue 8", "out/stratified_blue_8%s.png", true, false);

    GeneratePixels<4, 2, RayPattern::Stratified, RNGSource::BlueNoiseGR>("Stratified Blue 4", "out/stratified4_blue_4%s.png", true, false);

    GeneratePixels<1, 1, RayPattern::Grid, RNGSource::WhiteNoise>("Hard", "out/hard_1%s.png", false, false);

    system("pause");
    return 0;
}

/*

TODO:

! verify your quad code with what you made before
* compare quad with disc. replace if appropriate

? does pcf make sense at all in this context?

* do a blur where you blur NxN sections instead of each pixel reading every other pixel. This simulates something more quickly / easily done in a ray dispatch shader
* try doing a blur where every 2x2 block is averaged. (a gbuffer blur! maybe depth aware)

* make shading pixels a function, so easier to re-use

? organize into multiple files?

* larger box filter? Especially on pre box
 * also try larger median filter
* stuff is pretty verbose right now

* a better scene with more interesting geo and lights

* Permutation options...
+ 1) Number of rays 
- 2) Random number source: white noise, hash without sine, blue noise + golden ratio
- 3) Filter in pre or in post
- 4) Filter type: median, box, depth aware box
- 5) filter size: 3, 5?
- 6) Sample target: no target (path tracing?), disc, square. Cone or no?
- 7) Multi sample (re-use shadow info from first sample though IMO). 4x SSAA
- 8) Low variance early out. Dont forget the coprime number (5) for shuffling!

* try gaussian blur too?

* try fitting the shadow data to a function? piecewise least squares?

* point light
* directional light
* IBL?

? combine directional and positional lights into one list?

* ray marched shadows (single ray that keeps track of closest point)
* maybe also that one implementation where it's 1 ray per pixel but then blurred data

* animated & make an animated gif of results? (gif is low quality though... maybe ffmpeg?)

* todos

? ambient lighting? or multibounce? (approaching path tracing then though...)

* hard shadows
* "path traced shadows"
* compare vs path traced and make sure it converges with higher ray counts?
* low variance early out?
* try denoising?
* i think maybe shooting rays at a disc is the right move, and that ray in cone is not correct.

? red/blue 3d? or nah...

? how does stratified sampling fit in for AA?

* try to find good results

Demos...
1) Hard shadows
2) stratified sampling
3) stratified sampling with jitter
4) cone

BLOG:
* Using gamma 2.2
* box, triangle, gaussian are separable which is faster
? is median seprable?

*/