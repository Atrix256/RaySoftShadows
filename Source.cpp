#include <atomic>
#include <thread>
#include <vector>
#include <array>
#include <stdint.h>
#include <stdlib.h>

#define IMAGE_WIDTH  1024
#define IMAGE_HEIGHT 768

// stb_image is an amazing header only image library (aka no linking, just include the headers!).  http://nothings.org/stb
#pragma warning( disable : 4996 ) 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#pragma warning( default : 4996 ) 

typedef uint8_t uint8;
typedef std::array<float, 3> float3;

float LinearTosRGB(float value);

static const float c_rayEpsilon = 0.01f; // value used to push the ray a little bit away from surfaces before doing shadow rays

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

float3 Normalize(const float3& a)
{
    float len = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    return a / len;
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

//-------------------------------------------------------------------------------------------------------------------
struct SImageData
{
    SImageData (size_t width, size_t height)
        : m_width(width)
        , m_height(height)
    {
        m_pixels.resize(m_width*m_height * 3);
    }

    bool Save (const char* fileName)
    {
        // convert from linear f32 to sRGB u8
        std::vector<uint8> pixelsU8;
        pixelsU8.resize(m_pixels.size());
        for (size_t i = 0; i < m_pixels.size(); ++i)
            pixelsU8[i] = uint8(LinearTosRGB(m_pixels[i])*255.0f);

        // save the image
        return (stbi_write_png(fileName, (int)m_width, (int)m_height, 3, &pixelsU8[0], (int)Pitch()) == 1);
    }

    size_t Pitch () const { return m_width * 3; }

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
    float3 direction;  // could be latitude and longitude to save a float
    float solidAngle;    // TODO: may be bad naming. This is the angle that the rays are allowed to deviate in.

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
    float3 albedo;
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
    { { -15.0f, 0.0f, 15.0f },{ 15.0f, 0.0f, 15.0f },{ 15.0f, 0.0f, -15.0f },{ -15.0f, 0.0f, -15.0f }, {0.0f, 1.0f, 0.0}},
};

static const SSphere g_spheres[] =
{
    {{0.0f, 2.0f, 5.0f}, 1.0f, {1.0f, 0.1f, 0.1f}},
    {{-2.0f, 2.3f, 7.0f}, 1.0f, {0.1f, 1.0f, 0.1f}},
    {{3.0f, 1.8f, 8.0f}, 1.0f, {0.1f, 0.1f, 1.0f}},
};

static const SDirectionalLight g_directionalLights[] =
{
    {{-0.3f, -1.0f, 0.0f}, 1.0f, {1.0f, 1.0f, 1.0f}},
};

static const SPositionalLight g_positionalLights[] =
{
    {{3.0f, 5.0f, 3.0f},1.0f,{5.0f, 5.0f, 10.0f}},
};

static const float  g_cameraDistance = 2.0f;
static const float3 g_cameraPos = { 0.0f, 2.0f, 0.0f };
static const float3 g_cameraX = { 1.0f, 0.0f, 0.0f };
static const float3 g_cameraY = { 0.0f, 1.0f, 0.0f };
static const float3 g_cameraZ = { 0.0f, 0.0f, 1.0f };

static const float3 g_skyColor = { 135.0f / 255.0f, 206.0f / 255.0f, 235.0f / 255.0f };

//-------------------------------------------------------------------------------------------------------------------
template <typename L>
void RunMultiThreaded (const char* label, const L& lambda)
{
    std::atomic<size_t> atomicCounter(0);
    auto wrapper = [&] ()
    {
        lambda(atomicCounter);
    };

    size_t numThreads = std::thread::hardware_concurrency();
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
float LinearTosRGB (float value)
{
    if (value > 1.0f)
        return 1.0f;
    else if (value < 0.0f)
        return 0.0f;

    if (value < 0.0031308f)
        return value * 12.92f;
    else
        return std::powf(value, 1.0f / 2.4f) *  1.055f - 0.055f;
}

//-------------------------------------------------------------------------------------------------------------------
void PixelFunction (float u, float v, float3& pixel)
{
    // init pixel to black
    pixel = { 0.0f, 0.0f, 0.0f };

    // calculate the ray
    float3 rayPos = g_cameraPos;

    float3 rayDir =
        g_cameraX * u +
        g_cameraY * v +
        g_cameraZ * g_cameraDistance;

    rayDir = Normalize(rayDir);

    // raytrace
    SHitInfo hitInfo;
    for (size_t i = 0; i < sizeof(g_spheres) / sizeof(g_spheres[0]); ++i)
        RayIntersect(rayPos, rayDir, g_spheres[i], hitInfo);
    for (size_t i = 0; i < sizeof(g_quads) / sizeof(g_quads[0]); ++i)
        RayIntersect(rayPos, rayDir, g_quads[i], hitInfo);

    // if we missed, return sky color
    if (hitInfo.collisionTime <= 0.0f)
    {
        pixel = g_skyColor;
        return;
    }

    // apply lighting
    float3 pixelPos = rayPos + rayDir * hitInfo.collisionTime;
    for (size_t lightIndex = 0; lightIndex < sizeof(g_directionalLights) / sizeof(g_directionalLights[0]); ++lightIndex)
    {
        float3 lightDir = Normalize(g_directionalLights[lightIndex].direction * -1.0f);
        float3 shadowPos = pixelPos + hitInfo.normal * c_rayEpsilon;

        // TODO: exit geo loops on first hit!
        SHitInfo shadowHitInfo;
        for (size_t i = 0; i < sizeof(g_spheres) / sizeof(g_spheres[0]); ++i)
            RayIntersect(shadowPos, lightDir, g_spheres[i], shadowHitInfo);
        for (size_t i = 0; i < sizeof(g_quads) / sizeof(g_quads[0]); ++i)
            RayIntersect(shadowPos, lightDir, g_quads[i], shadowHitInfo);

        if (shadowHitInfo.collisionTime > 0.0f)
            continue;

        pixel = pixel + g_directionalLights[lightIndex].color * hitInfo.albedo;
    }

    /*
    pixel[0] = u;
    pixel[1] = v;
    pixel[2] = 0.0f;

    pixel = rayDir * 0.5f + 0.5f;

    pixel = hitInfo.albedo;
    */
}

//-------------------------------------------------------------------------------------------------------------------
template <typename LAMBDA>
void GeneratePixels(const char* task, const char* fileName, LAMBDA& lambda)
{
    SImageData output(IMAGE_WIDTH, IMAGE_HEIGHT);
    const size_t numPixels = output.m_width * output.m_height;

    float aspectRatio = float(output.m_width) / float(output.m_height);

    // calculate image
    RunMultiThreaded(task,
        [&] (std::atomic<size_t>& atomicPixelIndex)
        {
            size_t pixelIndex = atomicPixelIndex.fetch_add(1);
            bool reportProgress = (pixelIndex == 0);
            int oldPercent = -1;
            while (pixelIndex < numPixels)
            {
                // calculate uv coordinate of pixel in [-1,1], also correcting for aspect ratio and flipping the vertical axis
                float u = ((float(pixelIndex % output.m_width) / float(output.m_width - 1)) * 2.0f - 1.0f) * aspectRatio;
                float v = ((float(pixelIndex / output.m_width) / float(output.m_height - 1)) * 2.0f - 1.0f) * -1.0f;

                // shade the pixel
                lambda(u, v, *(float3*)&output.m_pixels[pixelIndex * 3]);

                // report progress
                if (reportProgress)
                {
                    int newPercent = (int)(100.0f * float(pixelIndex) / float(numPixels));
                    if (oldPercent != newPercent)
                        printf("\rProgress: %i%%", newPercent);
                }

                pixelIndex = atomicPixelIndex.fetch_add(1);
            }

            if (reportProgress)
                printf("\rProgress: 100%%\n");
        }
    );
    output.Save(fileName);
}

//-------------------------------------------------------------------------------------------------------------------
int main (int argc, char** argv)
{
    for (size_t i = 0; i < sizeof(g_quads) / sizeof(g_quads[0]); ++i)
        g_quads[i].CalculateNormal();

    GeneratePixels("Soft Shadows", "out.png", PixelFunction);

    system("pause");
    return 0;
}

/*

TODO:
* point light
* directional light

* analytical (many rays) vs ray marched (single ray)

* white noise sampling vs blue noise sampling

* handle aspect ratio, from center of screen

* animated & make an animated gif of results? (gif is low quality though... maybe ffmpeg?)

? do we need reinhard tone mapping?

? multi sample per pixel?
? should we have a thread do a row of pixels instead of a single pixel at a time? i think so.

* ambient light settings? or at least a "miss" ray color?
* projection matrix settings

*/