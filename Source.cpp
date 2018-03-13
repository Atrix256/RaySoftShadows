#include <atomic>
#include <thread>
#include <vector>
#include <array>
#include <stdint.h>
#include <stdlib.h>

// stb_image is an amazing header only image library (aka no linking, just include the headers!).  http://nothings.org/stb
#pragma warning( disable : 4996 ) 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#pragma warning( default : 4996 ) 

typedef uint8_t uint8;
typedef std::array<float, 3> float3;

float LinearTosRGB(float value);

//-------------------------------------------------------------------------------------------------------------------

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

    float albedo[3];
};

struct SPlane
{
    float3 N;
    float D;

    float3 albedo;
};

//-------------------------------------------------------------------------------------------------------------------

static const SPlane g_planes[] =
{
    {{0.0f, 1.0f, 0.0f},0.0f, {0.0f, 1.0f, 0.0f}},
};

static const SSphere g_spheres[] =
{
    {{0.0f, 5.0f, 5.0f}, 1.0f, {1.0f, 0.0f, 0.0f}},
};

static const SDirectionalLight g_directionalLights[] =
{
    {{0.0f, 1.0f, 0.0f}, 1.0f, {10.0f, 5.0f, 5.0f}},
};

static const SPositionalLight g_positionalLights[] =
{
    {{3.0f, 5.0f, 3.0f},1.0f,{5.0f, 5.0f, 10.0f}},
};

static const float g_cameraDistance = 2.0f;
static const float3 g_cameraPos = { 0.0f, 5.0f, 0.0f };
static const float3 g_cameraX = { 1.0f, 0.0f, 0.0f };
static const float3 g_cameraY = { 0.0f, 1.0f, 0.0f };
static const float3 g_cameraZ = { 0.0f, 0.0f, 1.0f };

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
    if (value < 0.0031308f)
        return value * 12.92f;
    else
        return std::powf(value, 1.0f / 2.4f) *  1.055f - 0.055f;
}

//-------------------------------------------------------------------------------------------------------------------
void PixelFunction (float u, float v, float3& pixel)
{
    float3 rayPos = g_cameraPos;

    float3 rayTarget =
        g_cameraPos +
        g_cameraX * (u * 2.0f - 1.0f) +
        g_cameraY * (v * 2.0f - 1.0f) +
        g_cameraZ * g_cameraDistance;

    float3 rayDir = rayTarget - rayPos;
    // TODO: normalize!
    // TODO: raytrace!

    pixel[0] = u;
    pixel[1] = v;
    pixel[2] = 0.0f;
}

//-------------------------------------------------------------------------------------------------------------------
template <typename LAMBDA>
void GeneratePixels(const char* task, const char* fileName, LAMBDA& lambda)
{
    SImageData output(640, 480);
    const size_t numPixels = output.m_width * output.m_height;

    // calculate image
    RunMultiThreaded(task,
        [&] (std::atomic<size_t>& atomicPixelIndex)
        {
            size_t pixelIndex = atomicPixelIndex.fetch_add(1);
            bool reportProgress = (pixelIndex == 0);
            int oldPercent = -1;
            while (pixelIndex < numPixels)
            {
                // calculate uv coordinate of pixel
                float u = float(pixelIndex % output.m_width) / float(output.m_width - 1);
                float v = float(pixelIndex / output.m_width) / float(output.m_width - 1);

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

* animated & make an animated gif of results? (gif is low quality though... maybe ffmpeg?)

? do we need reinhard tone mapping?


* ambient light settings? or at least a "miss" ray color?
* projection matrix settings

*/