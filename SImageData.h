#pragma once

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

    bool Save (const char* fileName, bool toSRGB = true, bool applyToneMapping = true)
    {
        // convert from linear f32 to sRGB u8
        std::vector<uint8> pixelsU8;
        pixelsU8.resize(m_pixels.size());
        for (size_t i = 0; i < m_pixels.size(); ++i)
        {
            // convert from sRGN to linear if we should
            float value = m_pixels[i];
            if (toSRGB)
                value = LinearTosRGB(value);

            // apply tone mapping if we should
            if (applyToneMapping)
                value = HDRToSDR(value);

            // clamp and convert to uint8
            pixelsU8[i] = uint8(clamp(value, 0.0f, 1.0f) * 255.0f);
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

//-------------------------------------------------------------------------------------------------------------------
inline float sRGBToLinear(float value)
{
    if (value < 0.04045f)
        return value / 12.92f;
    else
        return std::powf(((value + 0.055f) / 1.055f), 2.4f);
}

inline float LinearTosRGB(float value)
{
    if (value < 0.0031308f)
        return value * 12.92f;
    else
        return std::powf(value, 1.0f / 2.4f) *  1.055f - 0.055f;
}

//-------------------------------------------------------------------------------------------------------------------
inline float HDRToSDR(float input)
{
    // Reinhard tone mapping with a hard coded exposure adjustment
    input = input * 2.0f; 
    return input / (1.0f + input);
}