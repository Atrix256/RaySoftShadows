#pragma once

typedef std::array<float, 3> float3;
typedef std::array<std::array<float, 3>, 3> float3x3;

inline float3 operator* (const float3& a, const float3& b)
{
    float3 ret;
    ret[0] = a[0] * b[0];
    ret[1] = a[1] * b[1];
    ret[2] = a[2] * b[2];
    return ret;
}

inline float3 operator+ (const float3& a, const float3& b)
{
    float3 ret;
    ret[0] = a[0] + b[0];
    ret[1] = a[1] + b[1];
    ret[2] = a[2] + b[2];
    return ret;
}

inline float3 operator- (const float3& a, const float3& b)
{
    float3 ret;
    ret[0] = a[0] - b[0];
    ret[1] = a[1] - b[1];
    ret[2] = a[2] - b[2];
    return ret;
}

inline float3 operator* (const float3& a, float b)
{
    float3 ret;
    ret[0] = a[0] * b;
    ret[1] = a[1] * b;
    ret[2] = a[2] * b;
    return ret;
}

inline float3 operator+ (const float3& a, float b)
{
    float3 ret;
    ret[0] = a[0] + b;
    ret[1] = a[1] + b;
    ret[2] = a[2] + b;
    return ret;
}

inline float3 operator/ (const float3& a, float b)
{
    return a * (1.0f / b);
}

inline float3x3 operator* (const float3x3& a, float b)
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

inline float3 operator* (const float3x3& a, const float3& b)
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

inline float3x3 operator+ (const float3x3& a, const float3x3& b)
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

inline float Length(const float3& a)
{
    return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

inline float3 Normalize(const float3& a)
{
    return a / Length(a);
}

inline float Dot(const float3& a, const float3& b)
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
