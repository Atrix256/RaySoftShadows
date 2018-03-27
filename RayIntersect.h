#pragma once

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
    float3 emissive = { 0.0f, 0.0f, 0.0f };
};

struct SPositionalLight
{
    float3 position;
    float radius;
    float3 color;
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
