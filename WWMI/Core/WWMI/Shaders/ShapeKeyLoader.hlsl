
Texture1D<float4> IniParams : register(t120);

// cbuffer cb0 : register(b0)
// {
//     uint4 cb0[66];
// }

Buffer<float4> t1 : register(t1);

Buffer<uint4> t0 : register(t0);


RWBuffer<uint> u0 : register(u0);
RWBuffer<int> u1 : register(u1);
RWBuffer<uint4> cb0 : register(u6);
// RWBuffer<float4> DebugRW : register(u7);

groupshared struct { float val; } g1[128];
groupshared struct { float val; } g0[128];

#define cmp -


[numthreads(2,32,1)]
void main(uint3 vThreadID : SV_DispatchThreadID, uint3 vThreadIDInGroup : SV_GroupThreadID, uint vThreadIDInGroupFlattened : SV_GroupIndex)
{
    // DebugRW[vThreadID.y] = float4(vThreadID.x, vThreadID.y, vThreadIDInGroup.x, vThreadIDInGroupFlattened.x);

    const float4 icb[] = { 
        { 1.000000, 0, 0, 0},
        { 0, 1.000000, 0, 0},
        { 0, 0, 1.000000, 0},
        { 0, 0, 0, 1.000000} 
    };

    float4 r0,r1,r2,r3;


    r0.x = cmp((uint)vThreadIDInGroupFlattened.x < 32);

    if (r0.x != 0) {
        r0.x = (uint)vThreadIDInGroupFlattened.x << 2;
        r0.y = 0;

        
        while (true) {
            r0.w = cmp((int)r0.y >= 4);
            if (r0.w != 0) {
                break;
            }

            
            r0.w = (int)r0.y + (int)r0.x;


            r1.x = vThreadIDInGroupFlattened.x;

            // Get Shape Key Value from CB
            r1.y = dot(asfloat(cb0[r1.x+32].xyzw), icb[r0.y+0].xyzw);

            g0[r0.w].val = r1.y;

            r1.y = -(int)r0.y;
            r2.xyz = cmp((uint3)r0.yyy < uint3(1,2,3));

            r3.y = r2.y ? r1.y : 0;
            r0.yz = (int2)r0.yy + int2(1,-3);
            r3.z = r2.y ? 0 : r0.z;
            r3.w = cmp((int)r2.z == 0);
            r3.x = r2.x;

            uint idx = r1.x;

            // Get Shape Key Offset from CB
            r1.xyzw = r3.xyzw ? cb0[r1.x+0].xyzw : 0;

            r1.xy = (int2)r1.yw | (int2)r1.xz;
            r0.z = (int)r1.y | (int)r1.x;

            g1[r0.w].val = r0.z;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Calculate Shape Key Id based on Thread Id
    r0.x = g1[64].val;

    r0.x = cmp((uint)vThreadID.y < (uint)r0.x);
    r0.xy = r0.xx ? float2(0,32) : float2(64,96);
    r0.y = g1[r0.y].val;
    r0.y = cmp((uint)vThreadID.y < (uint)r0.y);
    r0.y = r0.y ? 0 : 32;
    r0.x = (int)r0.x + (int)r0.y;
    r0.y = (int)r0.x + 16;
    r0.y = g1[r0.y].val;
    r0.y = cmp((uint)vThreadID.y < (uint)r0.y);
    r0.y = r0.y ? 0 : 16;
    r0.x = (int)r0.x + (int)r0.y;
    r0.y = (int)r0.x + 8;
    r0.y = g1[r0.y].val;
    r0.y = cmp((uint)vThreadID.y < (uint)r0.y);
    r0.y = r0.y ? 0 : 8;
    r0.x = (int)r0.x + (int)r0.y;
    r0.y = (int)r0.x + 4;
    r0.y = g1[r0.y].val;
    r0.y = cmp((uint)vThreadID.y < (uint)r0.y);
    r0.y = r0.y ? 0 : 4;
    r0.x = (int)r0.x + (int)r0.y;
    r0.y = (int)r0.x + 2;
    r0.y = g1[r0.y].val;
    r0.y = cmp((uint)vThreadID.y < (uint)r0.y);
    r0.y = r0.y ? 0 : 2;
    r0.x = (int)r0.x + (int)r0.y;
    r0.y = (int)r0.x + 1;
    r0.y = g1[r0.y].val;
    r0.y = cmp((uint)vThreadID.y < (uint)r0.y);
    r0.y = r0.y ? 0 : 1;
    r0.x = (int)r0.x + (int)r0.y;

    r0.y = asfloat(cb0[65].y);
    r0.z = (int)r0.x + 1;
    r0.z = min((int)r0.z, 127);

    // uint sk_id = r0.z;
    // uint sk_val_id = r0.x;

    // Get Shape Key Offset
    r0.z = g1[r0.z].val;
    r1.y = (float)r0.z;

    r1.x = 0;
    r0.yz = r1.xy + r0.yy;
    r0.yz = (uint2)r0.yz;

    // Get Shape Key Value
    r0.x = g0[r0.x].val;
    
    r0.y = (int)r0.y + (int)vThreadID.y;
    r0.z = cmp((uint)r0.y < (uint)r0.z);
    r0.w = cmp(r0.x != 0.000000);
    r0.z = r0.w ? r0.z : 0;

    // DebugRW[sk_id] = float4(sk_id, g1[sk_id].val, g0[sk_val_id].val, r0.x);

    if (r0.z != 0) {

        r0.z = t0.Load(r0.y).x;
        r0.w = 0;

        while (true) {
            r1.x = cmp((uint)r0.w >= 3);
            if (r1.x != 0) {
                break;
            }
            r1.x = mad(3, (int)vThreadIDInGroup.x, (int)r0.w);

            r1.y = mad(6, (int)r0.y, (int)r1.x);
            
            r1.y = t1.Load(r1.y).x;

            r1.y = r1.y * r0.x;

            r1.z = dot(asfloat(cb0[64].xyzw), icb[r0.w+0].xyzw);

            r1.z = vThreadIDInGroup.x ? cb0[64].w : r1.z;

            r1.y = r1.y * r1.z;
            r1.y = round(r1.y);
            r1.y = (int)r1.y;
            r1.x = mad(6, (int)r0.z, (int)r1.x);

            InterlockedAdd(u0[r1.x], int(r1.y));

            // DebugRW[r0.w] = asfloat(cb0[65].x);
            // DebugRW[4] = asfloat(cb0[64].xyzw);

            r0.w = (int)r0.w + 1;
        }

        if (vThreadIDInGroup.x == 0) {
            r0.x = asfloat(cb0[65].x) * r0.x;
            r0.x = round(r0.x);
            r0.x = (int)r0.x;
            InterlockedAdd(u1[r0.y], int(r0.x));
        }
    }

    return;
}