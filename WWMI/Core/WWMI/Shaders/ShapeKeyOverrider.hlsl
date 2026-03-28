// Shader: WWMI Shape Key Overrider
// Version: 2.0
// Creator: SpectumQT
// Comment: Builds custom CB that controls Shape Key CS chain:
// * Filters out objects with mismatching shape key offsets checksum (THREAD_GROUP_COUNT_Y alone may not be enough)
// * Overrides default shape key offsets with ones from custom buffer (exported from Blender)
// * Overrides default shape key values with ones from custom buffer (modified via ini WWMI calls)
// * Overrides default vertex offset with one from INI scope (required when more than 1 batch is used)

Texture1D<float4> IniParams : register(t120);

#define ShapeKeyCheckSumBatch0 IniParams[0].x
#define OriginalVertexOffsetBatch0 IniParams[0].y
#define CustomVertexOffsetBatch0 IniParams[0].z

#define ShapeKeyCheckSumBatch1 IniParams[1].x
#define OriginalVertexOffsetBatch1 IniParams[1].y
#define CustomVertexOffsetBatch1 IniParams[1].z

// IniParams[0] x = ShapeKeyCheckSum, y = BatchOffset, z = OriginalVertexOffset, 2 = BatchCustomVertexOffset

// CB0 structure (len == 66):
// 1. 32 uint4 shapekey offsets (static, total 128)
// 2. 32 unorm4 shapekey values (dynamic, total 128)
// 3. 2 uint4 CS logic vars (dynamic)
cbuffer cb0 : register(b0)
{
  uint4 cb0[66];
}

// Custom shape key offsets exported from Blender
Buffer<uint4> CustomShapeKeys : register(t33);

// Custom shape key values configured via ini WWMI calls
RWBuffer<float4> CustomShapeKeyValuesRW : register(u5);

// Output: custom CB constructed based on default CB0 and configured overrides
RWBuffer<uint4> ShapeKeysControlCBRW : register(u6);

// RWBuffer<float4> DebugRW : register(u7);


#ifdef COMPUTE_SHADER

groupshared uint is_custom_mesh;
groupshared uint shapekey_offset;
groupshared uint vertex_offset_original;
groupshared uint vertex_offset_custom;

[numthreads(64,1,1)]
void main(uint3 ThreadId : SV_DispatchThreadID)
{
    uint idx = ThreadId.x;

    // Get config for current batch
    if (idx == 0) {
        // Simple sum of cb0[0] xyzw components done in HLSL way
        uint checksum = dot(cb0[0], uint4(1,1,1,1));

        is_custom_mesh = 0;

        // if (cb0[65].y == 0) {
        //     DebugRW[0] = float4(cb0[65].y, checksum, asfloat(ShapeKeyCheckSumBatch0), 0);
        // } else {
        //     DebugRW[1] = float4(cb0[65].y, checksum, asfloat(ShapeKeyCheckSumBatch1), 0);
        // }

        if (checksum == uint(ShapeKeyCheckSumBatch0)) {
            is_custom_mesh = 1;
            shapekey_offset = 0;
            vertex_offset_original = uint(OriginalVertexOffsetBatch0);
            vertex_offset_custom = asuint(CustomVertexOffsetBatch0);
            // DebugRW[0] = float4(cb0[65].y, checksum, ShapeKeyCheckSumBatch0, 1);
        } else if (checksum == uint(ShapeKeyCheckSumBatch1)) {
            is_custom_mesh = 1;
            shapekey_offset = 32;
            vertex_offset_original = uint(OriginalVertexOffsetBatch1);
            vertex_offset_custom = asuint(CustomVertexOffsetBatch1);
            // DebugRW[1] = float4(cb0[65].y, checksum, ShapeKeyCheckSumBatch1, 1);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Copy logic vars (64 and 65 indices) from CB0 to ShapeKeysControlCBRW
    if (idx < 2) {
        uint4 val = cb0[64 + idx];

        // Override original vertex offset for current batch
        if (is_custom_mesh == 1 && idx == 1 && val.y == vertex_offset_original) {
            val.y = vertex_offset_custom;
        }

        ShapeKeysControlCBRW[64 + idx] = val;
    }

    uint4 output_values;

    if (is_custom_mesh == 1) {
        // Original CB0 data matches the checksum, apply overrides
        if (idx < 32) {
            // Override current shape key offsets with custom ones if current CB0 matches dumped CB0
            // Copy custom shape key offsets if current CB0 matches dumped CB0
            output_values = CustomShapeKeys[shapekey_offset+idx];
        } else {
            // Override default values with custom values (zero is encoded as 1000000.0)
            // 1. If 'shape_key_value == 0.0', default shape key value won't be overriden
            // 2. If 'shape_key_value == 1000000.0', default shape key will be overrided with zero (0.0)
            // 3. If 'shape_key_value != 0', default shape key will be overrided with 'shape_key_value - 1000000.0'
            // Use original cb0 values as base
            uint4 base_values = cb0[idx];
            // Copy custom shape key values to override defaults
            float4 custom_values = CustomShapeKeyValuesRW[shapekey_offset + idx - 32];
            output_values.x = custom_values.x != 0 ? asuint(custom_values.x - 1000000.0) : base_values.x;
            output_values.y = custom_values.y != 0 ? asuint(custom_values.y - 1000000.0) : base_values.y;
            output_values.z = custom_values.z != 0 ? asuint(custom_values.z - 1000000.0) : base_values.z;
            output_values.w = custom_values.w != 0 ? asuint(custom_values.w - 1000000.0) : base_values.w;
        }
        // DebugRW[idx] = CustomShapeKeys[idx];
    } else {
        // Original CB0 data does not match the checksum and should not be overriden.
        // Just copy original values to allow the data passthrough.
        output_values = cb0[idx];
    }

    ShapeKeysControlCBRW[idx] = output_values;
    
    // DebugRW[idx] = ShapeKeysControlCBRW[idx];
}

#endif