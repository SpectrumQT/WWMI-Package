
Buffer<uint> ReverseMap : register(t34);
Buffer<uint> FullRangeVGBuffer : register(t35);

RWBuffer<uint> RemappedBlendBuffer : register(u4);
// RWBuffer<float4> DebugRW : register(u7);

Texture1D<float4> IniParams : register(t120);

#define VertexCount IniParams[0].y
#define RemapId IniParams[2].x
#define WeightsPerVertexCount (IniParams[2].y > 0 ? IniParams[2].y : 4)
#define RemapBoneCount IniParams[2].w


#ifdef COMPUTE_SHADER

[numthreads(64,1,1)]
void main(uint3 ThreadId : SV_DispatchThreadID)
{
	int vertex_id = ThreadId.x;

    if (vertex_id >= VertexCount) {
        return;
    }

    // Remap matrices are stored as concatenated array where each map is RemapBoneCount values long
    int map_offset = RemapId * (int)RemapBoneCount;

    // We're working with continuous arrays instead of structured buffers here
    // So the whole magic has to be done via direct index-based addressing:
    
    // * RemappedBlendBuffer stores data as flat array of [VG0_ID, VG1_ID, VG2_ID, VG3_ID, VG0_WEIGHT, VG1_WEIGHT, VG2_WEIGHT, VG3_WEIGHT]
    //   - Note: Stored VG ids are 8-bit and cannot be used for remapping, so we use FullRangeVGBuffer instead
    int remapped_data_offset = vertex_id * WeightsPerVertexCount * 2;

    // * FullRangeVGBuffer stores data as flat array of [VG0_ID, VG1_ID, VG2_ID, VG3_ID]
    //   - Note: Stored VG ids are 16-bit
    int vg_data_offset = vertex_id * WeightsPerVertexCount;

    // Loop through all VGs of current vertex and remap each one
    for (int i = 0; i < WeightsPerVertexCount; i++)
    {
        RemappedBlendBuffer[remapped_data_offset+i] = ReverseMap[map_offset+FullRangeVGBuffer[vg_data_offset+i]];
    }
}

#endif