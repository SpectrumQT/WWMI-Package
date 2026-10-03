
Buffer<uint> ForwardMap : register(t37);
Buffer<float4> MergedSkeleton : register(t38);

RWBuffer<float4> RemappedSkeleton : register(u5);
// RWBuffer<float4> DebugRW : register(u7);

Texture1D<float4> IniParams : register(t120);

#define VertexGroupCount IniParams[1].w
#define RemapId IniParams[2].x
#define RemapBoneCount IniParams[2].w


#ifdef COMPUTE_SHADER

[numthreads(64,1,1)]
void main(uint3 ThreadId : SV_DispatchThreadID)
{
	int vg_id = ThreadId.x;

    if (vg_id >= VertexGroupCount) {
        return;
    }

    int vg_offset = vg_id * 3;
    int map_offset = RemapId * (int)RemapBoneCount;
    int remapped_vg_id = ForwardMap[map_offset+vg_id] * 3;

    RemappedSkeleton[vg_offset] = MergedSkeleton[remapped_vg_id];
    RemappedSkeleton[vg_offset+1] = MergedSkeleton[remapped_vg_id+1];
    RemappedSkeleton[vg_offset+2] = MergedSkeleton[remapped_vg_id+2];
    
    // DebugRW[VertexGroupOffset+vg_id] = Skeleton[vg_id];
}

#endif