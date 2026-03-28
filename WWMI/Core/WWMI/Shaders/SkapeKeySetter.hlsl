
Texture1D<float4> IniParams : register(t120);

#define ActionType IniParams[0].x
#define ShapeKeyId IniParams[0].y
#define ShapeKeyValue IniParams[0].z

RWBuffer<float4> CustomShapeKeyValuesRW : register(u5);

// RWBuffer<float4> DebugRW : register(u7);


#ifdef COMPUTE_SHADER

[numthreads(1,1,1)]
void main(uint3 ThreadId : SV_DispatchThreadID)
{
    // Expected user input for ShapeKeyId is in [0, 253] range
    // When more than 1 shapekeys batch is used, we need to align value pos to 128
    // It's required because 1 batch can fit only 127 shapekeys, while ShapeKeyOverrider reads aligned sequence of 128
    // We don't want to confuse the user with gaps in ids, and don't want to complicate component math of ShapeKeyOverrider
    // So we add +1 to offset per each batch before the one where ShapeKeyId belongs
    uint container_offset = uint(ShapeKeyId / 127);
    // Custom shape key values are stored in CustomShapeKeyValuesRW buffer with length of 32 float4 per batch
    // Each element of CustomShapeKeyValuesRW buffer stores values of 4 shape keys in its components
    // So, first of all, we need to calculate id of element and id of its component based on provided ShapeKeyId
    uint shape_key_id = container_offset + uint(ShapeKeyId);
    uint shape_key_component_id = shape_key_id % uint(4);
    uint shape_key_element_id = (shape_key_id - shape_key_component_id) / uint(4);
    float shape_key_value = float(ShapeKeyValue);
    // DebugRW[0] = float4(shape_key_id, shape_key_component_id, shape_key_element_id, shape_key_value);
    
    // Get element that stores value of requested shape key
    float4 element = CustomShapeKeyValuesRW[shape_key_element_id];

    // ShapeKeyOverrider.hlsl treats custom shape key values in a following way:
    // 1. If 'shape_key_value == 0.0', default shape key value won't be overriden
    // 2. If 'shape_key_value == 1000000.0', default shape key will be overrided with zero (0.0)
    // 3. If 'shape_key_value != 0', default shape key will be overrided with 'shape_key_value - 1000000.0'
    
    // So, following 'shape_key_value' encoding is required:
    // 1. To enable override, we need to set 'shape_key_value += 1.0'
    // 2. To disable override, we need to set 'shape_key_value = 0'
    if (ActionType == 0) {
        // Enable override for this Shape Key
        shape_key_value += 1000000.0;
    } else if (ActionType == 1) {
        // Disable override for this Shape Key
        shape_key_value = 0;
    }

    // Update element that stores value of requested shape key with new value
	switch(shape_key_component_id) {
		case 0:
			element.x = shape_key_value;
			break;
		case 1:
			element.y = shape_key_value;
			break;
		case 2:
			element.z = shape_key_value;
			break;
		case 3:
			element.w = shape_key_value;
			break;
	};

    // Set updated element that stores value of requested shape key
    CustomShapeKeyValuesRW[shape_key_element_id] = element;

}

#endif