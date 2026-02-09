struct SortUniforms {
    element_count: u32,
    block_size: u32,
    comparison_distance: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> sort_uniforms: SortUniforms;
@group(0) @binding(1) var<storage, read_write> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> values: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_index = global_id.x;
    let comparison_distance = sort_uniforms.comparison_distance;
    let block_size = sort_uniforms.block_size;
    let element_count = sort_uniforms.element_count;

    let pair_index = thread_index;
    let block_offset = (pair_index / comparison_distance) * comparison_distance * 2u;
    let local_offset = pair_index % comparison_distance;
    let left = block_offset + local_offset;
    let right = left + comparison_distance;

    if right >= element_count {
        return;
    }

    let ascending = (left & block_size) == 0u;

    let key_left = keys[left];
    let key_right = keys[right];

    let should_swap = select((key_left < key_right), (key_left > key_right), ascending);

    if should_swap {
        keys[left] = key_right;
        keys[right] = key_left;

        let val_left = values[left];
        let val_right = values[right];
        values[left] = val_right;
        values[right] = val_left;
    }
}
