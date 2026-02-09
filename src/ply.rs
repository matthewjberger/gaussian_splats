use crate::gaussian::RawGaussian;

pub fn load_ply_from_bytes(data: &[u8]) -> Vec<RawGaussian> {
    let (header_end, line_ending_len) = find_header_end(data);
    let header_str =
        std::str::from_utf8(&data[..header_end]).expect("PLY header is not valid UTF-8");

    let vertex_count = parse_vertex_count(header_str);

    let body_start = header_end + b"end_header".len() + line_ending_len;
    let body = &data[body_start..];

    let expected_size = vertex_count * std::mem::size_of::<RawGaussian>();
    assert!(
        body.len() >= expected_size,
        "PLY body too small: expected at least {} bytes for {} vertices, got {}",
        expected_size,
        vertex_count,
        body.len()
    );

    let raw_slice: &[RawGaussian] = bytemuck::cast_slice(&body[..expected_size]);

    raw_slice.to_vec()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_ply(path: &std::path::Path) -> Vec<RawGaussian> {
    let data = std::fs::read(path).unwrap_or_else(|error| {
        panic!("Failed to read PLY file {}: {}", path.display(), error);
    });
    load_ply_from_bytes(&data)
}

fn find_header_end(data: &[u8]) -> (usize, usize) {
    let needle_lf = b"end_header\n";
    let needle_crlf = b"end_header\r\n";

    if let Some(position) = data
        .windows(needle_crlf.len())
        .position(|window| window == needle_crlf)
    {
        return (position, 2);
    }

    if let Some(position) = data
        .windows(needle_lf.len())
        .position(|window| window == needle_lf)
    {
        return (position, 1);
    }

    panic!("Could not find 'end_header' in PLY file");
}

fn parse_vertex_count(header: &str) -> usize {
    for line in header.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("element vertex ") {
            let count_str = trimmed
                .strip_prefix("element vertex ")
                .expect("Failed to parse vertex count");
            return count_str
                .trim()
                .parse::<usize>()
                .expect("Vertex count is not a valid number");
        }
    }
    panic!("PLY header does not contain 'element vertex' line");
}
