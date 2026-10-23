pub fn restore_reverse_blocks(src: &[u8], dst: &mut [u8], block: usize) {
    assert!(block > 0 && src.len() == dst.len() && src.len() % block == 0, "invalid block geometry");
    for (from, to) in src.chunks_exact(block).rev().zip(dst.chunks_exact_mut(block)) {
        to.copy_from_slice(from);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn restores_odd_block_count_without_reversing_bytes_inside_blocks() {
        let mut dst = [0; 10];
        restore_reverse_blocks(&[0,1,2,3,4,5,6,7,8,9], &mut dst, 2);
        assert_eq!(dst, [8,9,6,7,4,5,2,3,0,1]);
    }
    #[test]
    #[should_panic(expected = "invalid block geometry")]
    fn rejects_partial_blocks() { restore_reverse_blocks(&[0;5], &mut [0;5], 2); }
}
