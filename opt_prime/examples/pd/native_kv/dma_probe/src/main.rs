use furiosa_opt_rt::{Device, image};
use kv_read_probe::restore_reverse_blocks;
use std::time::Instant;
#[tokio::main(flavor="current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::builder((1, 8)).among([0]).open()?;
    let rank = device.rank(0)?;
    let block = 2048;
    for count in [127,511,1023] {
        let bytes = count*block;
        let source: Vec<u8> = (0..bytes).map(|i| ((i as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ ((i as u64)>>11)) as u8).collect();
        let expected: Vec<u8> = (0..bytes).map(|i| source[(count-1-i/block)*block+i%block]).collect();
        let buffer = device.alloc(image::Memory::Dram, bytes)?;
        device.write([(source.as_slice(), buffer.on(rank))]).await?;
        let mut destination = vec![0u8;bytes];
        let mut staging = vec![0u8;bytes];
        for repeat in 0..21 {
            for merged in if repeat%2==0 { [false,true] } else { [true,false] } {
                let start = Instant::now();
                if merged {
                    device.read([(buffer.on(rank),staging.as_mut_slice())]).await?;
                } else {
                    let pairs = destination.chunks_exact_mut(block).enumerate().map(|(i, chunk)| {
                        let from = (count-1-i)*block;
                        (buffer.slice(from..from+block).on(rank),chunk)
                    });
                    device.read(pairs).await?;
                }
                let dma_ms = start.elapsed().as_secs_f64()*1000.;
                if merged { restore_reverse_blocks(&staging,&mut destination,block); }
                let total_ms = start.elapsed().as_secs_f64()*1000.;
                assert_eq!(destination,expected,"read/order mismatch");
                println!("{}",serde_json::json!({"blocks":count,"block_bytes":block,"bytes":bytes,"repeat":repeat,"warmup":repeat==0,"merged":merged,"dma_ms":dma_ms,"total_ms":total_ms,"equal":true}));
            }
        }
    }
    Ok(())
}
