use std::time::Instant;
use debabelizer::VoiceProcessor;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let _processor = VoiceProcessor::new()?;
    let duration = start.elapsed();
    
    println\!("Rust VoiceProcessor init: {:.2}ms", duration.as_secs_f64() * 1000.0);
    Ok(())
}
EOF < /dev/null
