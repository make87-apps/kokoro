use anyhow::{bail, Context, Result};
use log::{debug, error, info, warn};
use make87::encodings::{Encoder, ProtobufEncoder};
use make87::interfaces::zenoh::{ConfiguredSubscriber, ZenohInterface};
use make87_messages::audio::FramePcmS16le;
use tokio::sync::mpsc;

// ======================== CPU Kokoro wrapper (v1.1; CMUdict g2p) ========================

use std::{fs, path::Path};
use kokoro_tts::{g2p, get_token_ids}; // make sure Cargo enables: default-features=false, features=["use-cmudict"]
use ort::session::Session;
use ort::value::Tensor;

use lru::LruCache;
use std::num::NonZeroUsize;

const SR: u32 = 24_000;
const SAMPLES_PER_FRAME: usize = 480; // 20ms @ 24k
const BYTES_PER_FRAME: usize = SAMPLES_PER_FRAME * 2;
const CHUNK_MAX_CHARS: usize = 140; // keep chunks short so they start fast

pub struct KokoroTtsCpu {
    session: Session,
    voices: Vec<f32>, // flat N*256
    speed: f32,
    // cache phoneme string -> token ids
    ph2ids: LruCache<String, Vec<i64>>,
}

impl KokoroTtsCpu {
    pub fn new(model_path: impl AsRef<Path>, voice_bin: impl AsRef<Path>) -> Result<Self> {
        // Preflight model
        let model_path = model_path.as_ref();
        let md = fs::metadata(model_path).with_context(|| format!("metadata({})", model_path.display()))?;
        if !md.is_file() { bail!("{} is not a file", model_path.display()); }
        if md.len() < 1024 { bail!("ONNX too small ({} bytes)", md.len()); }

        // CPU session
        let _env = ort::init().commit()?;
        let builder = ort::session::Session::builder()?
            .with_intra_threads(num_cpus::get().max(2))?;
        let session = builder
            .commit_from_file(model_path)
            .with_context(|| format!("load model {}", model_path.display()))?;
        info!("Loaded ONNX (CPU): {}", model_path.display());

        // Load voices (raw float32 N×256)
        let voices = {
            let bytes = fs::read(&voice_bin).with_context(|| format!("read voice {}", voice_bin.as_ref().display()))?;
            if bytes.len() % 4 != 0 { bail!("voice file size {} not multiple of 4", bytes.len()); }
            let mut v = vec![0f32; bytes.len() / 4];
            bytemuck::cast_slice_mut::<f32, u8>(&mut v).copy_from_slice(&bytes);
            if v.len() % 256 != 0 { bail!("voice float count {} not multiple of 256", v.len()); }
            info!("Loaded voice: {} ({} × 256)", voice_bin.as_ref().display(), v.len()/256);
            v
        };

        let speed = std::env::var("KOKORO_SPEED").ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(1.1);

        let mut tts = Self {
            session,
            voices,
            speed,
            ph2ids: LruCache::new(NonZeroUsize::new(4096).unwrap()),
        };

        // Warm-up: 2 pads + first style, speed=1.0
        tts.warm_up()?;
        Ok(tts)
    }

    fn warm_up(&mut self) -> Result<()> {
        let ids = vec![0i64, 0i64]; // pads
        let style = &self.voices[0..256];
        let ids_tensor = Tensor::from_array(([1usize, ids.len()], ids.into_boxed_slice()))?;
        let style_tensor = Tensor::from_array(([1usize, 256usize], style.to_vec().into_boxed_slice()))?;
        let speed_tensor = Tensor::from_array(([1usize], vec![1.0f32].into_boxed_slice()))?;
        let _ = self.session.run(ort::inputs! {
            "input_ids" => ids_tensor,
            "style"     => style_tensor,
            "speed"     => speed_tensor,
        })?;
        info!("ONNX warm-up completed");
        Ok(())
    }

    #[inline]
    fn sanitize_en(s: &str) -> String {
        // Normalize quotes/hyphens/ellipses, drop emoji/control (keeps ASCII + common punct)
        s.chars()
            .map(|c| match c {
                '“'|'”' => '"',
                '‘'|'’' => '\'',
                '—'|'–' => '-',
                '…'     => '.',
                _ => c,
            })
            .filter(|c| c.is_ascii() || " .,!?:;-'\"()".contains(*c))
            .collect()
    }

    /// Synthesize one utterance -> PCM S16LE bytes (whole utterance; caller streams frames)
    pub fn synth_bytes(&mut self, text: &str) -> Result<Vec<u8>> {
        let t = text.trim();
        if t.is_empty() {
            return Ok(Vec::new());
        }

        // 1) text -> phonemes (CMUdict path enabled via crate feature)
        let clean = Self::sanitize_en(t);
        let phonemes = g2p(&clean, true)
            .map_err(|e| anyhow::anyhow!("g2p failed: {e}"))?;
        debug!("IPA: {}", phonemes);

        // 2) phonemes -> token ids (pads included) with caching
        let mut ids = if let Some(ids) = self.ph2ids.get(&phonemes).cloned() {
            ids
        } else {
            let ids = get_token_ids(&phonemes, true);
            self.ph2ids.put(phonemes.clone(), ids.clone());
            ids
        };

        if ids.is_empty() {
            return Ok(Vec::new());
        }
        if ids.len() > 512 {
            warn!("token_ids len {} > 512; truncating", ids.len());
            ids.truncate(512);
        }

        // 3) style by phoneme length (pre-pad)
        let phones_len = phonemes.chars().count();
        let n_rows = self.voices.len() / 256;
        let idx = phones_len.min(n_rows.saturating_sub(1));
        let style = &self.voices[idx * 256 .. idx * 256 + 256];

        // 4) Build tensors
        let ids_tensor = Tensor::from_array(([1usize, ids.len()], ids.into_boxed_slice()))?;
        let style_tensor = Tensor::from_array(([1usize, 256usize], style.to_vec().into_boxed_slice()))?;
        let speed_tensor = Tensor::from_array(([1usize], vec![self.speed].into_boxed_slice()))?;

        // 5) Run ONNX
        let outputs = self.session.run(ort::inputs! {
            "input_ids" => ids_tensor,
            "style"     => style_tensor,
            "speed"     => speed_tensor,
        })?;

        // 6) Extract waveform
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let samples = data.to_vec();

        Ok(f32_to_pcm_s16le(&samples))
    }
}

fn f32_to_pcm_s16le(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &x in samples {
        let s = (x.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

// ------------------------------ incremental sentence extraction ------------------------------

fn extract_ready_sentences(buf: &mut String) -> Vec<String> {
    // Emit whenever we see ., !, ? — keep remainder in buf.
    let mut out = Vec::new();
    let mut cut = None;

    for (i, ch) in buf.char_indices() {
        if ch == '.' || ch == '!' || ch == '?' {
            cut = Some(i + ch.len_utf8());
        }
    }

    if let Some(end) = cut {
        let head = buf[..end].trim().to_string();
        if !head.is_empty() { out.push(head); }
        let tail = buf[end..].trim_start().to_string();
        *buf = tail;
    }

    // If still too long, spill a chunk at a word boundary to keep latency down.
    if buf.len() > CHUNK_MAX_CHARS {
        // find last space before the limit
        let mut cut_idx = None;
        for (i, ch) in buf.char_indices() {
            if i >= CHUNK_MAX_CHARS { break; }
            if ch.is_whitespace() { cut_idx = Some(i); }
        }
        let end = cut_idx.unwrap_or(CHUNK_MAX_CHARS.min(buf.len()));
        let head = buf[..end].trim().to_string();
        if !head.is_empty() { out.push(head); }
        let tail = buf[end..].trim_start().to_string();
        *buf = tail;
    }

    out
}

// ============================== main: as-live-as-possible ==============================

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    // ENV
    let model = std::env::var("KOKORO_MODEL").unwrap_or("/models/onnx/model_quantized.onnx".into());
    let voice = std::env::var("KOKORO_VOICE").unwrap_or("/models/voices/af_bella.bin".into());
    let threads = std::env::var("ORT_THREADS").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(2);

    info!("Starting Kokoro TTS node");
    info!("MODEL={} VOICE={} SPEED={} THREADS={}",
        model, voice,
        std::env::var("KOKORO_SPEED").unwrap_or("1.0".into()),
        threads
    );

    // Zenoh
    let zenoh = ZenohInterface::from_default_env("zenoh").expect("missing zenoh config");
    let session = zenoh.get_session().await.expect("zenoh session");
    info!("Zenoh session established");
    let sub = zenoh.get_subscriber(&session, "tts_text").await.expect("subscriber");
    let pub_audio = zenoh.get_publisher(&session, "tts_audio").await.expect("publisher");
    info!("Subscribed: tts_text; Publishing: tts_audio");

    let enc = ProtobufEncoder::<FramePcmS16le>::new();

    // Channels (keep PCM queue short to limit lag)
    let (tx_text_chunks, mut rx_text_chunks) = mpsc::channel::<String>(256);
    let (tx_pcm_out, mut rx_pcm_out) = mpsc::channel::<FramePcmS16le>(48); // ~1s max backlog

    // Subscriber task: accumulate incoming text, emit ready sentences / short chunks ASAP
    tokio::spawn({
        let mut rx_fifo;
        let mut rx_ring;
        match sub {
            ConfiguredSubscriber::Fifo(s) => { info!("Using FIFO subscriber for tts_text"); rx_fifo = Some(s); rx_ring = None; }
            ConfiguredSubscriber::Ring(s) => { info!("Using RING subscriber for tts_text"); rx_fifo = None; rx_ring = Some(s); }
        }
        let tx = tx_text_chunks.clone();
        async move {
            let mut buf = String::new();
            loop {
                let sample = match (&mut rx_fifo, &mut rx_ring) {
                    (Some(s), None) => s.recv_async().await,
                    (None, Some(s)) => s.recv_async().await,
                    _ => unreachable!(),
                };
                match sample {
                    Ok(sample) => {
                        let bytes = sample.payload().to_bytes();
                        match std::str::from_utf8(&bytes) {
                            Ok(s) => {
                                let trimmed = s.trim();
                                if trimmed.is_empty() { continue; }
                                if !buf.is_empty() { buf.push(' '); }
                                buf.push_str(trimmed);
                                debug!("RX agg buf len={} after append", buf.len());

                                let ready = extract_ready_sentences(&mut buf);
                                for seg in ready {
                                    debug!("Emit seg: '{}'", seg);
                                    if tx.send(seg).await.is_err() {
                                        warn!("text chunk channel closed; stopping subscriber");
                                        return;
                                    }
                                }
                                // If nothing was emitted and buffer is still big, extract_ready_sentences already spilled.
                            }
                            Err(e) => error!("utf8 decode error: {e:?}"),
                        }
                    }
                    Err(e) => {
                        error!("subscriber recv error: {e:?}");
                        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    }
                }
            }
        }
    });

    // TTS worker (blocking thread): synth each emitted chunk and STREAM frames immediately
    tokio::task::spawn_blocking({
        let model_path = model.clone();
        let voice_path = voice.clone();
        move || -> Result<()> {
            let mut tts = KokoroTtsCpu::new(&model_path, &voice_path)?;
            info!("kokoro ready @ {} Hz", SR);

            // global timeline PTS (in samples)
            let mut next_pts: i64 = 0;

            while let Some(chunk) = rx_text_chunks.blocking_recv() {
                let text = chunk.trim();
                if text.is_empty() { continue; }
                info!("Synth: '{}'", text);

                match tts.synth_bytes(text) {
                    Ok(pcm) if !pcm.is_empty() => {
                        debug!("PCM bytes for chunk: {}", pcm.len());
                        // STREAM: 20ms frames
                        let mut offset = 0usize;
                        while offset < pcm.len() {
                            let end = (offset + BYTES_PER_FRAME).min(pcm.len());
                            let data = pcm[offset..end].to_vec();

                            let frame = FramePcmS16le {
                                header: None,
                                data,
                                pts: next_pts,
                                time_base: Some(make87_messages::audio::frame_pcm_s16le::Fraction { num: 1, den: SR }),
                                channels: 1,
                            };
                            if tx_pcm_out.blocking_send(frame).is_err() {
                                warn!("pcm channel closed; exiting worker");
                                return Ok(());
                            }
                            let samples_in_frame = (end - offset) / 2;
                            next_pts += samples_in_frame as i64;
                            offset = end;
                        }
                    }
                    Ok(_) => warn!("Empty PCM for '{}'", text),
                    Err(e) => error!("synth error: {e:?}"),
                }
            }
            info!("Worker: text chunk channel ended; exiting");
            Ok(())
        }
    });

    // Publisher (async): send frames immediately
    tokio::spawn(async move {
        while let Some(frame) = rx_pcm_out.recv().await {
            let bytes = frame.data.len();
            match enc.encode(&frame) {
                Ok(enc_msg) => {
                    if let Err(e) = pub_audio.put(enc_msg).await {
                        error!("publish error: {e:?}");
                    } else {
                        debug!("Published PCM frame ({} bytes) -> tts_audio", bytes);
                    }
                }
                Err(e) => error!("encode error: {e:?}"),
            }
        }
        info!("Publisher: pcm channel ended; exiting");
    });

    futures::future::pending::<()>().await;
    Ok(())
}
