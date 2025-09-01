use anyhow::{anyhow, Context, Result};
use log::{error, info};
use make87::encodings::{Encoder, ProtobufEncoder};
use make87::interfaces::zenoh::{ConfiguredSubscriber, ZenohInterface};
use make87_messages::audio::FramePcmS16le;
use ndarray::{arr1, Array, Axis};
#[cfg(feature = "gpu")]
use ort::execution_providers::CUDAExecutionProvider;
use ort::{session::Session};
use tokenizers::Tokenizer;
use std::sync::Arc;
use ort::session::SessionOutputs;
use ort::value::Tensor;
use tokio::sync::mpsc;

pub struct Model {
    pub session: Session,
    pub tokenizer: Tokenizer,
    pub voice_vecs: Vec<f32>, // concatenated voices: N * 256
    pub sr: u32,
    pub speed: f32,
}

impl Model {
    pub fn new(
        weights_path: &str,
        tokenizer_path: &str,
        voice_bin_path: &str,
        num_threads: usize,
        speed: f32,
    ) -> Result<Self> {
        let _env = ort::init().commit()?;
        let mut builder = Session::builder()?;
        #[cfg(feature = "gpu")]
        {
            builder = builder.with_execution_providers([CUDAExecutionProvider::default().build()])?;
        }
        #[cfg(not(feature = "gpu"))]
        {
            builder = builder.with_intra_threads(num_threads)?;
        }
        let session = builder
            .commit_from_file(weights_path)
            .with_context(|| format!("load model {weights_path}"))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("{e}"))
            .with_context(|| format!("load tokenizer {tokenizer_path}"))?;

        let voice_vecs = load_voice_bin(voice_bin_path)?;
        Ok(Self { session, tokenizer, voice_vecs, sr: 24_000, speed })
    }

    /// Synthesize one utterance -> PCM S16LE bytes
    pub fn synth_bytes(&mut self, text: &str) -> Result<Vec<u8>> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // tokenize
        let enc = self.tokenizer.encode(text, true).map_err(|e| anyhow!("{e}"))?;
        let ids_i64: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();

        // voice style (256 f32)
        let style_slice = pick_voice_style(&self.voice_vecs, ids_i64.len());

        // speed
        let speed_val = [self.speed];

        // ----- build ONNX tensors WITHOUT ndarray -----
        // input_ids: [1, T] i64
        let input_ids = Tensor::from_array(
            ([1usize, ids_i64.len()], ids_i64.clone().into_boxed_slice())
        )?;

        // style: [1, 256] f32
        let style = Tensor::from_array(
            ([1usize, 256usize], style_slice.to_vec().into_boxed_slice())
        )?;

        // speed: [1] f32
        let speed = Tensor::from_array(
            ([1usize], vec![self.speed].into_boxed_slice())
        )?;

        // run
        let outputs: SessionOutputs = self.session.run(ort::inputs! {
            "input_ids" => input_ids,
            "style"     => style,
            "speed"     => speed,
        })?;

        // ----- extract output as view -----
        // expect a tensor of f32 with shape [1, N]
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        // let dims = **shape;
        // if dims.len() != 2 || dims[0] != 1 {
        //     return Err(anyhow!("unexpected output shape: {:?}", dims));
        // }

        // data is &[f32] length N
        let samples: Vec<f32> = data.to_vec();

        Ok(f32_to_pcm_s16le(&samples))
    }
}

fn load_voice_bin(path: &str) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read voice bin {path}"))?;
    if bytes.len() % 4 != 0 {
        return Err(anyhow!("voice bin size not multiple of 4"));
    }
    let mut v = vec![0f32; bytes.len() / 4];
    bytemuck::cast_slice_mut::<f32, u8>(&mut v).copy_from_slice(&bytes);
    Ok(v)
}

fn pick_voice_style(voices: &[f32], token_count: usize) -> &[f32] {
    let n = voices.len() / 256;
    let idx = token_count.min(n.saturating_sub(1));
    let start = idx * 256;
    &voices[start..start + 256]
}

fn f32_to_pcm_s16le(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &x in samples {
        let s = (x.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    // env
    let model = std::env::var("KOKORO_MODEL").unwrap_or("/models/onnx/model_quantized.onnx".into());
    let tok = std::env::var("KOKORO_TOKENIZER").unwrap_or("/models/tokenizer.json".into());
    let voice = std::env::var("KOKORO_VOICE").unwrap_or("/models/voices/af_bella.bin".into());
    let speed: f32 = std::env::var("KOKORO_SPEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let threads: usize = std::env::var("ORT_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(2);

    // model (Arc so we can move into blocking thread)


    // zenoh IO: UTF-8 in, PCM16 proto out
    let zenoh = ZenohInterface::from_default_env("zenoh").expect("missing zenoh config");
    let session = zenoh.get_session().await.expect("zenoh session");
    let sub = zenoh.get_subscriber(&session, "tts_text").await.expect("subscriber"); // UTF-8 in
    let pub_audio = zenoh.get_publisher(&session, "tts_audio").await.expect("publisher"); // PCM16 out
    let enc = ProtobufEncoder::<FramePcmS16le>::new();

    // Channels:
    //   text_in: subscriber -> blocking TTS worker
    //   pcm_out: blocking TTS worker -> async publisher
    let (tx_text_in, mut rx_text_in) = mpsc::channel::<String>(256);
    let (tx_pcm_out, mut rx_pcm_out) = mpsc::channel::<FramePcmS16le>(64);

    // 1) Async subscriber task: read Zenoh text, push to text_in
    tokio::spawn({
        let mut rx_fifo;
        let mut rx_ring;
        match sub {
            ConfiguredSubscriber::Fifo(s) => { rx_fifo = Some(s); rx_ring = None; }
            ConfiguredSubscriber::Ring(s) => { rx_fifo = None;    rx_ring = Some(s); }
        }
        let tx = tx_text_in.clone();
        async move {
            loop {
                let sample = match (&mut rx_fifo, &mut rx_ring) {
                    (Some(s), None) => s.recv_async().await,
                    (None, Some(s)) => s.recv_async().await,
                    _ => unreachable!(),
                };
                match sample {
                    Ok(sample) => {
                        match std::str::from_utf8(&sample.payload().to_bytes()) {
                            Ok(s) => {
                                if tx.send(s.trim().to_string()).await.is_err() {
                                    break;
                                }
                            }
                            Err(e) => error!("utf8 decode error: {e:?}"),
                        }
                    }
                    Err(e) => {
                        error!("subscriber recv error: {e:?}");
                        // brief backoff to avoid tight error loop
                        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    }
                }
            }
        }
    });

    // 2) Blocking TTS worker: pull text, run synth, push PCM frames out
    tokio::task::spawn_blocking({
        move || -> Result<()> {
            let mut tts = Model::new(&model, &tok, &voice, threads, speed).expect("load kokoro model");
            info!("kokoro ready @ {} Hz", tts.sr);
            while let Some(text) = rx_text_in.blocking_recv() {
                let text = text.trim();
                if text.is_empty() { continue; }

                let chunks = text.split_inclusive(&['.', '?', '!', ';', ',', ':'][..])
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>();

                for chunk in chunks {
                    match tts.synth_bytes(chunk) {
                        Ok(pcm) if !pcm.is_empty() => {
                            let frame = FramePcmS16le {
                                header: None,
                                data: pcm,
                                pts: 0,
                                time_base: Some(make87_messages::audio::frame_pcm_s16le::Fraction {
                                    num: 1,
                                    den: tts.sr.clone(),
                                }),
                                channels: 1,
                            };
                            // If channel is full, drop oldest by using try_send, or block with blocking_send for backpressure
                            if tx_pcm_out.blocking_send(frame).is_err() {
                                // receiver gone; exit
                                break;
                            }
                        }
                        Ok(_) => {} // empty output, skip
                        Err(e) => error!("synth error: {e:?}"),
                    }
                }
            }
            Ok(())
        }
    });

    // 3) Async publisher: pull PCM frames, encode, publish over Zenoh
    tokio::spawn(async move {
        while let Some(frame) = rx_pcm_out.recv().await {
            match enc.encode(&frame) {
                Ok(enc_msg) => {
                    if let Err(e) = pub_audio.put(enc_msg).await {
                        error!("publish error: {e:?}");
                    }
                }
                Err(e) => error!("encode error: {e:?}"),
            }
        }
    });

    // Park forever (or add shutdown signal handling)
    futures::future::pending::<()>().await;
    Ok(())
    // Ok::<_, anyhow::Error>(())  // unreachable
}
