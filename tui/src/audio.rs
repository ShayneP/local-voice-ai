use std::collections::VecDeque;
use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Stream;
use livekit::webrtc::{
    audio_frame::AudioFrame,
    audio_source::native::NativeAudioSource,
    prelude::AudioSourceOptions,
};
use parking_lot::Mutex;

use crate::livekit_client::{NUM_CHANNELS, SAMPLE_RATE};

const SAMPLES_PER_10MS: usize = (SAMPLE_RATE / 100) as usize;
const VIS_BANDS: usize = 32;

/// Shared audio state accessed by cpal callbacks and the app.
pub struct AudioHandle {
    audio_source: NativeAudioSource,
    capture_buf: Mutex<VecDeque<i16>>,
    playback_buf: Mutex<VecDeque<i16>>,
    /// RMS levels for the visualizer (0.0–1.0), one per band.
    mic_levels: Mutex<Vec<f32>>,
    agent_levels: Mutex<Vec<f32>>,
    mic_muted: Mutex<bool>,
}

impl AudioHandle {
    pub fn new() -> Arc<Self> {
        let audio_source = NativeAudioSource::new(
            AudioSourceOptions {
                echo_cancellation: true,
                noise_suppression: true,
                auto_gain_control: true,
            },
            SAMPLE_RATE,
            NUM_CHANNELS,
            100, // 100ms buffer
        );

        Arc::new(Self {
            audio_source,
            capture_buf: Mutex::new(VecDeque::with_capacity(SAMPLE_RATE as usize)),
            playback_buf: Mutex::new(VecDeque::with_capacity(SAMPLE_RATE as usize)),
            mic_levels: Mutex::new(vec![0.0; VIS_BANDS]),
            agent_levels: Mutex::new(vec![0.0; VIS_BANDS]),
            mic_muted: Mutex::new(true), // start muted (text mode)
        })
    }

    pub fn audio_source(&self) -> &NativeAudioSource {
        &self.audio_source
    }

    pub fn set_mic_muted(&self, muted: bool) {
        *self.mic_muted.lock() = muted;
    }

    pub fn mic_levels(&self) -> Vec<f32> {
        self.mic_levels.lock().clone()
    }

    pub fn agent_levels(&self) -> Vec<f32> {
        self.agent_levels.lock().clone()
    }

    /// Called by the LiveKit event loop when a remote audio frame arrives.
    pub fn push_playback_frame(&self, frame: &AudioFrame<'_>) {
        let samples: &[i16] = frame.data.as_ref();

        // Compute agent audio levels
        update_levels(&self.agent_levels, samples);

        // Buffer for cpal output
        let mut buf = self.playback_buf.lock();
        let max_buf = SAMPLE_RATE as usize; // 1s max
        while buf.len() + samples.len() > max_buf {
            buf.pop_front();
        }
        buf.extend(samples.iter());
    }

    /// Drain capture buffer and send 10ms frames to LiveKit.
    pub async fn flush_capture(&self) {
        loop {
            let frame_data = {
                let mut buf = self.capture_buf.lock();
                if buf.len() < SAMPLES_PER_10MS {
                    break;
                }
                let data: Vec<i16> = buf.drain(..SAMPLES_PER_10MS).collect();
                data
            };

            let frame = AudioFrame {
                data: frame_data.into(),
                sample_rate: SAMPLE_RATE,
                num_channels: NUM_CHANNELS,
                samples_per_channel: SAMPLES_PER_10MS as u32,
            };

            if let Err(e) = self.audio_source.capture_frame(&frame).await {
                log::error!("capture_frame error: {}", e);
                break;
            }
        }
    }
}

/// Start cpal input (mic) and output (speaker) streams.
/// Returns the stream handles (must be kept alive).
pub fn start_audio_streams(handle: Arc<AudioHandle>) -> anyhow::Result<(Stream, Stream)> {
    let host = cpal::default_host();

    // --- Input (mic) ---
    let input_device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device found"))?;

    let input_config = cpal::StreamConfig {
        channels: NUM_CHANNELS as u16,
        sample_rate: cpal::SampleRate::from(SAMPLE_RATE),
        buffer_size: cpal::BufferSize::Default,
    };

    let handle_in = handle.clone();
    let input_stream = input_device.build_input_stream(
        &input_config,
        move |data: &[i16], _: &cpal::InputCallbackInfo| {
            if *handle_in.mic_muted.lock() {
                // Still compute levels for visualizer but don't capture
                let mut levels = handle_in.mic_levels.lock();
                for l in levels.iter_mut() {
                    *l *= 0.85; // decay
                }
                return;
            }

            // Update mic visualizer levels
            update_levels(&handle_in.mic_levels, data);

            // Buffer samples for LiveKit
            let mut buf = handle_in.capture_buf.lock();
            buf.extend(data.iter());
        },
        |err| log::error!("Input stream error: {}", err),
        None,
    )?;

    // --- Output (speakers) ---
    let output_device = host
        .default_output_device()
        .ok_or_else(|| anyhow::anyhow!("No output device found"))?;

    let output_config = cpal::StreamConfig {
        channels: NUM_CHANNELS as u16,
        sample_rate: cpal::SampleRate::from(SAMPLE_RATE),
        buffer_size: cpal::BufferSize::Default,
    };

    let handle_out = handle.clone();
    let output_stream = output_device.build_output_stream(
        &output_config,
        move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
            let mut buf = handle_out.playback_buf.lock();
            for sample in data.iter_mut() {
                *sample = buf.pop_front().unwrap_or(0);
            }
        },
        |err| log::error!("Output stream error: {}", err),
        None,
    )?;

    input_stream.play()?;
    output_stream.play()?;

    Ok((input_stream, output_stream))
}

/// Compute pseudo-frequency band levels from raw PCM samples.
fn update_levels(levels: &Mutex<Vec<f32>>, samples: &[i16]) {
    let mut lvl = levels.lock();
    let band_count = lvl.len();
    if samples.is_empty() || band_count == 0 {
        return;
    }

    let chunk_size = (samples.len() / band_count).max(1);
    for (i, chunk) in samples.chunks(chunk_size).enumerate() {
        if i >= band_count {
            break;
        }
        let rms = (chunk.iter().map(|&s| (s as f64).powi(2)).sum::<f64>()
            / chunk.len() as f64)
            .sqrt()
            / 32768.0;
        // Amplify — normal speech is ~1-5% of full scale
        let rms = (rms as f32 * 12.0).min(1.0);
        // Smooth: rise fast, decay slow
        let current = lvl[i];
        lvl[i] = if rms > current {
            current * 0.2 + rms * 0.8
        } else {
            current * 0.92 + rms * 0.08
        };
    }
}
