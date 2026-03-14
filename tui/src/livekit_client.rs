use std::sync::Arc;

use futures::StreamExt;
use livekit::{
    prelude::*,
    track::{LocalAudioTrack, LocalTrack, RemoteTrack, TrackSource},
    webrtc::{
        audio_stream::native::NativeAudioStream,
        prelude::RtcAudioSource,
    },
    Room, RoomOptions,
};
use tokio::sync::mpsc;

use crate::audio::AudioHandle;

pub const SAMPLE_RATE: u32 = 48000;
pub const NUM_CHANNELS: u32 = 1;

/// Events forwarded from the LiveKit room to the app.
#[derive(Debug, Clone)]
pub enum LkEvent {
    Connected {
        room_name: String,
    },
    /// Transcription of what the user said (STT result)
    UserTranscription {
        text: String,
        is_final: bool,
    },
    /// Transcription of what the agent said (agent response)
    AgentTranscription {
        text: String,
        is_final: bool,
    },
    ChatMessage {
        sender: String,
        message: String,
    },
    Disconnected,
}

pub struct LkClient {
    pub room: Arc<Room>,
    event_rx: mpsc::UnboundedReceiver<RoomEvent>,
}

impl LkClient {
    pub async fn connect(url: &str, token: &str) -> anyhow::Result<(Self, Arc<Room>)> {
        let mut opts = RoomOptions::default();
        opts.auto_subscribe = true;

        let (room, events) = Room::connect(url, token, opts).await?;
        let room = Arc::new(room);

        Ok((
            Self {
                room: room.clone(),
                event_rx: events,
            },
            room,
        ))
    }

    /// Spawn a task that publishes local mic audio and listens for room events.
    /// Returns a channel of LkEvents for the app to consume.
    pub fn spawn_event_loop(
        mut self,
        audio_handle: Arc<AudioHandle>,
    ) -> mpsc::UnboundedReceiver<LkEvent> {
        let (tx, rx) = mpsc::unbounded_channel();
        let room = self.room.clone();

        // Capture local identity for comparing against transcription participants
        let local_identity = room.local_participant().identity().to_string();

        // Publish mic audio track
        let audio_source = audio_handle.audio_source().clone();
        let room_for_publish = room.clone();
        tokio::spawn(async move {
            let track = LocalAudioTrack::create_audio_track(
                "microphone",
                RtcAudioSource::Native(audio_source),
            );
            if let Err(e) = room_for_publish
                .local_participant()
                .publish_track(
                    LocalTrack::Audio(track),
                    livekit::options::TrackPublishOptions {
                        source: TrackSource::Microphone,
                        ..Default::default()
                    },
                )
                .await
            {
                log::error!("Failed to publish mic track: {}", e);
            }
        });

        let _ = tx.send(LkEvent::Connected {
            room_name: room.name().to_string(),
        });

        // Event loop
        let audio_for_events = audio_handle.clone();
        tokio::spawn(async move {
            while let Some(event) = self.event_rx.recv().await {
                match event {
                    RoomEvent::TrackSubscribed {
                        track,
                        publication: _,
                        participant,
                    } => {
                        if let RemoteTrack::Audio(audio_track) = track {
                            log::info!(
                                "Subscribed to audio from {}",
                                participant.identity()
                            );
                            let audio = audio_for_events.clone();
                            tokio::spawn(async move {
                                let mut stream = NativeAudioStream::new(
                                    audio_track.rtc_track(),
                                    SAMPLE_RATE as i32,
                                    NUM_CHANNELS as i32,
                                );
                                while let Some(frame) = stream.next().await {
                                    audio.push_playback_frame(&frame);
                                }
                            });
                        }
                    }
                    RoomEvent::TranscriptionReceived {
                        participant,
                        segments,
                        ..
                    } => {
                        // Determine if this is user speech or agent speech
                        // by comparing the track owner to our local identity.
                        // The agent sends transcriptions on behalf of the user's
                        // track (user STT) and on its own track (agent response).
                        let is_user = participant
                            .as_ref()
                            .map(|p| p.identity().to_string() == local_identity)
                            .unwrap_or(false);

                        for seg in segments {
                            let event = if is_user {
                                LkEvent::UserTranscription {
                                    text: seg.text.clone(),
                                    is_final: seg.r#final,
                                }
                            } else {
                                LkEvent::AgentTranscription {
                                    text: seg.text.clone(),
                                    is_final: seg.r#final,
                                }
                            };
                            let _ = tx.send(event);
                        }
                    }
                    RoomEvent::ChatMessage {
                        message,
                        participant,
                    } => {
                        let sender = participant
                            .map(|p| p.identity().to_string())
                            .unwrap_or_else(|| "unknown".into());
                        let _ = tx.send(LkEvent::ChatMessage {
                            sender,
                            message: message.message,
                        });
                    }
                    RoomEvent::Disconnected { .. } => {
                        let _ = tx.send(LkEvent::Disconnected);
                        break;
                    }
                    _ => {}
                }
            }
        });

        rx
    }
}
