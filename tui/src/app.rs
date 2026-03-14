use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::DefaultTerminal;
use tokio::sync::mpsc;

use crate::audio::{self, AudioHandle};
use crate::livekit_client::{LkClient, LkEvent};
use crate::token;
use crate::ui;
use crate::Args;

#[derive(Debug, Clone, PartialEq)]
pub enum InputMode {
    Text,
    Voice,
}

#[derive(Debug, Clone)]
pub enum ChatEntry {
    Agent(String),
    AgentPartial(String),
    User(String),
    UserPartial(String),
    System(String),
}

pub struct App {
    pub input_mode: InputMode,
    pub input_buf: String,
    pub chat_history: Vec<ChatEntry>,
    pub connected: bool,
    pub mic_muted: bool,
    pub mic_levels: Vec<f32>,
    pub agent_levels: Vec<f32>,
    pub waiting_for_agent: bool,
    pub tick_count: usize,
    room: Option<Arc<livekit::Room>>,
    audio_handle: Arc<AudioHandle>,
    /// Accumulates partial transcription text before final
    agent_partial: String,
    user_partial: String,
}

impl App {
    fn new(audio_handle: Arc<AudioHandle>) -> Self {
        Self {
            input_mode: InputMode::Text,
            input_buf: String::new(),
            chat_history: vec![ChatEntry::System(
                "Starting local-voice-ai TUI...".into(),
            )],
            connected: false,
            mic_muted: true,
            mic_levels: vec![0.0; 32],
            agent_levels: vec![0.0; 32],
            waiting_for_agent: false,
            tick_count: 0,
            room: None,
            audio_handle,
            agent_partial: String::new(),
            user_partial: String::new(),
        }
    }

    fn toggle_mode(&mut self) {
        match self.input_mode {
            InputMode::Text => {
                self.input_mode = InputMode::Voice;
                self.mic_muted = false;
                self.audio_handle.set_mic_muted(false);
                self.chat_history
                    .push(ChatEntry::System("Switched to voice mode".into()));
            }
            InputMode::Voice => {
                self.input_mode = InputMode::Text;
                self.mic_muted = true;
                self.audio_handle.set_mic_muted(true);
                self.chat_history
                    .push(ChatEntry::System("Switched to text mode".into()));
            }
        }
        self.auto_scroll();
    }

    fn toggle_mic(&mut self) {
        self.mic_muted = !self.mic_muted;
        self.audio_handle.set_mic_muted(self.mic_muted);
    }

    fn auto_scroll(&mut self) {
        // Scroll is now computed at render time in ui.rs
    }

    async fn send_message(&mut self) {
        let text = self.input_buf.trim().to_string();
        if text.is_empty() {
            return;
        }
        self.input_buf.clear();

        self.chat_history.push(ChatEntry::User(text.clone()));
        self.waiting_for_agent = true;
        self.auto_scroll();

        if let Some(room) = &self.room {
            let opts = livekit::StreamTextOptions {
                topic: "lk.chat".into(),
                ..Default::default()
            };
            if let Err(e) = room
                .local_participant()
                .send_text(&text, opts)
                .await
            {
                log::error!("Failed to send text: {}", e);
                self.chat_history
                    .push(ChatEntry::System(format!("Send failed: {}", e)));
            }
        }
    }

    fn handle_lk_event(&mut self, event: LkEvent) {
        match event {
            LkEvent::Connected { room_name } => {
                self.connected = true;
                self.chat_history.push(ChatEntry::System(format!(
                    "Connected to room: {}",
                    room_name
                )));
                self.auto_scroll();
            }
            LkEvent::UserTranscription { text, is_final } => {
                if is_final {
                    if let Some(pos) = self.chat_history.iter().rposition(|e| {
                        matches!(e, ChatEntry::UserPartial(_))
                    }) {
                        self.chat_history.remove(pos);
                    }
                    self.user_partial.clear();
                    if !text.trim().is_empty() {
                        self.chat_history.push(ChatEntry::User(text));
                        self.waiting_for_agent = true;
                    }
                } else {
                    self.user_partial = text.clone();
                    if let Some(pos) = self.chat_history.iter().rposition(|e| {
                        matches!(e, ChatEntry::UserPartial(_))
                    }) {
                        self.chat_history[pos] = ChatEntry::UserPartial(text);
                    } else {
                        self.chat_history.push(ChatEntry::UserPartial(text));
                    }
                }
                self.auto_scroll();
            }
            LkEvent::AgentTranscription { text, is_final } => {
                self.waiting_for_agent = false;
                if is_final {
                    // Remove partial entry if present
                    if !self.agent_partial.is_empty() {
                        // Remove the last AgentPartial
                        if let Some(pos) = self.chat_history.iter().rposition(|e| {
                            matches!(e, ChatEntry::AgentPartial(_))
                        }) {
                            self.chat_history.remove(pos);
                        }
                    }
                    self.agent_partial.clear();
                    if !text.trim().is_empty() {
                        self.chat_history.push(ChatEntry::Agent(text));
                    }
                } else {
                    self.agent_partial = text.clone();
                    // Update or add partial
                    if let Some(pos) = self.chat_history.iter().rposition(|e| {
                        matches!(e, ChatEntry::AgentPartial(_))
                    }) {
                        self.chat_history[pos] = ChatEntry::AgentPartial(text);
                    } else {
                        self.chat_history.push(ChatEntry::AgentPartial(text));
                    }
                }
                self.auto_scroll();
            }
            LkEvent::ChatMessage { sender, message } => {
                // Don't duplicate our own messages
                if sender != "tui-user" {
                    let label = if sender.contains("agent") || sender.starts_with("agent") {
                        ChatEntry::Agent(message)
                    } else {
                        ChatEntry::System(format!("{}: {}", sender, message))
                    };
                    self.chat_history.push(label);
                    self.auto_scroll();
                }
            }
            LkEvent::Disconnected => {
                self.connected = false;
                self.chat_history
                    .push(ChatEntry::System("Disconnected".into()));
                self.auto_scroll();
            }
            LkEvent::Error(msg) => {
                self.chat_history
                    .push(ChatEntry::System(format!("Error: {}", msg)));
                self.auto_scroll();
            }
        }
    }
}

pub async fn run(args: Args) -> anyhow::Result<()> {
    // Set up audio
    let audio_handle = AudioHandle::new();
    let (_input_stream, _output_stream) = audio::start_audio_streams(audio_handle.clone())?;

    // Get connection details
    let (url, token_str) = if let Some(ref frontend_url) = args.frontend_url {
        let details = token::fetch_connection_details(frontend_url).await?;
        (details.server_url, details.participant_token)
    } else {
        let room_name = format!("tui-room-{}", rand_id());
        let token_str =
            token::generate_token(&args.api_key, &args.api_secret, &room_name, "tui-user")?;
        (args.url.clone(), token_str)
    };

    // Connect to LiveKit
    let (client, room) = LkClient::connect(&url, &token_str).await?;
    let lk_rx = client.spawn_event_loop(audio_handle.clone());

    // Set up terminal
    let mut terminal = ratatui::init();
    let result = run_loop(&mut terminal, room, audio_handle, lk_rx).await;
    ratatui::restore();
    result
}

async fn run_loop(
    terminal: &mut DefaultTerminal,
    room: Arc<livekit::Room>,
    audio_handle: Arc<AudioHandle>,
    mut lk_rx: mpsc::UnboundedReceiver<LkEvent>,
) -> anyhow::Result<()> {
    let mut app = App::new(audio_handle.clone());
    app.room = Some(room.clone());

    // Ticker for UI refresh + audio flush
    let mut tick_interval = tokio::time::interval(Duration::from_millis(50));

    loop {
        // Draw
        terminal.draw(|f| ui::draw(f, &app))?;

        tokio::select! {
            // Terminal events
            _ = tokio::task::spawn_blocking(|| event::poll(Duration::from_millis(16))) => {
                if event::poll(Duration::from_millis(0))? {
                    if let Event::Key(key) = event::read()? {
                        if handle_key(&mut app, key).await {
                            break;
                        }
                    }
                }
            }

            // LiveKit events
            Some(ev) = lk_rx.recv() => {
                app.handle_lk_event(ev);
            }

            // Tick: flush audio, update visualizer
            _ = tick_interval.tick() => {
                audio_handle.flush_capture().await;
                app.mic_levels = audio_handle.mic_levels();
                app.agent_levels = audio_handle.agent_levels();
                app.tick_count = app.tick_count.wrapping_add(1);
            }
        }
    }

    // Cleanup
    if let Err(e) = room.close().await {
        log::error!("Error closing room: {}", e);
    }

    Ok(())
}

/// Returns true if the app should quit.
async fn handle_key(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return true,
        KeyCode::Char('t') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.toggle_mode();
        }
        KeyCode::Char('m') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.toggle_mic();
        }
        _ if app.input_mode == InputMode::Text => match key.code {
            KeyCode::Enter => {
                app.send_message().await;
            }
            KeyCode::Char(c) => {
                app.input_buf.push(c);
            }
            KeyCode::Backspace => {
                app.input_buf.pop();
            }
            KeyCode::Esc => return true,
            _ => {}
        },
        _ => {} // Voice mode: keys are ignored (mic is active)
    }
    false
}

fn rand_id() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos()
        % 10000
}
