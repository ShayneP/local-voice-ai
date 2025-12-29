use std::collections::{HashMap, VecDeque};
use std::io;
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::execute;
use futures_util::StreamExt;
use ratatui::prelude::*;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph};
use serde::Deserialize;
use tokio::io::AsyncBufReadExt;
use tokio::process::Command;
use tokio::sync::{mpsc, watch};
use tokio::time::interval;

const DEFAULT_SERVICES: [&str; 6] = [
    "livekit",
    "whisper",
    "llama_cpp",
    "kokoro",
    "livekit_agent",
    "frontend",
];

#[derive(Parser, Debug)]
#[command(name = "local-voice-ai-tui", version, about = "Compose TUI for Local Voice AI")]
struct Cli {
    #[arg(short = 'f', long = "file", value_name = "FILE", action = clap::ArgAction::Append)]
    files: Vec<String>,

    #[arg(short = 'p', long = "project-name", value_name = "NAME")]
    project_name: Option<String>,

    #[arg(long = "max-lines", default_value_t = 800)]
    max_lines: usize,

    #[arg(long = "tail", default_value_t = 200)]
    tail: usize,

    #[arg(long = "interval-ms", default_value_t = 1000)]
    interval_ms: u64,
}

#[derive(Clone, Debug)]
struct StatusSnapshot {
    statuses: HashMap<String, ServiceStatus>,
    error: Option<String>,
    updated_at: Instant,
}

#[derive(Clone, Debug)]
struct ServiceStatus {
    state: ServiceState,
    health: Option<String>,
    status: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ServiceState {
    Running,
    Starting,
    Unhealthy,
    Stopped,
    Unknown,
}

struct ServicePane {
    name: String,
    logs: VecDeque<String>,
    paused: bool,
}

struct App {
    services: Vec<ServicePane>,
    statuses: HashMap<String, ServiceStatus>,
    selected: usize,
    max_lines: usize,
    status_error: Option<String>,
    started_at: Instant,
    status_updated_at: Option<Instant>,
}

#[derive(Debug)]
struct LogEvent {
    service: String,
    line: String,
}

#[derive(Deserialize, Debug)]
struct PsEntry {
    #[serde(rename = "Service")]
    service: Option<String>,
    #[serde(rename = "Name")]
    name: Option<String>,
    #[serde(rename = "State")]
    state: Option<String>,
    #[serde(rename = "Health")]
    health: Option<String>,
    #[serde(rename = "Status")]
    status: Option<String>,
}

impl App {
    fn new(services: Vec<String>, max_lines: usize) -> Self {
        let panes = services
            .into_iter()
            .map(|name| ServicePane {
                name,
                logs: VecDeque::new(),
                paused: false,
            })
            .collect();

        Self {
            services: panes,
            statuses: HashMap::new(),
            selected: 0,
            max_lines,
            status_error: None,
            started_at: Instant::now(),
            status_updated_at: None,
        }
    }

    fn selected_service_name(&self) -> Option<&str> {
        self.services.get(self.selected).map(|svc| svc.name.as_str())
    }

    fn select_next(&mut self) {
        if self.services.is_empty() {
            self.selected = 0;
            return;
        }
        self.selected = (self.selected + 1) % self.services.len();
    }

    fn select_prev(&mut self) {
        if self.services.is_empty() {
            self.selected = 0;
            return;
        }
        if self.selected == 0 {
            self.selected = self.services.len() - 1;
        } else {
            self.selected -= 1;
        }
    }

    fn select_first(&mut self) {
        if !self.services.is_empty() {
            self.selected = 0;
        }
    }

    fn select_last(&mut self) {
        if !self.services.is_empty() {
            self.selected = self.services.len() - 1;
        }
    }

    fn toggle_pause(&mut self) {
        if let Some(service) = self.services.get_mut(self.selected) {
            service.paused = !service.paused;
        }
    }

    fn clear_selected_logs(&mut self) {
        if let Some(service) = self.services.get_mut(self.selected) {
            service.logs.clear();
        }
    }

    fn push_log(&mut self, service: &str, line: String) {
        if let Some(pane) = self.services.iter_mut().find(|svc| svc.name == service) {
            if pane.paused {
                return;
            }
            pane.logs.push_back(line);
            while pane.logs.len() > self.max_lines {
                pane.logs.pop_front();
            }
        }
    }

    fn apply_status_snapshot(&mut self, snapshot: StatusSnapshot) {
        self.statuses = snapshot.statuses;
        self.status_error = snapshot.error;
        self.status_updated_at = Some(snapshot.updated_at);
    }

    fn running_count(&self) -> usize {
        self.services
            .iter()
            .filter(|svc| matches!(self.status_state(&svc.name), ServiceState::Running))
            .count()
    }

    fn status_state(&self, service: &str) -> ServiceState {
        self.statuses
            .get(service)
            .map(|status| status.state)
            .unwrap_or(ServiceState::Unknown)
    }

    fn status_detail(&self, service: &str) -> Option<&ServiceStatus> {
        self.statuses.get(service)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut terminal = setup_terminal()?;
    let result = run_app(&mut terminal).await;
    restore_terminal(&mut terminal)?;
    result
}

async fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    let cli = Cli::parse();
    let compose_args = build_compose_args(&cli);

    let services = fetch_services(&compose_args)
        .await
        .unwrap_or_else(|_| DEFAULT_SERVICES.iter().map(|s| s.to_string()).collect());

    let (log_tx, mut log_rx) = mpsc::channel::<LogEvent>(2000);
    let (status_tx, mut status_rx) = watch::channel(StatusSnapshot {
        statuses: HashMap::new(),
        error: None,
        updated_at: Instant::now(),
    });

    for service in services.iter().cloned() {
        let compose_args = compose_args.clone();
        let log_tx = log_tx.clone();
        tokio::spawn(stream_logs(service, compose_args, cli.tail, log_tx));
    }

    tokio::spawn(status_poller(
        compose_args.clone(),
        status_tx,
        Duration::from_millis(cli.interval_ms),
    ));

    let mut app = App::new(services, cli.max_lines);
    let mut events = EventStream::new();
    let mut tick = interval(Duration::from_millis(200));
    let mut should_quit = false;
    let mut dirty = true;

    while !should_quit {
        tokio::select! {
            _ = tick.tick() => {
                dirty = true;
            }
            maybe_event = events.next() => {
                if let Some(Ok(event)) = maybe_event {
                    if handle_event(event, &mut app, &compose_args, &log_tx).await? {
                        should_quit = true;
                    }
                    dirty = true;
                }
            }
            Some(log) = log_rx.recv() => {
                app.push_log(&log.service, log.line);
                dirty = true;
            }
            Ok(()) = status_rx.changed() => {
                let snapshot = status_rx.borrow().clone();
                app.apply_status_snapshot(snapshot);
                dirty = true;
            }
        }

        if dirty {
            terminal.draw(|frame| ui(frame, &app))?;
            dirty = false;
        }
    }

    Ok(())
}

async fn handle_event(
    event: Event,
    app: &mut App,
    compose_args: &[String],
    log_tx: &mpsc::Sender<LogEvent>,
) -> Result<bool> {
    match event {
        Event::Key(KeyEvent { code, modifiers, .. }) => {
            match (code, modifiers) {
                (KeyCode::Char('q'), _) | (KeyCode::Esc, _) => return Ok(true),
                (KeyCode::Char('c'), _) => app.clear_selected_logs(),
                (KeyCode::Char('p'), _) => app.toggle_pause(),
                (KeyCode::Char('g'), _) => app.select_first(),
                (KeyCode::Char('G'), _) => app.select_last(),
                (KeyCode::Char('j'), _)
                | (KeyCode::Down, _)
                | (KeyCode::Right, _) => app.select_next(),
                (KeyCode::Char('k'), _)
                | (KeyCode::Up, _)
                | (KeyCode::Left, _) => app.select_prev(),
                (KeyCode::Char('r'), _) => {
                    if let Some(service) = app.selected_service_name().map(str::to_string) {
                        let compose_args = compose_args.to_vec();
                        let log_tx = log_tx.clone();
                        tokio::spawn(async move {
                            let _ = log_tx
                                .send(LogEvent {
                                    service: service.clone(),
                                    line: "[tui] restart requested".to_string(),
                                })
                                .await;
                            match restart_service(&compose_args, &service).await {
                                Ok(()) => {
                                    let _ = log_tx
                                        .send(LogEvent {
                                            service,
                                            line: "[tui] restart completed".to_string(),
                                        })
                                        .await;
                                }
                                Err(err) => {
                                    let _ = log_tx
                                        .send(LogEvent {
                                            service,
                                            line: format!("[tui] restart failed: {err}"),
                                        })
                                        .await;
                                }
                            }
                        });
                    }
                }
                (KeyCode::Char('l'), KeyModifiers::CONTROL) => {
                    if let Some(service) = app.selected_service_name().map(str::to_string) {
                        let _ = log_tx
                            .send(LogEvent {
                                service: service.clone(),
                                line: "[tui] screen cleared".to_string(),
                            })
                            .await;
                        app.clear_selected_logs();
                    }
                }
                _ => {}
            }
        }
        Event::Resize(_, _) => {}
        _ => {}
    }

    Ok(false)
}

fn build_compose_args(cli: &Cli) -> Vec<String> {
    let mut args = vec!["compose".to_string()];

    if cli.files.is_empty() {
        args.push("-f".to_string());
        args.push("docker-compose.yml".to_string());
    } else {
        for file in &cli.files {
            args.push("-f".to_string());
            args.push(file.clone());
        }
    }

    if let Some(project) = &cli.project_name {
        args.push("-p".to_string());
        args.push(project.clone());
    }

    args
}

async fn fetch_services(compose_args: &[String]) -> Result<Vec<String>> {
    let output = Command::new("docker")
        .args(compose_args)
        .arg("config")
        .arg("--services")
        .output()
        .await
        .context("failed to run docker compose config")?;

    if !output.status.success() {
        return Err(anyhow!("docker compose config failed"));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let services = stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| line.to_string())
        .collect::<Vec<_>>();

    if services.is_empty() {
        return Err(anyhow!("no services discovered"));
    }

    Ok(services)
}

async fn status_poller(
    compose_args: Vec<String>,
    status_tx: watch::Sender<StatusSnapshot>,
    poll_interval: Duration,
) {
    let mut ticker = interval(poll_interval);
    loop {
        ticker.tick().await;
        let snapshot = match fetch_statuses(&compose_args).await {
            Ok(statuses) => StatusSnapshot {
                statuses,
                error: None,
                updated_at: Instant::now(),
            },
            Err(err) => StatusSnapshot {
                statuses: HashMap::new(),
                error: Some(err.to_string()),
                updated_at: Instant::now(),
            },
        };

        let _ = status_tx.send(snapshot);
    }
}

async fn fetch_statuses(compose_args: &[String]) -> Result<HashMap<String, ServiceStatus>> {
    let output = Command::new("docker")
        .args(compose_args)
        .arg("ps")
        .arg("--format")
        .arg("json")
        .output()
        .await
        .context("failed to run docker compose ps")?;

    if !output.status.success() {
        return Err(anyhow!("docker compose ps failed"));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let entries = parse_ps_entries(&stdout)?;

    let mut map = HashMap::new();
    for entry in entries {
        let service_name = entry
            .service
            .or(entry.name)
            .unwrap_or_else(|| "unknown".to_string());

        let state = entry.state.as_deref().unwrap_or("");
        let health = entry.health.clone();
        let status = entry.status.clone();

        let state = if state == "running" {
            ServiceState::Running
        } else if state.is_empty() {
            ServiceState::Unknown
        } else if state == "restarting" || state == "starting" {
            ServiceState::Starting
        } else {
            ServiceState::Stopped
        };

        map.insert(
            service_name,
            ServiceStatus {
                state,
                health,
                status,
            },
        );
    }

    Ok(map)
}

fn parse_ps_entries(output: &str) -> Result<Vec<PsEntry>> {
    let trimmed = output.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    if let Ok(entries) = serde_json::from_str::<Vec<PsEntry>>(trimmed) {
        return Ok(entries);
    }
    if let Ok(entry) = serde_json::from_str::<PsEntry>(trimmed) {
        return Ok(vec![entry]);
    }

    let mut entries = Vec::new();
    for line in trimmed.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            if let Ok(mut batch) = serde_json::from_str::<Vec<PsEntry>>(line) {
                entries.append(&mut batch);
            }
            continue;
        }
        if line.starts_with('{') {
            if let Ok(entry) = serde_json::from_str::<PsEntry>(line) {
                entries.push(entry);
            }
            continue;
        }
        if let Some(start) = line.find(|c| c == '{' || c == '[') {
            let slice = &line[start..];
            if slice.starts_with('[') {
                if let Ok(mut batch) = serde_json::from_str::<Vec<PsEntry>>(slice) {
                    entries.append(&mut batch);
                }
            } else if slice.starts_with('{') {
                if let Ok(entry) = serde_json::from_str::<PsEntry>(slice) {
                    entries.push(entry);
                }
            }
        }
    }

    if entries.is_empty() {
        return Err(anyhow!("failed to parse docker compose ps json"));
    }

    Ok(entries)
}

async fn restart_service(compose_args: &[String], service: &str) -> Result<()> {
    let output = Command::new("docker")
        .args(compose_args)
        .arg("restart")
        .arg(service)
        .output()
        .await
        .context("failed to run docker compose restart")?;

    if !output.status.success() {
        return Err(anyhow!("docker compose restart failed"));
    }

    Ok(())
}

async fn stream_logs(service: String, compose_args: Vec<String>, tail: usize, log_tx: mpsc::Sender<LogEvent>) {
    let mut had_error = false;
    loop {
        let mut cmd = Command::new("docker");
        cmd.args(&compose_args)
            .arg("logs")
            .arg("-f")
            .arg("--no-color")
            .arg("--tail")
            .arg(tail.to_string())
            .arg(&service)
            .stdout(Stdio::piped())
            .stderr(Stdio::null());

        let mut child = match cmd.spawn() {
            Ok(child) => {
                if had_error {
                    let _ = log_tx
                        .send(LogEvent {
                            service: service.clone(),
                            line: "[tui] log stream connected".to_string(),
                        })
                        .await;
                }
                had_error = false;
                child
            }
            Err(err) => {
                if !had_error {
                    let _ = log_tx
                        .send(LogEvent {
                            service: service.clone(),
                            line: format!("[tui] log stream error: {err}"),
                        })
                        .await;
                }
                had_error = true;
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }
        };

        let stdout = match child.stdout.take() {
            Some(stdout) => stdout,
            None => {
                if !had_error {
                    let _ = log_tx
                        .send(LogEvent {
                            service: service.clone(),
                            line: "[tui] log stream closed".to_string(),
                        })
                        .await;
                }
                had_error = true;
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }
        };

        let mut reader = tokio::io::BufReader::new(stdout).lines();
        while let Ok(Some(line)) = reader.next_line().await {
            let line = strip_log_prefix(&line, &service);
            let line = sanitize_log_line(&line);
            if log_tx
                .send(LogEvent {
                    service: service.clone(),
                    line,
                })
                .await
                .is_err()
            {
                return;
            }
            had_error = false;
        }

        let _ = child.wait().await;
        if !had_error {
            let _ = log_tx
                .send(LogEvent {
                    service: service.clone(),
                    line: "[tui] log stream ended, reconnecting".to_string(),
                })
                .await;
        }
        had_error = true;
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

fn strip_log_prefix(line: &str, service: &str) -> String {
    if let Some(index) = line.find('|') {
        let left = line[..index].trim_end();
        if is_compose_prefix(left, service) {
            return line[index + 1..].trim_start().to_string();
        }
    }

    line.to_string()
}

fn sanitize_log_line(line: &str) -> String {
    let stripped = strip_ansi_sequences(line);
    let mut out = String::with_capacity(stripped.len());
    for ch in stripped.chars() {
        match ch {
            '\t' => out.push_str("  "),
            ch if ch.is_control() => {}
            _ => out.push(ch),
        }
    }
    out
}

fn is_compose_prefix(left: &str, service: &str) -> bool {
    if left == service {
        return true;
    }
    if !left.starts_with(service) {
        return false;
    }
    let suffix = &left[service.len()..];
    if suffix.is_empty() {
        return true;
    }
    if !suffix.starts_with('-') {
        return false;
    }
    let rest = &suffix[1..];
    !rest.is_empty() && rest.chars().all(|ch| ch.is_ascii_digit())
}

fn strip_ansi_sequences(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
            i += 2;
            while i < bytes.len() {
                let b = bytes[i];
                if (0x40..=0x7e).contains(&b) {
                    i += 1;
                    break;
                }
                i += 1;
            }
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;
    Ok(terminal)
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

fn ui(frame: &mut Frame, app: &App) {
    let size = frame.size();

    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(3),
        ])
        .split(size);

    let header_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(vertical[0]);

    let body_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(28), Constraint::Min(10)])
        .split(vertical[1]);

    render_header(frame, header_chunks[0], header_chunks[1], app);
    render_services(frame, body_chunks[0], app);
    render_logs(frame, body_chunks[1], app);
    render_footer(frame, vertical[2], app);
}

fn render_header(frame: &mut Frame, left: Rect, right: Rect, app: &App) {
    let title = Paragraph::new(Line::from(vec![
        Span::styled(
            "LOCAL VOICE AI",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(
            "Compose Monitor",
            Style::default().fg(Color::Gray).add_modifier(Modifier::BOLD),
        ),
    ]))
    .block(Block::default().borders(Borders::ALL));

    let running = app.running_count();
    let total = app.services.len();
    let uptime = format_duration(app.started_at.elapsed());
    let summary = format!("Running {running}/{total} | Uptime {uptime}");

    let summary = Paragraph::new(summary)
        .alignment(Alignment::Right)
        .block(Block::default().borders(Borders::ALL));

    frame.render_widget(title, left);
    frame.render_widget(summary, right);
}

fn render_services(frame: &mut Frame, area: Rect, app: &App) {
    let items = app
        .services
        .iter()
        .map(|service| {
            let state = app.status_state(&service.name);
            let (marker, style) = status_marker(state);
            let mut spans = vec![Span::styled(marker, style), Span::raw(" ")];
            spans.push(Span::raw(service.name.as_str()));
            if service.paused {
                spans.push(Span::raw(" "));
                spans.push(Span::styled("P", Style::default().fg(Color::Magenta)));
            }
            ListItem::new(Line::from(spans))
        })
        .collect::<Vec<_>>();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Services"))
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("> ");

    let mut state = ListState::default();
    if !app.services.is_empty() {
        state.select(Some(app.selected));
    }

    frame.render_stateful_widget(list, area, &mut state);
}

fn render_logs(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(6)])
        .split(area);

    let service = app
        .services
        .get(app.selected)
        .map(|svc| svc.name.as_str())
        .unwrap_or("(no service)");

    let paused = app
        .services
        .get(app.selected)
        .map(|svc| svc.paused)
        .unwrap_or(false);

    let title = if paused {
        format!("Logs: {service} (paused)")
    } else {
        format!("Logs: {service}")
    };

    let block = Block::default().borders(Borders::ALL).title(title);
    let log_area = chunks[0];
    let inner_width = log_area.width.saturating_sub(2) as usize;
    let inner_height = log_area.height.saturating_sub(2) as usize;

    let mut items = Vec::new();
    if let Some(selected) = app.services.get(app.selected) {
        let start = selected.logs.len().saturating_sub(inner_height);
        for line in selected.logs.iter().skip(start) {
            items.push(ListItem::new(trim_line(line, inner_width)));
        }
    }

    frame.render_widget(Clear, log_area);
    let list = List::new(items).block(block);
    frame.render_widget(list, chunks[0]);

    render_details(frame, chunks[1], app);
}

fn render_details(frame: &mut Frame, area: Rect, app: &App) {
    let service = app
        .services
        .get(app.selected)
        .map(|svc| svc.name.as_str())
        .unwrap_or("(no service)");

    let status = app.status_detail(service);
    let state = status.map(|s| state_label(s.state)).unwrap_or("unknown");
    let health = status
        .and_then(|s| s.health.as_deref())
        .unwrap_or("-");
    let status_text = status
        .and_then(|s| s.status.as_deref())
        .unwrap_or("-");

    let updated = app
        .status_updated_at
        .map(|instant| format_duration(instant.elapsed()))
        .unwrap_or_else(|| "-".to_string());

    let key_style = Style::default().fg(Color::DarkGray);

    let lines = vec![
        Line::from(vec![Span::styled("Service: ", key_style), Span::raw(service)]),
        Line::from(vec![Span::styled("State: ", key_style), Span::raw(state)]),
        Line::from(vec![Span::styled("Health: ", key_style), Span::raw(health)]),
        Line::from(vec![Span::styled("Status: ", key_style), Span::raw(status_text)]),
        Line::from(vec![Span::styled("Last update: ", key_style), Span::raw(updated)]),
    ];

    let block = Block::default().borders(Borders::ALL).title("Details");
    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, area);
}

fn render_footer(frame: &mut Frame, area: Rect, app: &App) {
    let help = "Keys: q quit | arrows/j/k move | r restart | p pause | c clear";
    let error = app
        .status_error
        .as_ref()
        .map(|err| {
            let message = format!("Status error: {err}");
            trim_line(&message, area.width.saturating_sub(1) as usize)
        })
        .unwrap_or_else(String::new);

    let error_line = if error.is_empty() {
        Line::from("")
    } else {
        Line::from(Span::styled(error, Style::default().fg(Color::Red)))
    };
    let help_line = Line::from(Span::styled(help, Style::default().fg(Color::Gray)));

    let paragraph = Paragraph::new(vec![error_line, help_line])
        .block(Block::default().borders(Borders::TOP));
    frame.render_widget(paragraph, area);
}

fn status_marker(state: ServiceState) -> (&'static str, Style) {
    match state {
        ServiceState::Running => ("O", Style::default().fg(Color::Green)),
        ServiceState::Starting => ("~", Style::default().fg(Color::Yellow)),
        ServiceState::Unhealthy => ("!", Style::default().fg(Color::Red)),
        ServiceState::Stopped => ("-", Style::default().fg(Color::DarkGray)),
        ServiceState::Unknown => ("?", Style::default().fg(Color::Blue)),
    }
}

fn state_label(state: ServiceState) -> &'static str {
    match state {
        ServiceState::Running => "running",
        ServiceState::Starting => "starting",
        ServiceState::Unhealthy => "unhealthy",
        ServiceState::Stopped => "stopped",
        ServiceState::Unknown => "unknown",
    }
}

fn format_duration(duration: Duration) -> String {
    let total = duration.as_secs();
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let seconds = total % 60;
    if hours > 0 {
        format!("{hours:02}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes:02}:{seconds:02}")
    }
}

fn trim_line(line: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }

    let mut chars = line.chars();
    let count = line.chars().count();
    if count <= max_width {
        return line.to_string();
    }

    if max_width <= 3 {
        return chars.take(max_width).collect();
    }

    let mut out: String = chars.take(max_width - 3).collect();
    out.push_str("...");
    out
}
