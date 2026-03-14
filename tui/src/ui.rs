use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::app::{App, ChatEntry, InputMode};

const BAR_CHARS: &[char] = &[' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

pub fn draw(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Min(6),    // chat
            Constraint::Length(5), // visualizer
            Constraint::Length(3), // input
        ])
        .split(f.area());

    draw_header(f, app, chunks[0]);
    draw_chat(f, app, chunks[1]);
    draw_visualizer(f, app, chunks[2]);
    draw_input(f, app, chunks[3]);
}

fn draw_header(f: &mut Frame, app: &App, area: Rect) {
    let mode_label = match app.input_mode {
        InputMode::Text => Span::styled(" TEXT ", Style::default().fg(Color::Black).bg(Color::Cyan)),
        InputMode::Voice => {
            Span::styled(" VOICE ", Style::default().fg(Color::Black).bg(Color::Magenta))
        }
    };

    let status = if app.connected {
        Span::styled(" ● Connected ", Style::default().fg(Color::Green))
    } else {
        Span::styled(" ○ Connecting... ", Style::default().fg(Color::Yellow))
    };

    let title_line = Line::from(vec![
        Span::styled(" local-voice-ai ", Style::default().fg(Color::White).bold()),
        status,
        Span::raw("  "),
        mode_label,
    ]);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    let header = Paragraph::new(title_line).block(block);
    f.render_widget(header, area);
}

fn draw_chat(f: &mut Frame, app: &App, area: Rect) {
    let mut lines: Vec<Line> = Vec::new();

    for entry in &app.chat_history {
        let (prefix, style, content) = match entry {
            ChatEntry::Agent(t) => (
                "  Agent: ",
                Style::default().fg(Color::Cyan),
                t.as_str(),
            ),
            ChatEntry::AgentPartial(t) => (
                "  Agent: ",
                Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
                t.as_str(),
            ),
            ChatEntry::User(t) => (
                "    You: ",
                Style::default().fg(Color::Green),
                t.as_str(),
            ),
            ChatEntry::UserPartial(t) => (
                "    You: ",
                Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
                t.as_str(),
            ),
            ChatEntry::System(t) => (
                "      ✦ ",
                Style::default().fg(Color::Yellow).add_modifier(Modifier::DIM),
                t.as_str(),
            ),
        };

        lines.push(Line::from(vec![
            Span::styled(prefix, style.add_modifier(Modifier::BOLD)),
            Span::styled(content, style),
        ]));
    }

    if app.waiting_for_agent {
        let dots = ".".repeat((app.tick_count / 10) % 3 + 1);
        lines.push(Line::from(vec![
            Span::styled("  Agent: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(format!("thinking{}", dots), Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC)),
        ]));
    }

    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "  Waiting for connection...",
            Style::default().fg(Color::DarkGray),
        )));
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(" Chat ", Style::default().fg(Color::White)));

    let visible_height = area.height.saturating_sub(2) as usize; // minus borders
    let total_lines = lines.len();
    let scroll = total_lines.saturating_sub(visible_height) as u16;

    let text = Text::from(lines);
    let para = Paragraph::new(text)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));

    f.render_widget(para, area);
}

fn draw_visualizer(f: &mut Frame, app: &App, area: Rect) {
    let inner_width = area.width.saturating_sub(2) as usize;
    let inner_height = area.height.saturating_sub(2) as usize;

    let agent_levels = &app.agent_levels;
    let mic_levels = &app.mic_levels;

    // Build bar visualization
    let mut lines: Vec<Line> = Vec::new();

    // Each row from top to bottom
    for row in (0..inner_height).rev() {
        let threshold = (row as f32 + 0.5) / inner_height as f32;
        let mut spans: Vec<Span> = Vec::new();

        let total_bands = agent_levels.len().min(inner_width);
        let bar_width = if total_bands > 0 {
            (inner_width / total_bands).max(1)
        } else {
            1
        };

        for i in 0..total_bands {
            let agent_val = agent_levels.get(i).copied().unwrap_or(0.0);
            let mic_val = mic_levels.get(i).copied().unwrap_or(0.0);

            let ch = if agent_val > threshold {
                let idx = ((agent_val - threshold) * (BAR_CHARS.len() - 1) as f32 * inner_height as f32)
                    .min((BAR_CHARS.len() - 1) as f32) as usize;
                let c = BAR_CHARS[idx.min(BAR_CHARS.len() - 1)];
                let bar_str: String = std::iter::repeat(c).take(bar_width).collect();
                Span::styled(bar_str, Style::default().fg(Color::Cyan))
            } else if mic_val > threshold {
                let idx = ((mic_val - threshold) * (BAR_CHARS.len() - 1) as f32 * inner_height as f32)
                    .min((BAR_CHARS.len() - 1) as f32) as usize;
                let c = BAR_CHARS[idx.min(BAR_CHARS.len() - 1)];
                let bar_str: String = std::iter::repeat(c).take(bar_width).collect();
                Span::styled(bar_str, Style::default().fg(Color::Green))
            } else {
                spans.push(Span::raw(" ".repeat(bar_width)));
                continue;
            };

            spans.push(ch);
        }

        lines.push(Line::from(spans));
    }

    let label = match app.input_mode {
        InputMode::Voice if !app.mic_muted => " Audio ◉ mic on ",
        InputMode::Voice => " Audio ○ mic off ",
        InputMode::Text => " Audio ",
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(label, Style::default().fg(Color::Magenta)));

    let para = Paragraph::new(Text::from(lines)).block(block);
    f.render_widget(para, area);
}

fn draw_input(f: &mut Frame, app: &App, area: Rect) {
    let (hint, style) = match app.input_mode {
        InputMode::Text => (
            "Type a message... (Ctrl+T: voice mode, Ctrl+C: quit)",
            Style::default().fg(Color::DarkGray),
        ),
        InputMode::Voice => (
            "Voice mode active (Ctrl+T: text mode, Ctrl+M: toggle mic, Ctrl+C: quit)",
            Style::default().fg(Color::DarkGray),
        ),
    };

    let display = if app.input_buf.is_empty() {
        Line::from(Span::styled(hint, style))
    } else {
        Line::from(Span::styled(
            format!("{}", app.input_buf),
            Style::default().fg(Color::White),
        ))
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(" > ", Style::default().fg(Color::Green)));

    let input = Paragraph::new(display).block(block);
    f.render_widget(input, area);

    // Show cursor in text mode
    if app.input_mode == InputMode::Text {
        f.set_cursor_position((
            area.x + 1 + app.input_buf.len() as u16,
            area.y + 1,
        ));
    }
}
