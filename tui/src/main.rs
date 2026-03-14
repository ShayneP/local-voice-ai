mod app;
mod audio;
mod livekit_client;
mod token;
mod ui;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "local-voice-ai-tui", about = "Terminal UI for local-voice-ai")]
pub struct Args {
    /// LiveKit server URL (ws://...)
    #[arg(long, default_value = "ws://localhost:7880")]
    url: String,

    /// LiveKit API key (for local token generation)
    #[arg(long, default_value = "devkey")]
    api_key: String,

    /// LiveKit API secret (for local token generation)
    #[arg(long, default_value = "secret")]
    api_secret: String,

    /// Frontend URL to fetch connection details from (alternative to api-key/secret)
    #[arg(long)]
    frontend_url: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp(None)
        .init();

    let args = Args::parse();
    app::run(args).await
}
