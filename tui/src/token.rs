use livekit_api::access_token::{AccessToken, VideoGrants};
use serde::Deserialize;

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct ConnectionDetails {
    #[serde(rename = "serverUrl")]
    pub server_url: String,
    #[serde(rename = "roomName")]
    pub room_name: String,
    #[serde(rename = "participantToken")]
    pub participant_token: String,
    #[serde(rename = "participantName")]
    pub participant_name: String,
}

/// Fetch connection details from the frontend API.
pub async fn fetch_connection_details(frontend_url: &str) -> anyhow::Result<ConnectionDetails> {
    let url = format!("{}/api/connection-details", frontend_url.trim_end_matches('/'));
    let body = serde_json::json!({});
    let resp = reqwest::Client::new()
        .post(&url)
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json::<ConnectionDetails>()
        .await?;
    Ok(resp)
}

/// Generate a token locally using API key + secret.
pub fn generate_token(
    api_key: &str,
    api_secret: &str,
    room_name: &str,
    identity: &str,
) -> anyhow::Result<String> {
    let token = AccessToken::with_api_key(api_key, api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(VideoGrants {
            room_join: true,
            room: room_name.into(),
            can_publish: true,
            can_publish_data: true,
            can_subscribe: true,
            ..Default::default()
        })
        .to_jwt()?;
    Ok(token)
}
