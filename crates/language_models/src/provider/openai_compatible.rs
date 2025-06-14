use anyhow::{Context as _, Result, anyhow};
use collections::{BTreeMap, HashMap};
use credentials_provider::CredentialsProvider;
use editor::{Editor, EditorElement, EditorStyle};
use futures::Stream;
use futures::{FutureExt, StreamExt, future::BoxFuture};
use gpui::{
    AnyView, App, AsyncApp, Context, Entity, FontStyle, Subscription, Task, TextStyle, WhiteSpace, Window,
};
use http_client::HttpClient;
use language_model::{
    AuthenticateError, LanguageModel, LanguageModelCompletionError, LanguageModelCompletionEvent,
    LanguageModelId, LanguageModelName, LanguageModelProvider, LanguageModelProviderId,
    LanguageModelProviderName, LanguageModelProviderState, LanguageModelRequest,
    LanguageModelToolChoice, LanguageModelToolResultContent, LanguageModelToolUse, MessageContent,
    RateLimiter, Role, StopReason,
};
use menu;
use open_ai::{ImageUrl, Model, ResponseStreamEvent, stream_completion};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use settings::{Settings, SettingsStore};
use std::pin::Pin;
use std::str::FromStr as _;
use std::sync::Arc;
use strum::IntoEnumIterator;
use theme::ThemeSettings;
use ui::{Icon, IconName, List, Tooltip, prelude::*};
use util::ResultExt;

use crate::{AllLanguageModelSettings, ui::InstructionListItem};
use crate::provider::open_ai::{OpenAiEventMapper, into_open_ai, count_open_ai_tokens};
use crate::provider::openai_shared::{SharedOpenAiState, SharedConfigurationView, SharedConfigurationViewConfig, SharedOpenAiLanguageModel};

const PROVIDER_ID: &str = "openai-compatible";
const PROVIDER_NAME: &str = "OpenAI Compatible";

#[derive(Clone, Debug, PartialEq)]
pub struct OpenAiCompatibleSettings {
    pub api_url: String,
    pub available_models: Vec<AvailableModel>,
}

impl Default for OpenAiCompatibleSettings {
    fn default() -> Self {
        Self {
            api_url: "https://your-openai-compatible-service.com/v1".to_string(),
            available_models: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AvailableModel {
    pub name: String,
    pub display_name: Option<String>,
    pub max_tokens: usize,
    pub max_output_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
}

pub struct OpenAiCompatibleLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: Entity<SharedOpenAiState>,
}

impl OpenAiCompatibleLanguageModelProvider {
    pub fn new(http_client: Arc<dyn HttpClient>, cx: &mut App) -> Self {
        let state = cx.new(|cx| SharedOpenAiState::new(cx));

        Self { http_client, state }
    }

    fn create_language_model(&self, model: open_ai::Model) -> Arc<dyn LanguageModel> {
        Arc::new(OpenAiCompatibleLanguageModel {
            shared: SharedOpenAiLanguageModel::new(
                model,
                self.state.clone(),
                self.http_client.clone(),
                PROVIDER_ID,
                PROVIDER_NAME,
            ),
        })
    }
}

const OPENAI_COMPATIBLE_API_KEY_VAR: &str = "OPENAI_COMPATIBLE_API_KEY";

impl LanguageModelProviderState for OpenAiCompatibleLanguageModelProvider {
    type ObservableEntity = SharedOpenAiState;

    fn observable_entity(&self) -> Option<gpui::Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for OpenAiCompatibleLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        LanguageModelProviderId(PROVIDER_ID.into())
    }

    fn name(&self) -> LanguageModelProviderName {
        LanguageModelProviderName(PROVIDER_NAME.into())
    }

    fn icon(&self) -> IconName {
        IconName::AiOpenAi
    }

    fn default_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        // Use a custom model since we don't know what the compatible API provides
        Some(self.create_language_model(open_ai::Model::Custom {
            name: "default".to_string(),
            display_name: Some("Default".to_string()),
            max_tokens: 4096,
            max_output_tokens: None,
            max_completion_tokens: None,
        }))
    }

    fn default_fast_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        // Use a custom model since we don't know what the compatible API provides
        Some(self.create_language_model(open_ai::Model::Custom {
            name: "default".to_string(),
            display_name: Some("Default".to_string()),
            max_tokens: 4096,
            max_output_tokens: None,
            max_completion_tokens: None,
        }))
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        let mut models = BTreeMap::default();

        // Use available models from settings - no base models since this is for custom APIs
        for model in &AllLanguageModelSettings::get_global(cx)
            .openai_compatible
            .available_models
        {
            models.insert(
                model.name.clone(),
                open_ai::Model::Custom {
                    name: model.name.clone(),
                    display_name: model.display_name.clone(),
                    max_tokens: model.max_tokens,
                    max_output_tokens: model.max_output_tokens,
                    max_completion_tokens: model.max_completion_tokens,
                },
            );
        }

        // If no models are configured, provide a default
        if models.is_empty() {
            models.insert(
                "default".to_string(),
                open_ai::Model::Custom {
                    name: "default".to_string(),
                    display_name: Some("Default".to_string()),
                    max_tokens: 4096,
                    max_output_tokens: None,
                    max_completion_tokens: None,
                },
            );
        }

        models
            .into_values()
            .map(|model| self.create_language_model(model))
            .collect()
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state.update(cx, |state, cx| {
            state.authenticate(
                |settings| settings.openai_compatible.api_url.clone(),
                OPENAI_COMPATIBLE_API_KEY_VAR,
                PROVIDER_NAME,
                cx,
            )
        })
    }

    fn configuration_view(&self, window: &mut Window, cx: &mut App) -> AnyView {
        let config = SharedConfigurationViewConfig {
            provider_name: PROVIDER_NAME.to_string(),
            env_var: OPENAI_COMPATIBLE_API_KEY_VAR.to_string(),
            placeholder: "Enter your API key".to_string(),
            label: "To use an OpenAI-compatible service with Zed, you need to configure it:".to_string(),
            instructions: vec![
                InstructionListItem::text_only(
                    "First, set the API URL in your settings.json under 'language_models.openai_compatible.api_url'",
                ),
                InstructionListItem::text_only(
                    "Then enter your API key below and hit enter",
                ),
                InstructionListItem::text_only(
                    "Configure available models in your settings if needed",
                ),
            ],
            footer_labels: vec![],
        };
        
        cx.new(|cx| ConfigurationView::new(self.state.clone(), config, window, cx))
            .into()
    }

    fn reset_credentials(&self, cx: &mut App) -> Task<Result<()>> {
        self.state.update(cx, |state, cx| {
            state.reset_api_key(
                |settings| settings.openai_compatible.api_url.clone(),
                cx,
            )
        })
    }
}

pub struct OpenAiCompatibleLanguageModel {
    shared: SharedOpenAiLanguageModel,
}

impl OpenAiCompatibleLanguageModel {
    fn stream_completion(
        &self,
        request: open_ai::Request,
        cx: &AsyncApp,
    ) -> BoxFuture<'static, Result<futures::stream::BoxStream<'static, Result<ResponseStreamEvent>>>> {
        self.shared.stream_completion(
            request,
            |settings| settings.openai_compatible.api_url.clone(),
            cx,
        )
    }
}

impl LanguageModel for OpenAiCompatibleLanguageModel {
    fn id(&self) -> LanguageModelId {
        self.shared.id()
    }

    fn name(&self) -> LanguageModelName {
        self.shared.name()
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        self.shared.provider_id()
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        self.shared.provider_name()
    }

    fn supports_tools(&self) -> bool {
        self.shared.supports_tools()
    }

    fn supports_images(&self) -> bool {
        self.shared.supports_images()
    }

    fn supports_tool_choice(&self, choice: LanguageModelToolChoice) -> bool {
        self.shared.supports_tool_choice(choice)
    }

    fn telemetry_id(&self) -> String {
        self.shared.telemetry_id()
    }

    fn max_token_count(&self) -> usize {
        self.shared.max_token_count()
    }

    fn max_output_tokens(&self) -> Option<u32> {
        self.shared.max_output_tokens()
    }

    fn count_tokens(
        &self,
        request: LanguageModelRequest,
        cx: &App,
    ) -> BoxFuture<'static, Result<usize>> {
        count_open_ai_tokens(request, self.shared.model.clone(), cx)
    }

    fn stream_completion(
        &self,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            futures::stream::BoxStream<
                'static,
                Result<LanguageModelCompletionEvent, LanguageModelCompletionError>,
            >,
            LanguageModelCompletionError,
        >,
    > {
        let request = into_open_ai(request, &self.shared.model, self.max_output_tokens());
        let completions = self.stream_completion(request, cx);
        async move {
            let mapper = OpenAiEventMapper::new();
            Ok(mapper.map_stream(completions.await?).boxed())
        }
        .boxed()
    }
}

struct ConfigurationView {
    shared: SharedConfigurationView,
}

impl ConfigurationView {
    fn new(state: Entity<SharedOpenAiState>, config: SharedConfigurationViewConfig, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let shared = SharedConfigurationView::new(
            state,
            config,
            |settings| settings.openai_compatible.api_url.clone(),
            window,
            cx,
        );

        Self { shared }
    }
}

impl Render for ConfigurationView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Simple delegation to the shared implementation
        self.shared.render(window, cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_constants() {
        assert_eq!(PROVIDER_ID, "openai-compatible");
        assert_eq!(PROVIDER_NAME, "OpenAI Compatible");
    }

    #[test]
    fn test_default_settings() {
        let settings = OpenAiCompatibleSettings::default();
        assert_eq!(settings.api_url, "https://your-openai-compatible-service.com/v1");
        assert!(settings.available_models.is_empty());
    }

    #[test]
    fn test_available_model_serialization() {
        let model = AvailableModel {
            name: "test-model".to_string(),
            display_name: Some("Test Model".to_string()),
            max_tokens: 4096,
            max_output_tokens: Some(1024),
            max_completion_tokens: Some(1024),
        };

        // Test that the model can be serialized and deserialized
        let json = serde_json::to_string(&model).unwrap();
        let deserialized: AvailableModel = serde_json::from_str(&json).unwrap();
        assert_eq!(model, deserialized);
    }
}