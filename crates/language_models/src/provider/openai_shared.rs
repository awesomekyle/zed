//! Shared utilities for OpenAI-based providers to reduce code duplication

use anyhow::{Context as _, Result, anyhow};
use credentials_provider::CredentialsProvider;
use editor::{Editor, EditorElement, EditorStyle};
use futures::{FutureExt, future::BoxFuture};
use gpui::{
    AnyView, App, AsyncApp, Context, Entity, FontStyle, Subscription, Task, TextStyle, WhiteSpace, Window,
};
use http_client::HttpClient;
use language_model::{
    AuthenticateError, LanguageModel, LanguageModelCompletionError, LanguageModelCompletionEvent,
    LanguageModelId, LanguageModelName, LanguageModelProviderId,
    LanguageModelProviderName, LanguageModelRequest, LanguageModelToolChoice, RateLimiter,
};
use open_ai::{Model, ResponseStreamEvent, stream_completion};
use settings::{Settings, SettingsStore};
use std::sync::Arc;
use theme::ThemeSettings;
use ui::{Icon, IconName, prelude::*};
use menu;
use util::ResultExt;

use crate::{AllLanguageModelSettings, ui::InstructionListItem};

/// Shared state implementation for OpenAI-based providers
pub struct SharedOpenAiState {
    api_key: Option<String>,
    api_key_from_env: bool,
    _subscription: Subscription,
}

impl SharedOpenAiState {
    pub fn new(cx: &mut Context<Self>) -> Self {
        Self {
            api_key: None,
            api_key_from_env: false,
            _subscription: cx.observe_global::<SettingsStore>(|_this: &mut Self, cx| {
                cx.notify();
            }),
        }
    }

    pub fn is_authenticated(&self) -> bool {
        self.api_key.is_some()
    }

    pub fn api_key_from_env(&self) -> bool {
        self.api_key_from_env
    }

    /// Reset API key with custom API URL and credentials handling
    pub fn reset_api_key(
        &self,
        api_url_getter: impl Fn(&AllLanguageModelSettings) -> String + Send + 'static,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let credentials_provider = <dyn CredentialsProvider>::global(cx);
        let api_url = api_url_getter(&AllLanguageModelSettings::get_global(cx));
        cx.spawn(async move |this, cx| {
            credentials_provider
                .delete_credentials(&api_url, &cx)
                .await
                .log_err();
            this.update(cx, |this, cx| {
                this.api_key = None;
                this.api_key_from_env = false;
                cx.notify();
            })
        })
    }

    /// Set API key with custom API URL and credentials handling
    pub fn set_api_key(
        &mut self,
        api_key: String,
        api_url_getter: impl Fn(&AllLanguageModelSettings) -> String + Send + 'static,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let credentials_provider = <dyn CredentialsProvider>::global(cx);
        let api_url = api_url_getter(&AllLanguageModelSettings::get_global(cx));
        cx.spawn(async move |this, cx| {
            credentials_provider
                .write_credentials(&api_url, "Bearer", api_key.as_bytes(), &cx)
                .await
                .log_err();
            this.update(cx, |this, cx| {
                this.api_key = Some(api_key);
                cx.notify();
            })
        })
    }

    /// Authenticate with custom parameters
    pub fn authenticate(
        &self,
        api_url_getter: impl Fn(&AllLanguageModelSettings) -> String + Send + 'static,
        env_var: &str,
        provider_name: &str,
        cx: &mut Context<Self>,
    ) -> Task<Result<(), AuthenticateError>> {
        if self.is_authenticated() {
            return Task::ready(Ok(()));
        }

        let credentials_provider = <dyn CredentialsProvider>::global(cx);
        let api_url = api_url_getter(&AllLanguageModelSettings::get_global(cx));
        let env_var = env_var.to_string();
        let provider_name = provider_name.to_string();
        
        cx.spawn(async move |this, cx| {
            let (api_key, from_env) = if let Ok(api_key) = std::env::var(&env_var) {
                (api_key, true)
            } else {
                let (_, api_key) = credentials_provider
                    .read_credentials(&api_url, &cx)
                    .await?
                    .ok_or(AuthenticateError::CredentialsNotFound)?;
                (
                    String::from_utf8(api_key).context(format!("invalid {} API key", provider_name))?,
                    false,
                )
            };
            this.update(cx, |this, cx| {
                this.api_key = Some(api_key);
                this.api_key_from_env = from_env;
                cx.notify();
            })?;

            Ok(())
        })
    }

    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }
}

/// Configuration for a shared OpenAI configuration view
pub struct SharedConfigurationViewConfig {
    pub provider_name: String,
    pub env_var: String,
    pub placeholder: String,
    pub label: String,
    pub instructions: Vec<InstructionListItem>,
    pub footer_labels: Vec<String>,
}

/// Shared configuration view implementation
pub struct SharedConfigurationView {
    api_key_editor: Entity<Editor>,
    state: Entity<SharedOpenAiState>,
    load_credentials_task: Option<Task<()>>,
    config: SharedConfigurationViewConfig,
}

impl SharedConfigurationView {
    pub fn new<F>(
        state: Entity<SharedOpenAiState>,
        config: SharedConfigurationViewConfig,
        api_url_getter: F,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self 
    where
        F: Fn(&AllLanguageModelSettings) -> String + Send + 'static + Clone,
    {
        let api_key_editor = cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_placeholder_text(&config.placeholder, cx);
            editor
        });

        cx.observe(&state, |_, _, cx| {
            cx.notify();
        })
        .detach();

        let load_credentials_task = Some(cx.spawn_in(window, {
            let state = state.clone();
            let api_url_getter_clone = api_url_getter.clone();
            let env_var = config.env_var.clone();
            let provider_name = config.provider_name.clone();
            async move |this, cx| {
                if let Some(task) = state
                    .update(cx, |state, cx| {
                        state.authenticate(api_url_getter_clone, &env_var, &provider_name, cx)
                    })
                    .log_err()
                {
                    // We don't log an error, because "not signed in" is also an error.
                    let _ = task.await;
                }

                this.update(cx, |this, cx| {
                    this.load_credentials_task = None;
                    cx.notify();
                })
                .log_err();
            }
        }));

        Self {
            api_key_editor,
            state,
            load_credentials_task,
            config,
        }
    }

    pub fn save_api_key<F>(
        &mut self,
        _: &menu::Confirm,
        api_url_getter: F,
        window: &mut Window,
        cx: &mut Context<Self>,
    )
    where
        F: Fn(&AllLanguageModelSettings) -> String + Send + 'static,
    {
        let api_key = self.api_key_editor.read(cx).text(cx);
        if api_key.is_empty() {
            return;
        }

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state
                .update(cx, |state, cx| state.set_api_key(api_key, api_url_getter, cx))?
                .await
        })
        .detach_and_log_err(cx);

        cx.notify();
    }

    pub fn reset_api_key<F>(
        &mut self,
        api_url_getter: F,
        window: &mut Window,
        cx: &mut Context<Self>,
    )
    where
        F: Fn(&AllLanguageModelSettings) -> String + Send + 'static,
    {
        self.api_key_editor
            .update(cx, |editor, cx| editor.set_text("", window, cx));

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state.update(cx, |state, cx| state.reset_api_key(api_url_getter, cx))?.await
        })
        .detach_and_log_err(cx);

        cx.notify();
    }

    fn render_api_key_editor(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let settings = ThemeSettings::get_global(cx);
        let text_style = TextStyle {
            color: cx.theme().colors().text,
            font_family: settings.ui_font.family.clone(),
            font_features: settings.ui_font.features.clone(),
            font_fallbacks: settings.ui_font.fallbacks.clone(),
            font_size: rems(0.875).into(),
            font_weight: settings.ui_font.weight,
            font_style: FontStyle::Normal,
            line_height: relative(1.3),
            white_space: WhiteSpace::Normal,
            ..Default::default()
        };
        EditorElement::new(
            &self.api_key_editor,
            EditorStyle {
                background: cx.theme().colors().editor_background,
                local_player: cx.theme().players().local(),
                text: text_style,
                ..Default::default()
            },
        )
    }

    fn should_render_editor(&self, cx: &mut Context<Self>) -> bool {
        !self.state.read(cx).is_authenticated()
    }
}

impl Render for SharedConfigurationView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let env_var_set = self.state.read(cx).api_key_from_env();
        let env_var = &self.config.env_var;

        if self.load_credentials_task.is_some() {
            div().child(Label::new("Loading credentials...")).into_any()
        } else if self.should_render_editor(cx) {
            let mut v_flex = v_flex()
                .size_full()
                .child(Label::new(self.config.label.clone()));

            // Add instruction list if provided
            if !self.config.instructions.is_empty() {
                let mut list = ui::List::new();
                for instruction in &self.config.instructions {
                    list = list.child(instruction.clone());
                }
                v_flex = v_flex.child(list);
            }

            // Add API key editor
            v_flex = v_flex.child(
                h_flex()
                    .w_full()
                    .my_2()
                    .px_2()
                    .py_1()
                    .bg(cx.theme().colors().editor_background)
                    .border_1()
                    .border_color(cx.theme().colors().border)
                    .rounded_sm()
                    .child(self.render_api_key_editor(cx)),
            );

            // Add environment variable hint
            v_flex = v_flex.child(
                Label::new(
                    format!("You can also assign the {env_var} environment variable and restart Zed."),
                )
                .size(LabelSize::Small).color(Color::Muted),
            );

            // Add footer labels
            for footer_text in &self.config.footer_labels {
                v_flex = v_flex.child(
                    Label::new(footer_text.clone())
                        .size(LabelSize::Small)
                        .color(Color::Muted)
                );
            }

            v_flex.into_any()
        } else {
            h_flex()
                .mt_1()
                .p_1()
                .justify_between()
                .rounded_md()
                .border_1()
                .border_color(cx.theme().colors().border)
                .bg(cx.theme().colors().background)
                .child(
                    h_flex()
                        .gap_1()
                        .child(Icon::new(IconName::Check).color(Color::Success))
                        .child(Label::new(if env_var_set {
                            format!("API key set in {env_var} environment variable.")
                        } else {
                            "API key configured.".to_string()
                        })),
                )
                .child(
                    Button::new("reset-key", "Reset Key")
                        .label_size(LabelSize::Small)
                        .icon(Some(IconName::Trash))
                        .icon_size(IconSize::Small)
                        .icon_position(IconPosition::Start)
                        .disabled(env_var_set)
                        .when(env_var_set, |this| {
                            this.tooltip(Tooltip::text(format!("To reset your API key, unset the {env_var} environment variable.")))
                        }),
                        // Note: on_click handlers will be added by the implementing provider
                )
                .into_any()
        }
    }
}

/// Shared language model implementation for reducing duplication
/// This contains all the common logic but delegates to provider-specific parts
pub struct SharedOpenAiLanguageModel {
    id: LanguageModelId,
    pub model: Model,
    state: Entity<SharedOpenAiState>,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
    provider_id: LanguageModelProviderId,
    provider_name: LanguageModelProviderName,
}

impl SharedOpenAiLanguageModel {
    pub fn new(
        model: Model,
        state: Entity<SharedOpenAiState>,
        http_client: Arc<dyn HttpClient>,
        provider_id: &str,
        provider_name: &str,
    ) -> Self {
        Self {
            id: LanguageModelId::from(model.id().to_string()),
            model,
            state,
            http_client,
            request_limiter: RateLimiter::new(4),
            provider_id: LanguageModelProviderId(provider_id.into()),
            provider_name: LanguageModelProviderName(provider_name.into()),
        }
    }

    /// Stream completion with custom API URL getter
    pub fn stream_completion<F>(
        &self,
        request: open_ai::Request,
        api_url_getter: F,
        cx: &AsyncApp,
    ) -> BoxFuture<'static, Result<futures::stream::BoxStream<'static, Result<ResponseStreamEvent>>>>
    where
        F: Fn(&AllLanguageModelSettings) -> String + Send + 'static,
    {
        let http_client = self.http_client.clone();
        let Ok((api_key, api_url)) = cx.read_entity(&self.state, |state, cx| {
            let settings = &AllLanguageModelSettings::get_global(cx);
            (state.api_key.clone(), api_url_getter(settings))
        }) else {
            return futures::future::ready(Err(anyhow!("App state dropped"))).boxed();
        };

        let provider_name = self.provider_name.0.to_string();
        let future = self.request_limiter.stream(async move {
            let api_key = api_key.context(format!("Missing {} API Key", provider_name))?;
            let request = stream_completion(http_client.as_ref(), &api_url, &api_key, request);
            let response = request.await?;
            Ok(response)
        });

        async move { Ok(future.await?.boxed()) }.boxed()
    }

    // Delegate methods to reduce duplication
    pub fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    pub fn name(&self) -> LanguageModelName {
        LanguageModelName::from(self.model.display_name().to_string())
    }

    pub fn provider_id(&self) -> LanguageModelProviderId {
        self.provider_id.clone()
    }

    pub fn provider_name(&self) -> LanguageModelProviderName {
        self.provider_name.clone()
    }

    pub fn supports_tools(&self) -> bool {
        true
    }

    pub fn supports_images(&self) -> bool {
        false
    }

    pub fn supports_tool_choice(&self, choice: LanguageModelToolChoice) -> bool {
        match choice {
            LanguageModelToolChoice::Auto => true,
            LanguageModelToolChoice::Any => true,
            LanguageModelToolChoice::None => true,
        }
    }

    pub fn telemetry_id(&self) -> String {
        format!("{}/{}", self.provider_id.0, self.model.id())
    }

    pub fn max_token_count(&self) -> usize {
        self.model.max_token_count()
    }

    pub fn max_output_tokens(&self) -> Option<u32> {
        self.model.max_output_tokens()
    }
}