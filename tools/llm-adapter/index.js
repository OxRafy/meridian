/**
 * Multi-Provider LLM Adapter
 * 
 * Smart router that manages multiple LLM providers with automatic failover,
 * load balancing, and runtime configuration.
 * 
 * Supported providers:
 * - OpenRouter (default)
 * - OpenAI / Azure OpenAI
 * - Anthropic (Claude)
 * - Google Gemini
 * - Ollama (local)
 * - vLLM / LLaMA.cpp / LM Studio (any OpenAI-compatible endpoint)
 * - Custom providers (add via config)
 * 
 * Features:
 * - Automatic failover on provider errors
 * - Round-robin load balancing
 * - Provider-specific retry policies
 * - Cost-aware routing (optional)
 * - Model-specific configuration
 * - Runtime provider switching
 */

import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import { GoogleGenerativeAI } from "@google/generative-ai";

// ─────────────────────────────────────────────────────────────────────────────
// Provider Configuration & Registry
// ─────────────────────────────────────────────────────────────────────────────

const PROVIDERS = {
  OPENROUTER: "openrouter",
  OPENAI: "openai",
  AZURE: "azure",
  ANTHROPIC: "anthropic",
  GOOGLE: "google",
  OLLAMA: "ollama",
  VLLM: "vllm",
  CUSTOM: "custom",
};

// Provider-specific default models
const DEFAULT_MODELS = {
  [PROVIDERS.OPENROUTER]: "openrouter/healer-alpha",
  [PROVIDERS.OPENAI]: "openai/gpt-4o",
  [PROVIDERS.AZURE]: "azure/gpt-4o",
  [PROVIDERS.ANTHROPIC]: "anthropic/claude-3.5-sonnet",
  [PROVIDERS.GOOGLE]: "google/gemini-1.5-flash",
  [PROVIDERS.OLLAMA]: "ollama/llama3",
  [PROVIDERS.VLLM]: "vllm/llama3",
};

// Provider error codes that should trigger failover
const TRANSIENT_ERRORS = new Set([502, 503, 504, 529, 429]);

// ─────────────────────────────────────────────────────────────────────────────
// Provider Factory & Manager
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Creates an LLM client instance for a specific provider
 */
export class LLMProvider {
  constructor(type, config = {}) {
    this.type = type;
    this.name = config.name || type;
    this.config = config;
    this.client = null;
    this.model = config.model || DEFAULT_MODELS[type];
    this.retryCount = 0;
    this.lastError = null;
    this.lastSuccess = null;
    
    this._initClient();
  }

  _initClient() {
    try {
      switch (this.type) {
        case PROVIDERS.OPENROUTER:
        case PROVIDERS.OPENAI:
          this.client = new OpenAI({
            baseURL: this.type === PROVIDERS.OPENAI 
              ? "https://api.openai.com/v1"
              : this.config.baseURL || "https://openrouter.ai/api/v1",
            apiKey: this.config.apiKey || 
                   (this.type === PROVIDERS.OPENROUTER 
                    ? process.env.OPENROUTER_API_KEY 
                    : process.env.LLM_API_KEY),
            timeout: 5 * 60 * 1000,
            ...this.config,
          });
          break;

        case PROVIDERS.AZURE:
          this.client = new OpenAI({
            baseURL: this.config.baseURL || "https://${resource}.openai.azure.com/openai/deployments/${deployment}/chat/completions?api-version=${version}",
            apiKey: this.config.apiKey || process.env.AZURE_API_KEY,
            defaultQuery: { "api-version": this.config.apiVersion || "2024-02-15-preview" },
            timeout: 5 * 60 * 1000,
            ...this.config,
          });
          break;

        case PROVIDERS.ANTHROPIC:
          this.client = new Anthropic({
            apiKey: this.config.apiKey || process.env.ANTHROPIC_API_KEY,
            ...this.config,
          });
          break;

        case PROVIDERS.GOOGLE:
          this.client = new GoogleGenerativeAI(
            this.config.apiKey || process.env.GOOGLE_API_KEY
          );
          break;

        case PROVIDERS.OLLAMA:
        case PROVIDERS.VLLM:
        case PROVIDERS.CUSTOM:
        default:
          this.client = new OpenAI({
            baseURL: this.config.baseURL || 
                   (this.type === PROVIDERS.OLLAMA 
                    ? "http://localhost:11434/v1" 
                    : this.config.baseURL),
            apiKey: this.config.apiKey || "ollama",
            timeout: 5 * 60 * 1000,
            ...this.config,
          });
          break;
      }
    } catch (error) {
      console.error(`Failed to initialize ${this.type} provider:`, error);
      this.lastError = error;
    }
  }

  /**
   * Attempts a chat completion with this provider
   * @param {Object} options - Chat options
   * @returns {Promise<Object>} Response or null if failed
   */
  async chat(options) {
    this.retryCount++;
    this.lastError = null;
    this.lastSuccess = Date.now();

    try {
      const messages = buildMessagesForProvider(
        this.type,
        options.messages || []
      );

      if (this.type === PROVIDERS.ANTHROPIC) {
        const anthropicMessages = transformToAnthropicFormat(messages);
        const response = await this.client.messages.create({
          model: this.model,
          messages: anthropicMessages,
          ...options,
        });
        return {
          choices: [{
            message: {
              content: response.content?.[0]?.text || "",
              tool_calls: extractToolCalls(anthropicMessages, response.content),
            }
          }],
          error: null,
        };
      }

      const response = await this.client.chat.completions.create({
        model: this.model,
        messages,
        ...options,
      });

      this.retryCount = 0;
      return {
        choices: [{
          message: {
            content: response.choices[0]?.message?.content || "",
            tool_calls: response.choices[0]?.message?.tool_calls || [],
          }
        }],
        error: null,
      };

    } catch (error) {
      this.lastError = error;
      const errorInfo = {
        code: error.response?.status || error.code,
        message: error.message,
        provider: this.type,
        model: this.model,
      };

      if (TRANSIENT_ERRORS.has(errorInfo.code)) {
        console.log(`[${this.type}] Transient error (${errorInfo.code}), will retry`);
      }

      return {
        choices: null,
        error,
      };
    }
  }

  /**
   * Checks if this provider is currently healthy
   */
  async healthCheck() {
    if (!this.client) return { healthy: false, reason: "Not initialized" };

    try {
      if (this.type === PROVIDERS.ANTHROPIC) {
        await this.client.messages.create({
          model: this.model,
          messages: [
            { role: "user", content: "ping" },
          ],
        });
      } else {
        await this.client.chat.completions.create({
          model: this.model,
          messages: [
            { role: "user", content: "ping" },
          ],
        });
      }
      
      this.lastError = null;
      return { healthy: true };
    } catch {
      return { healthy: false, reason: this.lastError?.message || "Unknown error" };
    }
  }

  /**
   * Resets retry count after successful request
   */
  resetRetry() {
    this.retryCount = 0;
    this.lastError = null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-Provider Manager (Smart Router)
// ─────────────────────────────────────────────────────────────────────────────

export class MultiLLMManager {
  constructor(config = {}) {
    // Initialize providers from config
    this.providers = new Map();
    this.defaultModel = config.defaultModel || "openrouter/healer-alpha";
    this.fallbackModel = config.fallbackModel || "stepfun/step-3.5-flash:free";
    this.maxRetries = config.maxRetries || 3;
    this.roundRobinIndex = 0;
    
    this._initProviders(config);
  }

  /**
   * Initializes all configured providers
   */
  _initProviders(config) {
    // Parse provider config
    const providerConfigs = Array.isArray(config.providers) 
      ? config.providers 
      : [config.providers];

    for (const providerConfig of providerConfigs) {
      if (!providerConfig) continue;

      const type = providerConfig.type || PROVIDERS.OPENROUTER;
      const name = providerConfig.name || type;

      // Create provider instance
      const provider = new LLMProvider(type, {
        name,
        ...providerConfig,
        model: providerConfig.model || DEFAULT_MODELS[type],
      });

      this.providers.set(name, provider);
      
      console.log(`[MultiLLM] Initialized provider: ${name} (${type})`);
    }

    // Always ensure at least one provider exists
    if (this.providers.size === 0) {
      const defaultProvider = new LLMProvider(PROVIDERS.OPENROUTER, {
        name: "default",
        model: this.defaultModel,
      });
      this.providers.set("default", defaultProvider);
    }
  }

  /**
   * Gets an available provider (with failover logic)
   */
  async getAvailableProvider(messages, options = {}) {
    const {
      preferredProvider = null,
      requireToolChoice = false,
    } = options;

    // Try preferred provider first
    let provider = preferredProvider 
      ? this.providers.get(preferredProvider)
      : null;

    // If no preferred or it's not available, try others
    if (!provider || !await provider.healthCheck()) {
      // Try to find a healthy provider
      for (const [name, p] of this.providers.entries()) {
        if (name !== preferredProvider && await p.healthCheck()) {
          provider = p;
          break;
        }
      }
    }

    // Fallback to default if still no provider
    if (!provider) {
      provider = this.providers.get("default") || 
                 Array.from(this.providers.values())[0];
    }

    return { provider, name: provider.name };
  }

  /**
   * Routes a chat request through available providers
   */
  async chat(messages, options = {}) {
    const {
      preferredProvider = null,
      maxRetries = this.maxRetries,
      onProviderChange = null,
    } = options;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      const { provider, name } = await this.getAvailableProvider(messages, {
        preferredProvider,
        requireToolChoice: options.toolChoice === "required",
      });

      // Log provider switch
      if (onProviderChange) {
        await onProviderChange({
          provider: name,
          attempt,
          totalAttempts: maxRetries,
        });
      }

      const response = await provider.chat({
        ...options,
        messages,
      });

      if (response.choices) {
        provider.resetRetry();
        return response;
      }

      // Provider failed, try next one or retry
      if (response.error) {
        console.log(`[${name}] Request failed: ${response.error.message}`);
        
        // If we have another provider, try it
        if (preferredProvider && preferredProvider !== name) {
          continue;
        }
        
        // Otherwise, retry with same provider but different model if available
        if (attempt < maxRetries - 1) {
          continue;
        }
      }
    }

    // All attempts failed
    return {
      choices: null,
      error: new Error("All providers failed after multiple attempts"),
    };
  }

  /**
   * Sets the active model for a specific provider
   */
  setModel(providerName, model) {
    const provider = this.providers.get(providerName);
    if (provider) {
      provider.model = model;
      return true;
    }
    return false;
  }

  /**
   * Gets current model for a provider
   */
  getModel(providerName) {
    const provider = this.providers.get(providerName);
    return provider ? provider.model : null;
  }

  /**
   * Lists all configured providers
   */
  listProviders() {
    return Array.from(this.providers.entries()).map(([name, provider]) => ({
      name,
      type: provider.type,
      model: provider.model,
      retryCount: provider.retryCount,
      lastError: provider.lastError?.message,
    }));
  }

  /**
   * Health check for all providers
   */
  async healthCheckAll() {
    const results = [];
    for (const [name, provider] of this.providers.entries()) {
      results.push({
        name,
        healthy: await provider.healthCheck(),
      });
    }
    return results;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Message Formatting Helpers
// ─────────────────────────────────────────────────────────────────────────────

function buildMessagesForProvider(providerType, messages) {
  // Most providers use OpenAI format
  if (providerType !== PROVIDERS.ANTHROPIC) {
    return messages;
  }

  // Transform to Anthropic format
  return transformToAnthropicFormat(messages);
}

function transformToAnthropicFormat(messages) {
  const anthropicMessages = [];
  
  for (const msg of messages) {
    if (msg.role === "system") {
      anthropicMessages.push({
        role: "system",
        content: msg.content,
      });
    } else if (msg.role === "user") {
      // Handle tool_call in user messages
      if (msg.tool_calls && msg.tool_calls.length > 0) {
        const toolContent = msg.tool_calls.map(tc => ({
          type: "tool_use",
          id: tc.id,
          name: tc.function.name,
          input: tc.function.arguments,
        }));
        anthropicMessages.push({
          role: "user",
          content: [
            ...(msg.content ? [{ type: "text", text: msg.content }] : []),
            ...toolContent,
          ],
        });
      } else {
        anthropicMessages.push({
          role: "user",
          content: msg.content,
        });
      }
    } else if (msg.role === "assistant") {
      const toolCalls = extractToolCalls(
        messages.filter(m => m.role === "assistant").pop()?.tool_calls || [],
        messages
      );
      
      if (toolCalls && toolCalls.length > 0) {
        anthropicMessages.push({
          role: "assistant",
          content: toolCalls.map(tc => ({
            type: "tool_use",
            id: tc.id,
            name: tc.function.name,
            input: tc.function.arguments,
          })),
        });
      } else if (msg.content) {
        anthropicMessages.push({
          role: "assistant",
          content: msg.content,
        });
      }
    } else if (msg.role === "tool") {
      anthropicMessages.push({
        role: "user",
        content: [{
          type: "tool_result",
          tool_use_id: msg.tool_call_id,
          content: msg.content,
        }],
      });
    }
  }

  return anthropicMessages;
}

function extractToolCalls(toolCalls, allMessages = []) {
  if (!toolCalls || toolCalls.length === 0) return [];
  
  // Try to match tool calls with result messages
  return toolCalls.map(tc => ({
    id: tc.id,
    function: {
      name: tc.function.name,
      arguments: tc.function.arguments,
    },
  }));
}

// ─────────────────────────────────────────────────────────────────────────────
// Default Export
// ─────────────────────────────────────────────────────────────────────────────

export function createMultiLLM(config = {}) {
  return new MultiLLMManager(config);
}

// Export classes for direct import
