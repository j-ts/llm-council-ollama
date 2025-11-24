import React, { useState, useEffect } from 'react';
import { api } from '../api';
import './SettingsModal.css';

const PROVIDERS = [
    { id: 'ollama', name: 'Ollama (Local)' },
    { id: 'openrouter', name: 'OpenRouter (Cloud)' },
    { id: 'openai', name: 'OpenAI Compatible' },
];

// Common OpenRouter models for quick reference
const POPULAR_OPENROUTER_MODELS = [
    'anthropic/claude-3.5-sonnet',
    'openai/gpt-4-turbo',
    'google/gemini-pro-1.5',
    'meta-llama/llama-3.1-70b-instruct',
    'mistralai/mistral-large',
    'deepseek/deepseek-chat',
];

function SettingsModal({ isOpen, onClose }) {
    const [config, setConfig] = useState(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [status, setStatus] = useState({ type: '', message: '' });
    const [availableModels, setAvailableModels] = useState({
        ollama: [],
        openrouter: [],
        openai: []
    });
    const [loadingModels, setLoadingModels] = useState(false);
    const [expandedProviders, setExpandedProviders] = useState({
        ollama: true,
        openrouter: true,
        openai: false
    });

    useEffect(() => {
        if (isOpen) {
            loadConfig();
            setStatus({ type: '', message: '' });
        }
    }, [isOpen]);

    const loadConfig = async () => {
        try {
            setLoading(true);
            const data = await api.getConfig();
            setConfig(data);
            loadAllModels();
        } catch (error) {
            console.error('Failed to load config:', error);
            setStatus({ type: 'error', message: 'Failed to load configuration. Using defaults.' });
            setConfig({
                providers: {
                    ollama: { base_url: 'http://localhost:11434' },
                    openrouter: { api_key: '' },
                    openai: { base_url: 'https://api.openai.com/v1', api_key: '' }
                },
                council_models: [],
                chairman_model: { name: '', provider: 'ollama' }
            });
        } finally {
            setLoading(false);
        }
    };

    const loadAllModels = async () => {
        try {
            setLoadingModels(true);
            const data = await api.listAllModels();
            setAvailableModels(data.models || {});
        } catch (error) {
            console.error('Failed to load models:', error);
            setAvailableModels({ ollama: [], openrouter: [], openai: [] });
        } finally {
            setLoadingModels(false);
        }
    };

    const toggleProvider = (provider) => {
        setExpandedProviders({
            ...expandedProviders,
            [provider]: !expandedProviders[provider]
        });
    };

    const validateConfig = () => {
        if (!config) return false;

        // Check if required provider credentials are present for selected models
        const usedProviders = new Set();

        // Collect providers from council models
        config.council_models.forEach(model => {
            if (model.provider) usedProviders.add(model.provider);
        });

        // Add chairman model provider
        if (config.chairman_model && config.chairman_model.provider) {
            usedProviders.add(config.chairman_model.provider);
        }

        // Validate each used provider has required credentials
        for (const provider of usedProviders) {
            if (provider === 'openrouter' && !config.providers.openrouter?.api_key) {
                setStatus({ type: 'error', message: 'OpenRouter API Key is required for selected models.' });
                return false;
            }
            if (provider === 'ollama' && !config.providers.ollama?.base_url) {
                setStatus({ type: 'error', message: 'Ollama Base URL is required for selected models.' });
                return false;
            }
            if (provider === 'openai') {
                if (!config.providers.openai?.base_url) {
                    setStatus({ type: 'error', message: 'OpenAI Base URL is required for selected models.' });
                    return false;
                }
                if (!config.providers.openai?.api_key) {
                    setStatus({ type: 'error', message: 'OpenAI API Key is required for selected models.' });
                    return false;
                }
            }
        }

        // Validate model names
        if (config.council_models.some(m => !m.name?.trim())) {
            setStatus({ type: 'error', message: 'All Council Models must have a name.' });
            return false;
        }

        if (!config.chairman_model?.name) {
            setStatus({ type: 'error', message: 'Chairman Model is required.' });
            return false;
        }

        return true;
    };

    const handleSave = async () => {
        setStatus({ type: '', message: '' });

        if (!validateConfig()) {
            return;
        }

        try {
            setSaving(true);
            await api.updateConfig(config);
            setStatus({ type: 'success', message: 'Settings saved successfully!' });
            setTimeout(() => {
                onClose();
            }, 1000);
        } catch (error) {
            console.error('Failed to save config:', error);
            setStatus({ type: 'error', message: 'Failed to save settings.' });
        } finally {
            setSaving(false);
        }
    };

    const handleCouncilModelChange = (index, field, value) => {
        const newModels = [...config.council_models];
        newModels[index] = { ...newModels[index], [field]: value };
        setConfig({ ...config, council_models: newModels });
    };

    const addCouncilModel = () => {
        setConfig({
            ...config,
            council_models: [...config.council_models, { name: '', provider: 'ollama' }]
        });
    };

    const removeCouncilModel = (index) => {
        const newModels = config.council_models.filter((_, i) => i !== index);
        setConfig({ ...config, council_models: newModels });
    };

    const updateProviderConfig = (provider, field, value) => {
        setConfig({
            ...config,
            providers: {
                ...config.providers,
                [provider]: {
                    ...config.providers[provider],
                    [field]: value
                }
            }
        });
    };

    const getProviderHelp = (provider) => {
        switch (provider) {
            case 'ollama':
                return {
                    text: 'Local models (free). List available: ollama list',
                    link: 'https://ollama.com/library',
                    linkText: 'Browse Models'
                };
            case 'openrouter':
                return {
                    text: 'Cloud models (paid). Use format: provider/model-name',
                    link: 'https://openrouter.ai/models',
                    linkText: 'Browse Models'
                };
            case 'openai':
                return {
                    text: 'OpenAI or compatible endpoints',
                    link: 'https://platform.openai.com/docs/models',
                    linkText: 'View Docs'
                };
            default:
                return { text: '', link: '', linkText: '' };
        }
    };

    if (!isOpen) return null;

    return (
        <div className="settings-modal-overlay">
            <div className="settings-modal">
                <div className="settings-header">
                    <h2>Settings</h2>
                    <button className="close-button" onClick={onClose}>&times;</button>
                </div>

                {loading ? (
                    <div className="loading">Loading settings...</div>
                ) : (
                    <div className="settings-content">
                        {status.message && (
                            <div className={`status-message ${status.type}`}>
                                {status.message}
                            </div>
                        )}

                        <div className="providers-section">
                            <h3>Provider Configurations</h3>
                            <p className="section-help">Configure credentials for providers you want to use. The same API key can be used for multiple models from the same provider.</p>

                            {/* Ollama */}
                            <div className="provider-config">
                                <div
                                    className="provider-header"
                                    onClick={() => toggleProvider('ollama')}
                                >
                                    <h4>
                                        <span className="toggle-icon">{expandedProviders.ollama ? '▼' : '▶'}</span>
                                        Ollama (Local)
                                    </h4>
                                    <span className="provider-badge ollama">Free</span>
                                </div>
                                {expandedProviders.ollama && (
                                    <div className="provider-content">
                                        <div className="form-group">
                                            <label>Base URL</label>
                                            <input
                                                type="text"
                                                value={config.providers.ollama?.base_url || ''}
                                                onChange={(e) => updateProviderConfig('ollama', 'base_url', e.target.value)}
                                                placeholder="http://localhost:11434"
                                            />
                                            <small>
                                                Ensure Ollama is running. <a href="https://ollama.com/library" target="_blank" rel="noopener noreferrer">Browse models</a>
                                            </small>
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* OpenRouter */}
                            <div className="provider-config">
                                <div
                                    className="provider-header"
                                    onClick={() => toggleProvider('openrouter')}
                                >
                                    <h4>
                                        <span className="toggle-icon">{expandedProviders.openrouter ? '▼' : '▶'}</span>
                                        OpenRouter (Cloud)
                                    </h4>
                                    <span className="provider-badge openrouter">Paid</span>
                                </div>
                                {expandedProviders.openrouter && (
                                    <div className="provider-content">
                                        <div className="form-group">
                                            <label>API Key</label>
                                            <input
                                                type="password"
                                                value={config.providers.openrouter?.api_key || ''}
                                                onChange={(e) => updateProviderConfig('openrouter', 'api_key', e.target.value)}
                                                placeholder="sk-or-v1-..."
                                            />
                                            <small>
                                                Get your API key from <a href="https://openrouter.ai/keys" target="_blank" rel="noopener noreferrer">OpenRouter dashboard</a>
                                            </small>
                                        </div>
                                        <div className="popular-models">
                                            <strong>Popular models:</strong>
                                            <ul>
                                                {POPULAR_OPENROUTER_MODELS.slice(0, 4).map(model => (
                                                    <li key={model}><code>{model}</code></li>
                                                ))}
                                            </ul>
                                            <a href="https://openrouter.ai/models" target="_blank" rel="noopener noreferrer">View all models →</a>
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* OpenAI */}
                            <div className="provider-config">
                                <div
                                    className="provider-header"
                                    onClick={() => toggleProvider('openai')}
                                >
                                    <h4>
                                        <span className="toggle-icon">{expandedProviders.openai ? '▼' : '▶'}</span>
                                        OpenAI Compatible
                                    </h4>
                                    <span className="provider-badge openai">Paid</span>
                                </div>
                                {expandedProviders.openai && (
                                    <div className="provider-content">
                                        <div className="form-group">
                                            <label>Base URL</label>
                                            <input
                                                type="text"
                                                value={config.providers.openai?.base_url || ''}
                                                onChange={(e) => updateProviderConfig('openai', 'base_url', e.target.value)}
                                                placeholder="https://api.openai.com/v1"
                                            />
                                        </div>
                                        <div className="form-group">
                                            <label>API Key</label>
                                            <input
                                                type="password"
                                                value={config.providers.openai?.api_key || ''}
                                                onChange={(e) => updateProviderConfig('openai', 'api_key', e.target.value)}
                                                placeholder="sk-..."
                                            />
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>

                        <hr />

                        <h3>Council Models</h3>
                        <p className="section-help">Add models that will provide diverse perspectives. You can mix models from different providers.</p>
                        <div className="council-models-list">
                            {config.council_models.map((model, index) => {
                                const help = getProviderHelp(model.provider || 'ollama');
                                return (
                                    <div key={index} className="model-row-container">
                                        <div className="model-row">
                                            <select
                                                value={model.provider || 'ollama'}
                                                onChange={(e) => handleCouncilModelChange(index, 'provider', e.target.value)}
                                                className="provider-select"
                                            >
                                                {PROVIDERS.map(p => (
                                                    <option key={p.id} value={p.id}>{p.name}</option>
                                                ))}
                                            </select>
                                            <input
                                                type="text"
                                                value={model.name || ''}
                                                onChange={(e) => handleCouncilModelChange(index, 'name', e.target.value)}
                                                placeholder={
                                                    model.provider === 'openrouter'
                                                        ? 'e.g., anthropic/claude-3.5-sonnet'
                                                        : model.provider === 'ollama'
                                                            ? 'e.g., deepseek-r1:1.5b'
                                                            : 'Model name'
                                                }
                                                list={`available-models-${model.provider || 'ollama'}`}
                                            />
                                            <button className="remove-button" onClick={() => removeCouncilModel(index)}>&times;</button>
                                        </div>
                                        {help.text && (
                                            <div className="model-help">
                                                <small>
                                                    {help.text} {help.link && (
                                                        <a href={help.link} target="_blank" rel="noopener noreferrer">{help.linkText}</a>
                                                    )}
                                                </small>
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                            <button className="add-button" onClick={addCouncilModel}>+ Add Model</button>
                        </div>

                        <div className="form-group">
                            <label>Chairman Model</label>
                            <p className="section-help">The model that synthesizes the final answer based on council input.</p>
                            <div className="model-row-container">
                                <div className="model-row">
                                    <select
                                        value={config.chairman_model?.provider || 'ollama'}
                                        onChange={(e) => setConfig({
                                            ...config,
                                            chairman_model: {
                                                ...config.chairman_model,
                                                provider: e.target.value
                                            }
                                        })}
                                        className="provider-select"
                                    >
                                        {PROVIDERS.map(p => (
                                            <option key={p.id} value={p.id}>{p.name}</option>
                                        ))}
                                    </select>
                                    <input
                                        type="text"
                                        value={config.chairman_model?.name || ''}
                                        onChange={(e) => setConfig({
                                            ...config,
                                            chairman_model: {
                                                ...config.chairman_model,
                                                name: e.target.value
                                            }
                                        })}
                                        placeholder={
                                            config.chairman_model?.provider === 'openrouter'
                                                ? 'e.g., anthropic/claude-3.5-sonnet'
                                                : config.chairman_model?.provider === 'ollama'
                                                    ? 'e.g., deepseek-r1:1.5b'
                                                    : 'Model name'
                                        }
                                        list={`available-models-${config.chairman_model?.provider || 'ollama'}`}
                                    />
                                </div>
                                {(() => {
                                    const help = getProviderHelp(config.chairman_model?.provider || 'ollama');
                                    return help.text && (
                                        <div className="model-help">
                                            <small>
                                                {help.text} {help.link && (
                                                    <a href={help.link} target="_blank" rel="noopener noreferrer">{help.linkText}</a>
                                                )}
                                            </small>
                                        </div>
                                    );
                                })()}
                            </div>
                        </div>

                        {/* Datalists for autocomplete */}
                        {Object.entries(availableModels).map(([provider, models]) => (
                            <datalist key={provider} id={`available-models-${provider}`}>
                                {models.map(m => (
                                    <option key={m} value={m} />
                                ))}
                            </datalist>
                        ))}

                    </div>
                )}

                <div className="settings-footer">
                    <button className="cancel-button" onClick={onClose}>Cancel</button>
                    <button className="save-button" onClick={handleSave} disabled={saving}>
                        {saving ? 'Saving...' : 'Save Changes'}
                    </button>
                </div>
            </div>
        </div>
    );
}

export default SettingsModal;
