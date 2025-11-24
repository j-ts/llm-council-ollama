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

            // Ensure openai is an array (handle legacy/migration edge cases in frontend state)
            if (data.providers.openai && !Array.isArray(data.providers.openai)) {
                data.providers.openai = [data.providers.openai];
            }
            if (!data.providers.openai) {
                data.providers.openai = [];
            }

            setConfig(data);
            loadAllModels();
        } catch (error) {
            console.error('Failed to load config:', error);
            setStatus({ type: 'error', message: 'Failed to load configuration. Using defaults.' });
            setConfig({
                providers: {
                    ollama: { base_url: 'http://localhost:11434' },
                    openrouter: { api_key: '' },
                    openai: [{ name: 'Default', base_url: 'https://api.openai.com/v1', api_key: '' }]
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
        const usedOpenAIConfigs = new Set();

        // Collect providers from council models
        config.council_models.forEach(model => {
            if (model.provider) usedProviders.add(model.provider);
            if (model.provider === 'openai' && model.openai_config_name) {
                usedOpenAIConfigs.add(model.openai_config_name);
            }
        });

        // Add chairman model provider
        if (config.chairman_model && config.chairman_model.provider) {
            usedProviders.add(config.chairman_model.provider);
            if (config.chairman_model.provider === 'openai' && config.chairman_model.openai_config_name) {
                usedOpenAIConfigs.add(config.chairman_model.openai_config_name);
            }
        }

        // Validate each used provider has required credentials
        if (usedProviders.has('openrouter') && !config.providers.openrouter?.api_key) {
            setStatus({ type: 'error', message: 'OpenRouter API Key is required for selected models.' });
            return false;
        }
        if (usedProviders.has('ollama') && !config.providers.ollama?.base_url) {
            setStatus({ type: 'error', message: 'Ollama Base URL is required for selected models.' });
            return false;
        }

        if (usedProviders.has('openai')) {
            const openaiConfigs = config.providers.openai || [];
            if (openaiConfigs.length === 0) {
                setStatus({ type: 'error', message: 'At least one OpenAI configuration is required.' });
                return false;
            }

            // Names must be unique so configs can be targeted reliably
            const openaiNameSet = new Set();

            // Check if used configs exist and are valid
            for (const configName of usedOpenAIConfigs) {
                const conf = openaiConfigs.find(c => c.name === configName);
                if (!conf) {
                    setStatus({ type: 'error', message: `OpenAI config "${configName}" is used but not defined.` });
                    return false;
                }
                if (!conf.base_url) {
                    setStatus({ type: 'error', message: `Base URL is required for OpenAI config "${configName}".` });
                    return false;
                }
                if (!conf.api_key) {
                    setStatus({ type: 'error', message: `API Key is required for OpenAI config "${configName}".` });
                    return false;
                }
            }

            // If no specific config is used but 'openai' provider is selected (e.g. default), check the first one or all?
            // Let's just check that all defined configs have at least a name and base_url
            for (const conf of openaiConfigs) {
                if (!conf.name) {
                    setStatus({ type: 'error', message: 'All OpenAI configurations must have a name.' });
                    return false;
                }
                const normalizedName = conf.name.trim().toLowerCase();
                if (openaiNameSet.has(normalizedName)) {
                    setStatus({ type: 'error', message: 'OpenAI configuration names must be unique.' });
                    return false;
                }
                openaiNameSet.add(normalizedName);
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

        // If provider changed to openai, set default config name if not present
        if (field === 'provider' && value === 'openai' && !newModels[index].openai_config_name) {
            if (config.providers.openai && config.providers.openai.length > 0) {
                newModels[index].openai_config_name = config.providers.openai[0].name;
            }
        }

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

    // OpenAI specific management
    const addOpenAIConfig = () => {
        const newConfig = {
            name: `Config ${config.providers.openai.length + 1}`,
            base_url: 'https://api.openai.com/v1',
            api_key: ''
        };
        setConfig({
            ...config,
            providers: {
                ...config.providers,
                openai: [...(config.providers.openai || []), newConfig]
            }
        });
    };

    const removeOpenAIConfig = (index) => {
        const newConfigs = [...config.providers.openai];
        newConfigs.splice(index, 1);
        setConfig({
            ...config,
            providers: {
                ...config.providers,
                openai: newConfigs
            }
        });
    };

    const updateOpenAIConfig = (index, field, value) => {
        const newConfigs = [...config.providers.openai];
        newConfigs[index] = { ...newConfigs[index], [field]: value };
        setConfig({
            ...config,
            providers: {
                ...config.providers,
                openai: newConfigs
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
                                        <div className="openai-configs-list">
                                            {(config.providers.openai || []).map((conf, index) => (
                                                <div key={index} className="openai-config-item">
                                                    <div className="openai-config-header">
                                                        <input
                                                            type="text"
                                                            className="config-name-input"
                                                            value={conf.name}
                                                            onChange={(e) => updateOpenAIConfig(index, 'name', e.target.value)}
                                                            placeholder="Provider Name (e.g. OpenAI, LocalAI)"
                                                        />
                                                        <button
                                                            className="remove-config-button"
                                                            onClick={() => removeOpenAIConfig(index)}
                                                            title="Remove this configuration"
                                                        >
                                                            &times;
                                                        </button>
                                                    </div>
                                                    <div className="form-group">
                                                        <label>Base URL</label>
                                                        <input
                                                            type="text"
                                                            value={conf.base_url || ''}
                                                            onChange={(e) => updateOpenAIConfig(index, 'base_url', e.target.value)}
                                                            placeholder="https://api.openai.com/v1"
                                                        />
                                                    </div>
                                                    <div className="form-group">
                                                        <label>API Key</label>
                                                        <input
                                                            type="password"
                                                            value={conf.api_key || ''}
                                                            onChange={(e) => updateOpenAIConfig(index, 'api_key', e.target.value)}
                                                            placeholder="sk-..."
                                                        />
                                                    </div>
                                                </div>
                                            ))}
                                            <button className="add-config-button" onClick={addOpenAIConfig}>
                                                + Add OpenAI Configuration
                                            </button>
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

                                            {model.provider === 'openai' && (
                                                <select
                                                    value={model.openai_config_name || (config.providers.openai?.[0]?.name) || ''}
                                                    onChange={(e) => handleCouncilModelChange(index, 'openai_config_name', e.target.value)}
                                                    className="openai-config-select"
                                                >
                                                    {(config.providers.openai || []).map(conf => (
                                                        <option key={conf.name} value={conf.name}>{conf.name}</option>
                                                    ))}
                                                </select>
                                            )}

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
                                                list={
                                                    model.provider === 'openai'
                                                        ? `available-models-openai:${model.openai_config_name || (config.providers.openai?.[0]?.name) || ''}`
                                                        : `available-models-${model.provider || 'ollama'}`
                                                }
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

                        <hr />

                        <h3>Chairman Model</h3>
                        <p className="section-help">The model that synthesizes the final answer based on council input.</p>
                        <div className="model-row-container">
                            <div className="model-row">
                                <select
                                    value={config.chairman_model?.provider || 'ollama'}
                                    onChange={(e) => {
                                        const newProvider = e.target.value;
                                        const updates = { provider: newProvider };
                                        if (newProvider === 'openai' && !config.chairman_model.openai_config_name) {
                                            if (config.providers.openai && config.providers.openai.length > 0) {
                                                updates.openai_config_name = config.providers.openai[0].name;
                                            }
                                        }
                                        setConfig({
                                            ...config,
                                            chairman_model: {
                                                ...config.chairman_model,
                                                ...updates
                                            }
                                        });
                                    }}
                                    className="provider-select"
                                >
                                    {PROVIDERS.map(p => (
                                        <option key={p.id} value={p.id}>{p.name}</option>
                                    ))}
                                </select>

                                {config.chairman_model?.provider === 'openai' && (
                                    <select
                                        value={config.chairman_model.openai_config_name || (config.providers.openai?.[0]?.name) || ''}
                                        onChange={(e) => setConfig({
                                            ...config,
                                            chairman_model: {
                                                ...config.chairman_model,
                                                openai_config_name: e.target.value
                                            }
                                        })}
                                        className="openai-config-select"
                                    >
                                        {(config.providers.openai || []).map(conf => (
                                            <option key={conf.name} value={conf.name}>{conf.name}</option>
                                        ))}
                                    </select>
                                )}

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
