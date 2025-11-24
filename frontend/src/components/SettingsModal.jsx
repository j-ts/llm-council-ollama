import React, { useState, useEffect } from 'react';
import { api } from '../api';
import './SettingsModal.css';

const PROVIDERS = [
    { id: 'openrouter', name: 'OpenRouter (Cloud)' },
    { id: 'ollama', name: 'Ollama (Local)' },
    { id: 'openai', name: 'OpenAI Compatible (Cloud/Local)' },
];

function SettingsModal({ isOpen, onClose }) {
    const [config, setConfig] = useState(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [availableModels, setAvailableModels] = useState([]);
    const [loadingModels, setLoadingModels] = useState(false);

    useEffect(() => {
        if (isOpen) {
            loadConfig();
        }
    }, [isOpen]);

    const loadConfig = async () => {
        try {
            setLoading(true);
            const data = await api.getConfig();
            setConfig(data);
            // Load models for the current provider
            if (data.provider) {
                loadModels(data.provider);
            }
        } catch (error) {
            console.error('Failed to load config:', error);
        } finally {
            setLoading(false);
        }
    };

    const loadModels = async (provider) => {
        try {
            setLoadingModels(true);
            const data = await api.listModels(provider);
            setAvailableModels(data.models || []);
        } catch (error) {
            console.error('Failed to load models:', error);
            setAvailableModels([]);
        } finally {
            setLoadingModels(false);
        }
    };

    const handleProviderChange = (e) => {
        const newProvider = e.target.value;
        setConfig({ ...config, provider: newProvider });
        loadModels(newProvider);
    };

    const handleSave = async () => {
        try {
            setSaving(true);
            await api.updateConfig(config);
            onClose();
        } catch (error) {
            console.error('Failed to save config:', error);
            alert('Failed to save settings');
        } finally {
            setSaving(false);
        }
    };

    const handleCouncilModelChange = (index, value) => {
        const newModels = [...config.council_models];
        newModels[index] = value;
        setConfig({ ...config, council_models: newModels });
    };

    const addCouncilModel = () => {
        setConfig({ ...config, council_models: [...config.council_models, ''] });
    };

    const removeCouncilModel = (index) => {
        const newModels = config.council_models.filter((_, i) => i !== index);
        setConfig({ ...config, council_models: newModels });
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
                        <div className="form-group">
                            <label>Provider</label>
                            <select value={config.provider} onChange={handleProviderChange}>
                                {PROVIDERS.map(p => (
                                    <option key={p.id} value={p.id}>{p.name}</option>
                                ))}
                            </select>
                        </div>

                        {config.provider === 'openrouter' && (
                            <div className="form-group">
                                <label>OpenRouter API Key</label>
                                <input
                                    type="password"
                                    value={config.openrouter_api_key || ''}
                                    onChange={(e) => setConfig({ ...config, openrouter_api_key: e.target.value })}
                                    placeholder="sk-or-..."
                                />
                            </div>
                        )}

                        {config.provider === 'ollama' && (
                            <div className="form-group">
                                <label>Ollama Base URL</label>
                                <input
                                    type="text"
                                    value={config.ollama_base_url || ''}
                                    onChange={(e) => setConfig({ ...config, ollama_base_url: e.target.value })}
                                    placeholder="http://localhost:11434"
                                />
                                <small>Ensure Ollama is running and accessible (use host.docker.internal if in Docker)</small>
                            </div>
                        )}

                        {config.provider === 'openai' && (
                            <>
                                <div className="form-group">
                                    <label>API Base URL</label>
                                    <input
                                        type="text"
                                        value={config.openai_base_url || ''}
                                        onChange={(e) => setConfig({ ...config, openai_base_url: e.target.value })}
                                        placeholder="https://api.openai.com/v1"
                                    />
                                </div>
                                <div className="form-group">
                                    <label>API Key</label>
                                    <input
                                        type="password"
                                        value={config.openai_api_key || ''}
                                        onChange={(e) => setConfig({ ...config, openai_api_key: e.target.value })}
                                        placeholder="sk-..."
                                    />
                                </div>
                            </>
                        )}

                        <hr />

                        <h3>Council Models</h3>
                        <div className="council-models-list">
                            {config.council_models.map((model, index) => (
                                <div key={index} className="model-row">
                                    <input
                                        type="text"
                                        value={model}
                                        onChange={(e) => handleCouncilModelChange(index, e.target.value)}
                                        placeholder="Model ID"
                                        list="available-models"
                                    />
                                    <button className="remove-button" onClick={() => removeCouncilModel(index)}>&times;</button>
                                </div>
                            ))}
                            <button className="add-button" onClick={addCouncilModel}>+ Add Model</button>
                        </div>

                        <div className="form-group">
                            <label>Chairman Model</label>
                            <input
                                type="text"
                                value={config.chairman_model || ''}
                                onChange={(e) => setConfig({ ...config, chairman_model: e.target.value })}
                                placeholder="Model ID"
                                list="available-models"
                            />
                        </div>

                        <datalist id="available-models">
                            {availableModels.map(m => (
                                <option key={m} value={m} />
                            ))}
                        </datalist>

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
