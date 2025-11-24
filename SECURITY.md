# Security Policy

## Environment Variables and Secrets

**CRITICAL**: Never commit secrets, API keys, passwords, or credentials to version control.

### Protected Files

The following files contain sensitive data and are gitignored:

- `.env` - Environment variables with API keys
- `.env.local`, `.env.*.local` - Local environment configs
- `*.key`, `*.pem` - Private keys and certificates
- `secrets/`, `credentials/` - Any directories containing secrets

### Setup Instructions

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your actual API keys:
   ```bash
   OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
   ```

3. **Never** commit the `.env` file

### Verification

Before committing, verify no secrets are staged:

```bash
# Check what files are staged
git status

# Verify .env is ignored
git check-ignore .env

# Search for potential secrets in tracked files
git grep -i "api.key\|password\|secret\|credential" -- ':!SECURITY.md' ':!README.md'
```

### API Key Management

- **OpenRouter API Key**: Stored in `.env` as `OPENROUTER_API_KEY`
- Loaded via `python-dotenv` in `backend/config.py`
- Never hardcoded in source files
- Get your key from: https://openrouter.ai/

### If You Accidentally Commit Secrets

1. **Immediately revoke/rotate** the exposed credentials
2. Remove from git history:
   ```bash
   git filter-branch --force --index-filter \
     'git rm --cached --ignore-unmatch .env' \
     --prune-empty --tag-name-filter cat -- --all
   ```
3. Force push (only if not shared): `git push --force`
4. For shared repos, contact all collaborators and rotate all secrets

## Code Security

### Input Validation

- User input is validated through Pydantic models
- No SQL injection risk (uses JSON file storage)
- API requests validated by FastAPI

### Dependencies

Keep dependencies updated:

```bash
uv lock --upgrade
uv sync
```

Check for known vulnerabilities:

```bash
# Audit Python packages
pip-audit
```

### Secure Communication

- Use HTTPS in production
- OpenRouter API uses Bearer token authentication
- CORS configured for specific origins only

### Data Storage

- Conversation data stored locally in `data/conversations/`
- No sensitive user data collected
- Session data not persistent across restarts

## Reporting Security Issues

If you discover a security vulnerability:

1. **Do NOT** open a public issue
2. Email the maintainers directly
3. Include detailed description and reproduction steps
4. Allow time for patch development before disclosure

## Security Checklist for Contributors

- [ ] No secrets in commits
- [ ] `.env` properly gitignored
- [ ] No hardcoded credentials
- [ ] Input validation in place
- [ ] Dependencies up to date
- [ ] HTTPS used for production
- [ ] Error messages don't leak sensitive info

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
