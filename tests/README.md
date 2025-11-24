# Test Suite for LLM Council

Comprehensive unit and integration tests for the LLM Council backend.

## Test Coverage

- **Unit Tests**: 44 tests covering individual modules
  - `test_openrouter.py`: OpenRouter API client (9 tests)
  - `test_council.py`: Council orchestration logic (17 tests)
  - `test_storage.py`: Storage operations (18 tests)

- **Integration Tests**: 9 tests covering API endpoints
  - `test_api.py`: FastAPI endpoints and request/response flows

**Total**: 53 tests, all passing

## Running Tests

```bash
# Run all tests
uv run pytest

# Run only unit tests
uv run pytest tests/unit/

# Run only integration tests
uv run pytest tests/integration/

# Run with coverage report
uv run pytest --cov=backend --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_council.py

# Run specific test
uv run pytest tests/unit/test_council.py::test_stage1_collect_responses_success

# Run in verbose mode
uv run pytest -v

# Run and stop at first failure
uv run pytest -x
```

## Test Organization

### Unit Tests (`tests/unit/`)

Test individual components in isolation with mocked dependencies:

- **OpenRouter Client**: API communication, error handling, parallel queries
- **Council Logic**: Multi-stage orchestration, ranking parsing, aggregation
- **Storage**: Conversation persistence, CRUD operations, file handling

### Integration Tests (`tests/integration/`)

Test API endpoints with real FastAPI TestClient:

- Conversation creation and retrieval
- Listing conversations
- Error handling (404s, validation)
- CORS configuration

## Coverage

Run with coverage to see detailed coverage reports:

```bash
uv run pytest --cov=backend --cov-report=html
open htmlcov/index.html
```

Current coverage: ~96% for core modules (openrouter, council, storage)

## Best Practices

- All tests use proper fixtures for isolation
- Temporary directories for storage tests
- AsyncMock for async function mocking
- Comprehensive edge case coverage
- Clear, descriptive test names
- Proper cleanup after tests

## CI/CD Integration

Tests are configured for CI/CD pipelines:

- Fast execution (<1 second)
- No external dependencies required for unit tests
- Isolated test environments
- Clear pass/fail reporting

## Adding New Tests

1. Create test file in appropriate directory (`unit` or `integration`)
2. Follow naming convention: `test_<module>.py`
3. Use descriptive test function names: `test_<functionality>_<scenario>`
4. Add docstrings explaining what is being tested
5. Use fixtures for common setup
6. Run tests to verify they pass

## Configuration

Test configuration is in `pytest.ini` at project root:

- Coverage thresholds
- Test discovery patterns
- Output formatting
- Marker definitions

For shared fixtures and configuration, see `conftest.py`.
