# CONTRIBUTING

## Getting Started

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Code Standards

All code must follow the guidelines in `STYLE_GUIDE.md`. Before submitting:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/

# Run tests
pytest tests/ -v
```

## Commit Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat(scope): description` - New feature
- `fix(scope): description` - Bug fix
- `docs(scope): description` - Documentation
- `refactor(scope): description` - Code refactoring
- `test(scope): description` - Tests
- `chore(scope): description` - Maintenance

Scopes: `preprocessing`, `retrieval`, `generation`, `evaluation`, `ui`, `config`, `deps`

## Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] Type hints added
- [ ] Docstrings added
- [ ] Tests pass locally
- [ ] No new warnings
- [ ] Dependencies pinned
