# AGENTS.md

## Cursor Cloud specific instructions

### Overview

ATLASky-AI is a single-service Python/Streamlit application for 4D Spatiotemporal Knowledge Graph verification. No databases, Docker, or external infrastructure required (except optional OpenAI API key for real LLM extraction).

### Running the application

```bash
streamlit run app.py --server.headless true --server.port 8501
```

The dashboard opens at `http://localhost:8501`. See `README.md` for full usage instructions.

### Linting

```bash
flake8 --max-line-length=120 --select=E9,F63,F7,F82 .
```

Note: There is one pre-existing `F821` warning in `app.py` (undefined name `test_domain`) that has been fixed on the development branch.

### Tests

```bash
python3 test_verification_demo.py        # Verification pipeline demo
python3 experiments/quick_demo.py        # Dataset experiment demo
python3 test_domain_adaptation.py        # Domain adaptation test
```

There is no formal test framework (pytest, unittest). The test scripts above exit 0 on success.

### Gotchas

- The `streamlit` binary installs to `~/.local/bin` which may not be on PATH. Ensure `export PATH="$HOME/.local/bin:$PATH"` is set or use `python3 -m streamlit` as an alternative.
- The app runs entirely in-memory (Streamlit session state). No database setup needed.
- OpenAI API integration is optional; the system falls back to simulation mode without `OPENAI_API_KEY`.
