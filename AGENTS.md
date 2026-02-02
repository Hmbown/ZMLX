## Project Agent Guidance

- Do **not** include machine-specific absolute paths (e.g., `/Volumes/VIXinSSD/...`) in README/docs or user-facing text.
- Use placeholders like `<REPO_ROOT>`, `$HF_HOME`, or repository-relative paths instead.
- If an absolute path is unavoidable for a command example, prefer an environment variable (e.g., `HF_HOME`) over a literal path.
