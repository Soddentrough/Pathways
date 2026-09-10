---
name: tool_usage
description: Enforce silent native read tools and consolidated build/test execution to minimize approval prompts.
trigger: always_on
---

# Tool Usage & Approval Minimization

1. **Native Read Tools Only**:
   - For all file reading, syntax inspection, and codebase exploration, use `view_file`.
   - For searching strings, patterns, symbols, and functions across the repo, use `grep_search`.
   - For finding files and listing directories, use `find_by_name` and `list_dir`.
   - Never run shell commands (`Select-String`, `Get-ChildItem`, `cat`, `dir`, `ls`, `grep`) for read-only inspection.

2. **Single-Command Orchestration**:
   - Always prefer `.\build.ps1 -Test` for compilation and unit test execution.
   - Always prefer `.\scripts\run_headless_tests.ps1` for end-to-end regression validation.
   - Do not invoke multiple shell commands when a single script handles the pipeline.
