# Naming & Code Conventions

<naming_conventions version="1.0">

<description>
  Universal code hygiene rules that apply across languages and projects.
  Project-specific linting/formatting rules live in `AGENTS.md` and
  `process/development-protocols/implementation-standards.md`.
</description>

---

## 1. Naming Rules

<naming_rules>
  <rule id="no_leading_underscore">
    Never use a leading underscore `_` for function or method names to imply
    "private". Write flat, clean, public functions with self-explanatory names.
    Exception: language-mandated dunder/magic methods (e.g., Python `__init__`).
  </rule>

  <rule id="self_explanatory">
    Names must reveal intent without requiring a comment to explain them.
    Prefer `calculate_retry_delay()` over `calc()` or `do_thing()`.
  </rule>

  <rule id="no_abbreviations">
    Avoid opaque abbreviations. Write `connection_timeout` not `conn_to`,
    `maximum_retry_count` not `max_rtry_cnt`.
    Exception: universally understood acronyms (e.g., `url`, `id`, `api`).
  </rule>
</naming_rules>

---

## 2. Structural Hygiene

<structural_rules>
  <rule id="single_responsibility">
    Each function or class has one clearly stated responsibility.
    Split when a unit does more than one thing.
  </rule>

  <rule id="no_magic_values">
    Extract all numeric/string literals into named constants or configuration.
  </rule>

  <rule id="no_suppression">
    Never silence linter or type checker errors with inline suppression comments
    (`# type: ignore`, `# noqa`, `eslint-disable`, `@SuppressWarnings`, etc.)
    unless the suppression itself is explicitly reviewed and annotated with a reason.
  </rule>
</structural_rules>

</naming_conventions>
