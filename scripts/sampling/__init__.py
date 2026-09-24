"""Sweep-authoring sampling logic, host-side only.

Consumed by ``scripts/lhs_sample.py`` (and the analysis tooling) — never
by the installed control stack, which is why it lives here and not in
``py_pkg``. The scenario *schema* these modules draw against stays in
``py_pkg.scenarios.spec``; resolving that import needs ``src/py_pkg`` on
``sys.path``, which every ``scripts/`` entry point already inserts.
"""
