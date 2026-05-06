"""
MkDocs hooks — fix .ipynb documentation when using mkdocs-jupyter + mkdocs-static-i18n.

mkdocs-static-i18n runs on_files last (priority -100) and replaces each File with a plain
File. That drops mkdocs_jupyter.plugin.NotebookFile, so .ipynb is no longer a documentation
page, mkdocs-jupyter's on_pre_page render patch never applies, and the built page is raw
notebook JSON instead of nbconvert HTML.

We re-wrap .ipynb files as NotebookFile in on_files at priority -101 (after i18n).

Important: return a new ``Files`` that preserves ``mkdocs_static_i18n``'s ``I18nFiles`` type.
If we replace ``I18nFiles`` with plain ``Files``, ``get_file_from_path("index.md")`` no longer
resolves ``en/index.md`` / ``zh/index.md``, and every nav entry becomes a ``Link`` whose
``href`` stays as ``*.md`` (404 in the browser).

on_page_content remains a safety net if content still looks like unconverted JSON.

Notebook HTML from nbconvert includes ``<script src="https://cdnjs.../mathjax/...">`` for LaTeX.
In some MkDocs + Material builds the external ``src`` is rewritten to empty, so MathJax never
loads and ``$$...$$`` stays unrendered. ``on_post_page`` restores the loader URL when needed.

If nbconvert fails with imports like ``jupyter_contrib_nbextensions``, your user-level
Jupyter config is enabling optional preprocessors. Build with an empty config dir, e.g.
``JUPYTER_CONFIG_DIR=$PWD/docs/.jupyter_site_build mkdocs build`` (see run_doc.sh).

Zh locale notebooks are not committed: ``docs/hooks.py`` mirrors ``docs/en/examples/notebooks/*.ipynb``
into ``docs/zh/examples/notebooks/`` on each build (``run_doc.sh`` does the same via rsync/cp
before ``mkdocs serve`` so the working tree matches).
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

from mkdocs.plugins import event_priority
from mkdocs.structure.files import Files


def _include_notebooks() -> bool:
    """Keep in sync with mkdocs.yml: ``enabled: !ENV [POWERZOO_DOCS_INCLUDE_NOTEBOOKS, false]``."""
    v = os.environ.get("POWERZOO_DOCS_INCLUDE_NOTEBOOKS", "false").strip().lower()
    return v not in ("0", "false", "no", "off")


def _notebook_skipped_placeholder() -> str:
    """When mkdocs-jupyter is disabled, avoid nbconvert here too (hooks would still run nb2html)."""
    return (
        '<div class="admonition note">'
        '<p class="admonition-title">Notebook rendering skipped</p>'
        "<p>Run <code>./run_doc.sh</code> without <code>--fast</code> (or set "
        "<code>POWERZOO_DOCS_INCLUDE_NOTEBOOKS=true</code>) to render tutorial notebooks.</p>"
        "</div>"
    )


def _sync_zh_notebooks_from_en(config) -> None:
    """Copy English tutorial notebooks into zh/ so mkdocs-static-i18n can build zh pages."""
    root = Path(config.docs_dir)
    src = root / "en" / "examples" / "notebooks"
    dst = root / "zh" / "examples" / "notebooks"
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for f in dst.glob("*.ipynb"):
        f.unlink()
    for f in sorted(src.glob("*.ipynb")):
        shutil.copy2(f, dst / f.name)


def on_pre_build(*, config):
    """Mirror en notebooks to zh before collecting files (also done in run_doc.sh for serve)."""
    if _include_notebooks():
        _sync_zh_notebooks_from_en(config)


@event_priority(-101)
def on_files(files: Files, *, config):
    """Restore NotebookFile after i18n so Jupyter integration + render patch apply."""
    from mkdocs_jupyter.plugin import NotebookFile

    out = []
    for f in files:
        if f.src_uri.endswith(".ipynb") and not _include_notebooks():
            continue
        if f.src_uri.endswith(".ipynb") and not isinstance(f, NotebookFile):
            f = NotebookFile(f, **config)
        out.append(f)

    rebuilt = Files(out)
    # mkdocs-static-i18n uses I18nFiles.get_file_from_path() so nav paths like "index.md"
    # resolve to "en/index.md". Plain Files would drop that and nav hrefs become raw *.md.
    if type(files).__name__ == "I18nFiles" and getattr(files, "plugin", None) is not None:
        try:
            from mkdocs_static_i18n.folder import I18nFiles as I18nFilesFolder
        except ImportError:
            from mkdocs_static_i18n.suffix import I18nFiles as I18nFilesSuffix

            return I18nFilesSuffix(files.plugin, rebuilt)
        return I18nFilesFolder(files.plugin, rebuilt)
    return rebuilt


def _page_is_ipynb(page) -> bool:
    uri = getattr(page.file, "src_uri", "") or ""
    return uri.endswith(".ipynb")


def _notebook_abs_path(page, config) -> str:
    p = getattr(page.file, "abs_src_path", None)
    if p:
        return p
    return os.path.normpath(os.path.join(config.docs_dir, page.file.src_uri))


def _looks_like_nbconvert_html(html: str) -> bool:
    h = html.lstrip()
    if h.startswith("<script"):
        return True
    head = html[:6000]
    return "jp-Notebook" in head or "jp-RenderedHTMLCommon" in head


def _looks_like_raw_notebook_json(html: str) -> bool:
    h = html.lstrip()
    if h.startswith("{"):
        return '"cells"' in html[:1200]
    if h.startswith("<p>{"):
        return '"cells"' in html[:2000]
    return '"nbformat"' in html[:1500] and '"cells"' in html[:2000]


def on_page_content(html: str, *, page, config, files) -> str:
    if not _page_is_ipynb(page):
        return html
    if _looks_like_nbconvert_html(html):
        return html
    if not _looks_like_raw_notebook_json(html):
        return html

    if not _include_notebooks():
        return _notebook_skipped_placeholder()

    from mkdocs_jupyter.convert import nb2html

    jcfg: dict[str, Any] = {}
    jp = config.plugins.get("mkdocs-jupyter")
    if jp is not None:
        jcfg = dict(jp.config)

    nb_path = _notebook_abs_path(page, config)

    custom_mathjax = jcfg.get("custom_mathjax_url") or (
        "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js"
        "?config=TeX-AMS_CHTML-full,Safe"
    )

    return nb2html(
        nb_path,
        execute=bool(jcfg.get("execute", False)),
        kernel_name=str(jcfg.get("kernel_name", "") or ""),
        theme=str(jcfg.get("theme", "") or ""),
        allow_errors=bool(jcfg.get("allow_errors", True)),
        show_input=bool(jcfg.get("show_input", True)),
        no_input=bool(jcfg.get("no_input", False)),
        remove_tag_config=dict(jcfg.get("remove_tag_config") or {}),
        highlight_extra_classes=str(jcfg.get("highlight_extra_classes", "") or ""),
        include_requirejs=bool(jcfg.get("include_requirejs", False)),
        custom_mathjax_url=custom_mathjax,
    )


on_page_content.mkdocs_priority = -100  # type: ignore[attr-defined]

# Default must match mkdocs-jupyter / nbconvert ``mathjax.html.j2`` and ``on_page_content`` above.
_DEFAULT_MATHJAX_URL = (
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js"
    "?config=TeX-AMS_CHTML-full,Safe"
)


def _mathjax_url_from_config(config) -> str:
    jp = config.plugins.get("mkdocs-jupyter")
    if jp is not None:
        jcfg = dict(jp.config)
        u = str(jcfg.get("custom_mathjax_url") or "").strip()
        if u:
            return u
    return _DEFAULT_MATHJAX_URL


@event_priority(-102)
def on_post_page(html: str, *, page, config) -> str:
    """Restore MathJax ``src`` when the site build strips the CDN URL from notebook HTML."""
    if _page_is_ipynb(page):
        if "Load mathjax" in html and "MathJax configuration" in html:
            url = _mathjax_url_from_config(config)

            def _repl(m: re.Match[str]) -> str:
                current = (m.group(2) or "").strip()
                if current:
                    return m.group(0)
                return f'{m.group(1)}{url}{m.group(3)}'

            html = re.sub(
                r"(<!-- Load mathjax -->\s*<script\s+src=\")([^\"]*)(\"\s*>\s*</script>)",
                _repl,
                html,
                count=1,
                flags=re.IGNORECASE | re.DOTALL,
            )
        return html

    return html
