"""The MkDocs hook ``docs/hooks/citations.py``: BibTeX parsing, APA 7 in-text citations and reference lists."""

from __future__ import annotations

import html
import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[2] / "docs" / "hooks" / "citations.py"
DOCS = HOOK.parents[1]


def load_hook() -> types.ModuleType:
    """The hook module, imported by path as ``mkdocs build`` does (registered in ``sys.modules`` for dataclasses)."""
    name = "citations_hook"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HOOK)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


hook = load_hook()

BIB = r"""
% A comment line outside any entry.
@comment{ignored, not an entry}

@inproceedings{frantar2023sparsegpt,
  author    = {Frantar, Elias and Alistarh, Dan},
  title     = {{SparseGPT}: Massive Language Models Can Be Accurately Pruned in One-Shot},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  year      = 2023,
}

@software{saxena2023promptlookup,
  author = "Apoorv Saxena",
  title  = {Prompt Lookup Decoding},
  year   = {2023},
  url    = {https://github.com/apoorvumang/prompt-lookup-decoding},
}

@misc{li2024eagle,
  author        = {Li, Yuhui and Wei, Fangyun and Zhang, Chao and Zhang, Hongyang},
  title         = {{EAGLE}: Speculative Sampling Requires Rethinking Feature Uncertainty},
  year          = {2024},
  eprint        = {2401.15077},
  archiveprefix = {arXiv},
}

@misc{li2024eagle2,
  author        = {Li, Yuhui and Wei, Fangyun and Zhang, Chao and Zhang, Hongyang},
  title         = {{EAGLE-2}: Faster Inference of Language Models with Dynamic Draft Trees},
  year          = {2024},
  eprint        = {2406.16858},
  archiveprefix = {arXiv},
}

@article{deepseek2024v3,
  author  = {{DeepSeek-AI}},
  title   = {{DeepSeek-V3} Technical Report},
  journal = {Journal of Examples},
  volume  = {3},
  pages   = {1--10},
  year    = {2024},
  doi     = {10.1000/example},
}

@online{google2026gemma4mtp,
  author    = {{Google}},
  title     = {Gemma 4 multi-token prediction drafters},
  year      = {2026},
  publisher = {Google AI for Developers},
  url       = {https://ai.google.dev/},
}
"""


@pytest.fixture(scope="module")
def bib() -> dict:
    return hook.parse_bibtex(BIB)


def render(markdown: str, bib: dict) -> str:
    return hook.render_page(markdown, bib, "test.md")


def in_text(markdown: str, bib: dict) -> str:
    """The rendered text before the reference list, with the citation links reduced to their (unescaped) text."""
    body = render(markdown, bib).split(hook.HEADING)[0].strip()
    return re.sub(r'<a class="citation" href="#ref-[\w-]+">(.*?)</a>', lambda m: html.unescape(m.group(1)), body)


# --------------------------------------------------------------------------------------------------------------------
# Parsing


def test_parses_entries_values_and_names(bib: dict) -> None:
    assert list(bib) == [
        "frantar2023sparsegpt",
        "saxena2023promptlookup",
        "li2024eagle",
        "li2024eagle2",
        "deepseek2024v3",
        "google2026gemma4mtp",
    ]
    sparsegpt = bib["frantar2023sparsegpt"]
    assert sparsegpt.type == "inproceedings"
    assert sparsegpt.year == "2023"  # a bare number
    assert [(n.last, n.first) for n in sparsegpt.authors] == [("Frantar", "Elias"), ("Alistarh", "Dan")]
    saxena = bib["saxena2023promptlookup"]
    assert [(n.last, n.first) for n in saxena.authors] == [("Saxena", "Apoorv")]  # quoted, "First Last"
    deepseek = bib["deepseek2024v3"].authors
    assert len(deepseek) == 1 and deepseek[0].corporate and deepseek[0].reference() == "DeepSeek-AI"
    assert bib["deepseek2024v3"].get("pages") == "1\u201310"


def test_initials() -> None:
    (name,) = hook.parse_names("Del Corro, Luciano")
    assert (name.last, name.reference()) == ("Del Corro", "Del Corro, L.")
    (name,) = hook.parse_names("Jean-François Ton")
    assert name.reference() == "Ton, J.-F."
    (name,) = hook.parse_names("Lee, Je-Yong Arthur")
    assert name.initials() == "J.-Y. A."


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("@string{venue = {ICML}}", "@string"),
        (
            "@misc{a2020x, author = {A, B}, title = {T}, year = {2020}}\n@misc{a2020x, author = {A, B}, title = {U}, "
            "year = {2020}}",
            "a2020x",
        ),
        ("@misc{a2020x, author = {A, B}, year = {2020}}", "title"),
        ("@misc{a2020x, author = {A, B}, title = {T}, title = {U}, year = {2020}}", "title"),
        ("@misc{a2020x, author = {A, B}, title = venue, year = {2020}}", "venue"),
    ],
)
def test_parse_errors(text: str, message: str) -> None:
    with pytest.raises(hook.CitationError, match=re.escape(message)):
        hook.parse_bibtex(text)


def test_sentence_case() -> None:
    assert hook.sentence_case("{SparseGPT}: Massive Language Models Can Be Accurately Pruned in One-Shot") == (
        "SparseGPT: Massive language models can be accurately pruned in one-shot"
    )
    assert hook.sentence_case("Fast Inference from Transformers via Speculative Decoding") == (
        "Fast inference from transformers via speculative decoding"
    )
    # Words with internal capitals stay as written.
    assert hook.sentence_case("Efficient LLMs with KV Caching") == "Efficient LLMs with KV caching"


# --------------------------------------------------------------------------------------------------------------------
# In-text citations


def test_one_author(bib: dict) -> None:
    assert in_text("Lookup [@saxena2023promptlookup].", bib) == "Lookup (Saxena, 2023)."
    assert in_text("As @saxena2023promptlookup shows.", bib) == "As Saxena (2023) shows."


def test_two_authors(bib: dict) -> None:
    assert in_text("[@frantar2023sparsegpt]", bib) == "(Frantar & Alistarh, 2023)"
    assert in_text("@frantar2023sparsegpt", bib) == "Frantar and Alistarh (2023)"


def test_three_or_more_authors_and_same_year_suffixes(bib: dict) -> None:
    assert in_text("EAGLE [@li2024eagle].", bib) == "EAGLE (Li et al., 2024)."
    # Two works by the same authors in one year get a and b (ordered by title) and share one author label.
    assert in_text("[@li2024eagle; @li2024eagle2]", bib) == "(Li et al., 2024a, 2024b)"
    assert in_text("@li2024eagle2 and @li2024eagle", bib) == "Li et al. (2024b) and Li et al. (2024a)"


def test_corporate_author(bib: dict) -> None:
    assert in_text("@deepseek2024v3", bib) == "DeepSeek-AI (2024)"


def test_multiple_keys_prefix_and_locator(bib: dict) -> None:
    text = in_text("[see @frantar2023sparsegpt; @saxena2023promptlookup, section 2]", bib)
    assert text == "(see Frantar & Alistarh, 2023; Saxena, 2023, section 2)"


def test_links_and_reference_list(bib: dict) -> None:
    out = render("A [@saxena2023promptlookup] and B [@frantar2023sparsegpt]. Again @frantar2023sparsegpt.", bib)
    assert out.count('href="#ref-frantar2023sparsegpt"') == 2
    _, listing = out.split(f"\n\n{hook.HEADING}\n\n")
    ids = re.findall(r'<li id="ref-([\w-]+)">', listing)
    assert ids == ["frantar2023sparsegpt", "saxena2023promptlookup"]  # alphabetical, each work once
    assert listing.startswith('<ul class="bibliography">') and listing.endswith("</ul>\n")


def test_shared_surnames_get_initials() -> None:
    bib = hook.parse_bibtex(
        "@misc{chen2023a, author = {Chen, Charlie and Borgeaud, S and Irving, G}, title = {A}, year = {2023}}\n"
        "@misc{chen2025b, author = {Chen, Lei and Wu, X and Liu, Y}, title = {B}, year = {2025}}\n"
    )
    assert in_text("[@chen2023a; @chen2025b]", bib) == "(C. Chen et al., 2023; L. Chen et al., 2025)"
    # Alone on a page, no other Chen needs telling apart.
    assert in_text("[@chen2023a]", bib) == "(Chen et al., 2023)"


def test_colliding_labels_list_more_surnames() -> None:
    bib = hook.parse_bibtex(
        "@misc{xu2024a, author = {Xu, A and Lin, B and Wu, C}, title = {A}, year = {2024}}\n"
        "@misc{xu2024b, author = {Xu, A and Ma, D and Wu, C}, title = {B}, year = {2024}}\n"
    )
    assert in_text("[@xu2024a; @xu2024b]", bib) == "(Xu, Lin, & Wu, 2024; Xu, Ma, & Wu, 2024)"


def test_code_comments_and_non_citations_are_left_alone(bib: dict) -> None:
    text = (
        "Mail me at someone@example.org, see [docs](a.md) and [@frantar2023sparsegpt].\n\n"
        "Inline `@saxena2023promptlookup` and <!-- @li2024eagle --> stay.\n\n"
        "```python\n@property\ndef f(x): ...  # @li2024eagle2\n```\n"
    )
    body = in_text(text, bib)
    assert "someone@example.org" in body and "(Frantar & Alistarh, 2023)" in body
    assert "`@saxena2023promptlookup`" in body and "<!-- @li2024eagle -->" in body
    assert "@property" in body and "# @li2024eagle2" in body
    listing = render(text, bib).split(hook.HEADING)[1]
    assert re.findall(r'<li id="ref-([\w-]+)">', listing) == ["frantar2023sparsegpt"]


def test_uncited_page_is_unchanged(bib: dict) -> None:
    text = "# Title\n\nNo citations here, only `@decorator` and a@b.c.\n"
    assert render(text, bib) == text


def test_unknown_key_raises(bib: dict) -> None:
    with pytest.raises(hook.CitationError, match=r"test\.md: unknown citation key\(s\) @nobody2020x, @none2021y"):
        render("[@frantar2023sparsegpt; @nobody2020x] and @none2021y", bib)


def test_plugin_error_from_the_mkdocs_event() -> None:
    from mkdocs.exceptions import PluginError

    page = types.SimpleNamespace(file=types.SimpleNamespace(src_uri="tutorials/x.md"))
    with pytest.raises(PluginError, match=r"tutorials/x\.md: unknown citation key\(s\) @nobody2020x"):
        hook.on_page_markdown("[@nobody2020x]", page=page, config={"docs_dir": str(DOCS)})


# --------------------------------------------------------------------------------------------------------------------
# Reference forms


def test_reference_forms(bib: dict) -> None:
    sparsegpt = hook.reference(bib["frantar2023sparsegpt"], "2023")
    assert sparsegpt == (
        "Frantar, E., &amp; Alistarh, D. (2023). SparseGPT: Massive language models can be accurately pruned in "
        "one-shot. In <em>Proceedings of the 40th International Conference on Machine Learning</em>."
    )
    eagle = hook.reference(bib["li2024eagle"], "2024a")
    assert eagle.startswith("Li, Y., Wei, F., Zhang, C., &amp; Zhang, H. (2024a). <em>EAGLE: Speculative sampling")
    assert "(arXiv:2401.15077). arXiv." in eagle
    assert eagle.endswith(
        '<a href="https://doi.org/10.48550/arXiv.2401.15077">https://doi.org/10.48550/arXiv.2401.15077</a>'
    )
    software = hook.reference(bib["saxena2023promptlookup"], "2023")
    assert "<em>Prompt lookup decoding</em> [Computer software]." in software
    assert 'href="https://github.com/apoorvumang/prompt-lookup-decoding"' in software
    article = hook.reference(bib["deepseek2024v3"], "2024")
    assert article.startswith(
        "DeepSeek-AI. (2024). DeepSeek-V3 technical report. <em>Journal of Examples</em>, <em>3</em>, 1\u201310."
    )
    assert 'href="https://doi.org/10.1000/example"' in article
    online = hook.reference(bib["google2026gemma4mtp"], "2026")
    assert "<em>Gemma 4 multi-token prediction drafters</em>. Google AI for Developers." in online


def test_more_than_twenty_authors_are_truncated() -> None:
    authors = " and ".join(f"Author{i:02d}, A" for i in range(1, 24))
    (entry,) = hook.parse_bibtex(f"@misc{{many2024x, author = {{{authors}}}, title = {{T}}, year = {{2024}}}}").values()
    text = hook.reference(entry, "2024")
    assert "Author19, A., . . . Author23, A. (2024)." in text
    assert "Author20" not in text


# --------------------------------------------------------------------------------------------------------------------
# The real bibliography and pages


def test_real_bibliography_parses_and_every_entry_is_cited() -> None:
    bib = hook.parse_bibtex((DOCS / hook.BIBLIOGRAPHY).read_text(encoding="utf-8"))
    cited: set[str] = set()
    for page in DOCS.rglob("*.md"):
        markdown = page.read_text(encoding="utf-8")
        hook.render_page(markdown, bib, str(page.relative_to(DOCS)))  # raises on an unknown key
        cited |= {item.key for c in hook.find_citations(markdown) for item in c.items}
    for entry in bib.values():
        assert hook.reference(entry, entry.year)
        assert hook._link(entry), f"{entry.key} has neither a DOI, an arXiv ID nor a URL"
    assert set(bib) == cited, f"uncited entries: {sorted(set(bib) - cited)}"
