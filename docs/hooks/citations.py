"""MkDocs hook: APA citations from the central bibliography ``docs/references.bib``.

Syntax, in the Markdown of any page (outside code blocks, inline code and HTML comments):

* ``[@key]`` renders a parenthetical citation, "(Liu et al., 2024)". Several keys are separated by semicolons,
  ``[@key1; @key2]`` renders "(Liu et al., 2024; Cai et al., 2024)". An item may carry a prefix and a locator,
  ``[see @key, section 2.2]`` renders "(see DeepSeek-AI, 2024, section 2.2)".
* ``@key`` renders a narrative citation, "Liu et al. (2024)". Only words of the key shape ``name2024suffix``
  (:data:`NARRATIVE`) are taken as narrative citations, so decorators and e-mail addresses stay as they are.

Each citation links to its entry in the "References" list the hook appends to the page: every work cited on the page
once, in APA 7 reference format, sorted by author, with a DOI or URL. The in-text labels follow APA 7: one or two
author surnames, "et al." for three or more, initials when different first authors share a surname, more surnames
when two works of one year would abbreviate to the same label, and the suffixes a, b, ... for works of identical
authors and year. All of this is decided per page, from the works cited on that page.

A key that is not in the bibliography fails the build, with the page and the key in the message.

The bibliography is parsed by the small BibTeX reader :func:`parse_bibtex` (entries ``@type{key, field = value}``
with braced, quoted or bare values; no ``@string`` macros). Titles are converted to sentence case
(:func:`sentence_case`): words with capitals after their first letter (acronyms such as "LLMs" or "KnapSpec") and
text in braces keep their case.

The module is imported by path by ``mkdocs build`` (``hooks:`` in ``mkdocs.yml``) and by
``tests/docs/test_citations.py``.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The bibliography, relative to ``docs_dir``.
BIBLIOGRAPHY = "references.bib"
#: The heading of the reference list appended to each page that cites a work.
HEADING = "## References"
#: The anchor of the entry of ``key`` in the reference list is ``REF_PREFIX + key``.
REF_PREFIX = "ref-"
#: The shape of a key that is taken as a narrative citation when it follows an ``@``.
KEY_SHAPE = r"[a-z][a-z-]*\d{4}[a-z0-9]*"
#: A narrative citation ``@key``, not preceded by a word character, ``@``, ``.``, ``/`` or ``[``.
NARRATIVE = re.compile(rf"(?<![\w@./\[-])@(?P<key>{KEY_SHAPE})\b")
#: A bracketed citation ``[... @key ...; ...]``, not the text of a Markdown link or a link definition.
BRACKETED = re.compile(r"\[(?P<body>[^\[\]\n]*(?<=[\s;\[])@[^\[\]\n]*)\](?![(\[:])")
#: One item of a bracketed citation: prefix, key and locator.
_ITEM = re.compile(r"^(?P<prefix>(?:.*?\s)?)@(?P<key>[\w:.-]+?)(?P<locator>(?:,.*)?)$", re.S)
#: Fenced code blocks (any indentation), whose content is never touched.
_FENCE = re.compile(r"^[ \t>]*(?P<fence>`{3,}|~{3,})")
#: Inline code spans and HTML comments, whose content is never touched.
_INLINE_PROTECTED = re.compile(r"<!--.*?-->|(?P<ticks>`+).+?(?P=ticks)", re.S)
#: More authors than this are shortened to the first 19, an ellipsis and the last (APA 7).
MAX_LISTED_AUTHORS = 20

#: These BibTeX types render as a paper in proceedings.
_PROCEEDINGS = {"inproceedings", "conference"}
#: These render as a journal article.
_ARTICLE = {"article"}
#: These render as software.
_SOFTWARE = {"software"}
#: These render as a web page.
_ONLINE = {"online", "webpage"}


class CitationError(ValueError):
    """A malformed bibliography or a citation of a key that is not in it."""


# --------------------------------------------------------------------------------------------------------------------
# BibTeX


@dataclass(frozen=True)
class Name:
    """One author: ``last`` is the surname, or the whole name of a group author (``corporate``)."""

    last: str
    first: str = ""
    corporate: bool = False

    def initials(self) -> str:
        """APA initials of the given names: "Jean-François Kagy" -> "J.-F.", "Vinh Q." -> "V. Q."."""
        parts = []
        for word in self.first.split():
            pieces = [p for p in word.split("-") if p]
            parts.append("-".join(p[0] + "." for p in pieces))
        return " ".join(parts)

    def reference(self) -> str:
        """The name as the reference list writes it: "Surname, I. I." or the group's name."""
        if self.corporate or not self.first:
            return self.last
        return f"{self.last}, {self.initials()}"


@dataclass(frozen=True)
class Entry:
    """One bibliography entry: its BibTeX type, key, fields (lowercase names, raw values) and parsed authors."""

    type: str
    key: str
    fields: dict[str, str]
    authors: tuple[Name, ...] = field(default=())

    @property
    def year(self) -> str:
        return latex_to_text(self.fields.get("year", "n.d."))

    def get(self, name: str) -> str | None:
        value = self.fields.get(name)
        return None if value is None else latex_to_text(value)


def _read_value(text: str, i: int, key: str) -> tuple[str, int]:
    """The field value starting at ``text[i]`` (braced, quoted or bare) and the index after it."""
    if text[i] == "{":
        depth, j = 0, i
        while j < len(text):
            if text[j] == "\\":
                j += 2
                continue
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[i + 1 : j], j + 1
            j += 1
        raise CitationError(f"references: unbalanced braces in entry {key!r}")
    if text[i] == '"':
        depth, j = 0, i + 1
        while j < len(text):
            if text[j] == "\\":
                j += 2
                continue
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            elif text[j] == '"' and depth == 0:
                return text[i + 1 : j], j + 1
            j += 1
        raise CitationError(f"references: unterminated quoted value in entry {key!r}")
    match = re.compile(r"[\w.+-]+").match(text, i)
    if not match:
        raise CitationError(f"references: cannot read a value at {text[i : i + 20]!r} in entry {key!r}")
    if not match.group().isdigit():
        raise CitationError(f"references: bare value {match.group()!r} in entry {key!r} (no @string macros)")
    return match.group(), match.end()


def parse_bibtex(text: str) -> dict[str, Entry]:
    """The entries of a BibTeX file, by key. Text outside entries, ``@comment`` and ``@preamble`` are ignored."""
    entries: dict[str, Entry] = {}
    head = re.compile(r"@(?P<type>\w+)\s*[{(]")
    pos = 0
    while (match := head.search(text, pos)) is not None:
        kind = match.group("type").lower()
        i = match.end()
        if kind in {"comment", "preamble"}:
            pos = _read_value(text, i - 1, kind)[1] if text[i - 1] == "{" else text.find(")", i) + 1 or len(text)
            continue
        if kind == "string":
            raise CitationError("references: @string macros are not supported")
        key_match = re.compile(r"\s*(?P<key>[^\s,{}]+)\s*,").match(text, i)
        if not key_match:
            raise CitationError(f"references: entry without a key at {text[match.start() : match.start() + 40]!r}")
        key = key_match.group("key")
        i = key_match.end()
        fields: dict[str, str] = {}
        name_re = re.compile(r"\s*(?P<name>[\w-]+)\s*=\s*")
        while True:
            i = len(text) - len(text[i:].lstrip())
            if i >= len(text):
                raise CitationError(f"references: entry {key!r} is not closed")
            if text[i] in "})":
                i += 1
                break
            name_match = name_re.match(text, i)
            if not name_match:
                raise CitationError(f"references: cannot read a field at {text[i : i + 30]!r} in entry {key!r}")
            name = name_match.group("name").lower()
            value, i = _read_value(text, name_match.end(), key)
            if name in fields:
                raise CitationError(f"references: field {name!r} twice in entry {key!r}")
            fields[name] = " ".join(value.split())
            i = len(text) - len(text[i:].lstrip())
            if i < len(text) and text[i] == ",":
                i += 1
        if key in entries:
            raise CitationError(f"references: key {key!r} is defined twice")
        for required in ("author", "title", "year"):
            if required not in fields:
                raise CitationError(f"references: entry {key!r} has no {required}")
        entries[key] = Entry(kind, key, fields, parse_names(fields["author"]))
        pos = i
    return entries


def _split_depth0(text: str, separator: re.Pattern[str]) -> list[str]:
    """``text`` split at the matches of ``separator`` that are not inside braces."""
    parts, depth, start, i = [], 0, 0, 0
    while i < len(text):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif depth == 0 and (match := separator.match(text, i)):
            parts.append(text[start:i])
            start = i = match.end()
            continue
        i += 1
    parts.append(text[start:])
    return [p.strip() for p in parts]


def parse_names(text: str) -> tuple[Name, ...]:
    """The authors of a BibTeX ``author`` field: "Last, First" or "First Last", separated by " and ".

    A name entirely in braces (``{DeepSeek-AI}``) is a group author.
    """
    names = []
    for part in _split_depth0(text, re.compile(r"\s+and\s+")):
        if not part:
            continue
        if part.startswith("{") and part.endswith("}") and _balanced(part[1:-1]):
            names.append(Name(latex_to_text(part[1:-1]), corporate=True))
            continue
        pieces = _split_depth0(part, re.compile(r","))
        if len(pieces) >= 2:
            names.append(Name(latex_to_text(pieces[0]), latex_to_text(" ".join(pieces[1:]))))
        else:
            first, _, last = latex_to_text(part).rpartition(" ")
            names.append(Name(last, first))
    return tuple(names)


def _balanced(text: str) -> bool:
    depth = 0
    for char in text:
        depth += {"{": 1, "}": -1}.get(char, 0)
        if depth < 0:
            return False
    return depth == 0


_LATEX = {r"\&": "&", r"\%": "%", r"\_": "_", r"\$": "$", r"\#": "#", "~": "\u00a0", "---": "\u2014", "--": "\u2013"}


def _unescape(text: str) -> str:
    for latex, plain in _LATEX.items():
        text = text.replace(latex, plain)
    return text


def latex_to_text(text: str) -> str:
    """Plain text of a BibTeX value: escapes resolved, braces removed."""
    return " ".join(_unescape(text).replace("{", "").replace("}", "").split())


# --------------------------------------------------------------------------------------------------------------------
# APA formatting

_WORD = re.compile(r"[^\W\d_][\w'\u2019]*|[:?!]")


def sentence_case(title: str) -> str:
    """``title`` in APA sentence case.

    The first word and the first word after a colon keep an initial capital; other words are lowercased, except words
    with a capital after their first letter (acronyms, "KnapSpec") and text in braces. Each part of a hyphenated word
    is treated as a word, so "Training-Free" becomes "Training-free".
    """
    out: list[str] = []
    capitalize = True
    depth, start = 0, 0
    segments: list[tuple[bool, str]] = []
    for i, char in enumerate(title):
        if char == "{":
            if depth == 0:
                segments.append((False, title[start:i]))
                start = i + 1
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                segments.append((True, title[start:i]))
                start = i + 1
    segments.append((False, title[start:]))
    for protected, text in segments:
        if protected:
            plain = _unescape(text).replace("{", "").replace("}", "")
            if plain.strip():
                capitalize = plain.rstrip().endswith((":", "?", "!"))
            out.append(plain)
            continue
        text = _unescape(text)
        pieces, last = [], 0
        for match in _WORD.finditer(text):
            word = match.group()
            pieces.append(text[last : match.start()])
            last = match.end()
            if word in ":?!":
                capitalize = True
                pieces.append(word)
                continue
            if capitalize:
                pieces.append(word[0].upper() + word[1:])
                capitalize = False
            elif any(c.isupper() for c in word[1:]):
                pieces.append(word)
            else:
                pieces.append(word.lower())
        pieces.append(text[last:])
        out.append("".join(pieces))
    return " ".join("".join(out).split())


def _sort_text(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).casefold()


def _title_key(entry: Entry) -> tuple[str, ...]:
    """The title's words without surrounding punctuation, so that "EAGLE: ..." sorts before "EAGLE-2: ..."."""
    words = (w.strip(".,:;!?()'\"") for w in _sort_text(entry.get("title") or "").split())
    return tuple(w for w in words if w)


def sort_key(entry: Entry) -> tuple[Any, ...]:
    """APA order: by the authors' surnames and initials, then year, then title."""
    names = tuple((_sort_text(n.last), _sort_text(n.initials())) for n in entry.authors)
    return names, entry.year, _title_key(entry)


def _join(items: list[str], conjunction: str) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    return ", ".join(items[:-1]) + f", {conjunction} {items[-1]}"


@dataclass(frozen=True)
class Label:
    """The in-text author label of one work: the surnames shown and whether "et al." follows."""

    names: tuple[str, ...]
    et_al: bool

    def render(self, conjunction: str) -> str:
        if self.et_al:
            return ", ".join(self.names) + (", et al." if len(self.names) > 1 else " et al.")
        return _join(list(self.names), conjunction)


def _surname(name: Name, with_initials: bool) -> str:
    if with_initials and not name.corporate and name.first:
        return f"{name.initials()} {name.last}"
    return name.last


def labels(entries: list[Entry]) -> tuple[dict[str, Label], dict[str, str]]:
    """The in-text label and the year (with a, b, ... suffix) of each of ``entries``, cited together on one page."""
    # Initials for first authors that share a surname with a different first author (APA 7, 8.20).
    initials_by_surname: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        first = entry.authors[0]
        if not first.corporate:
            initials_by_surname[first.last].add(first.initials())
    with_initials = {last for last, initials in initials_by_surname.items() if len(initials) > 1}

    def names_of(entry: Entry, count: int) -> tuple[str, ...]:
        shown = [_surname(entry.authors[0], entry.authors[0].last in with_initials)]
        shown += [n.last for n in entry.authors[1:count]]
        return tuple(shown)

    def label_of(entry: Entry, count: int) -> Label:
        n = len(entry.authors)
        if n <= 2 or count >= n - 1:
            return Label(names_of(entry, n), et_al=False)
        return Label(names_of(entry, count), et_al=True)

    result = {e.key: label_of(e, 1) for e in entries}
    # More surnames when works of one year by different authors abbreviate to the same label (APA 7, 8.18).
    groups: dict[tuple[Label, str], list[Entry]] = defaultdict(list)
    for entry in entries:
        groups[result[entry.key], entry.year].append(entry)
    for group in groups.values():
        for entry in group:
            others = [o for o in group if o.authors != entry.authors]
            count = 1
            while count < len(entry.authors) and any(names_of(o, count) == names_of(entry, count) for o in others):
                count += 1
            result[entry.key] = label_of(entry, count)
    # Suffixes a, b, ... for works by the same authors in the same year, ordered by title.
    years: dict[str, str] = {}
    same: dict[tuple[tuple[Name, ...], str], list[Entry]] = defaultdict(list)
    for entry in entries:
        same[entry.authors, entry.year].append(entry)
    for (_, year), group in same.items():
        group.sort(key=_title_key)
        for index, entry in enumerate(group):
            years[entry.key] = year + (chr(ord("a") + index) if len(group) > 1 else "")
    return result, years


def _authors_reference(names: tuple[Name, ...]) -> str:
    """The author part of an APA reference."""
    written = [n.reference() for n in names]
    if len(written) > MAX_LISTED_AUTHORS:
        return ", ".join(written[:19]) + ", . . . " + written[-1]
    if len(written) == 1:
        return written[0]
    return ", ".join(written[:-1]) + ", & " + written[-1]


def _link(entry: Entry) -> str | None:
    if doi := entry.get("doi"):
        return f"https://doi.org/{doi}"
    eprint, prefix = entry.get("eprint"), (entry.get("archiveprefix") or "arXiv").lower()
    if eprint and prefix == "arxiv":
        return f"https://doi.org/10.48550/arXiv.{eprint}"
    return entry.get("url")


def reference(entry: Entry, year: str) -> str:
    """The APA 7 reference of ``entry`` as HTML (``year`` may carry a suffix)."""
    esc = html.escape
    authors = esc(_authors_reference(entry.authors))
    parts = [authors + (" " if authors.endswith(".") else ". ") + f"({esc(year)})."]
    title = esc(sentence_case(entry.fields["title"]))
    eprint = entry.get("eprint")
    if entry.type in _PROCEEDINGS:
        parts.append(f"{title}. In <em>{esc(entry.get('booktitle') or '')}</em>.")
    elif entry.type in _ARTICLE:
        volume = f", <em>{esc(v)}</em>" if (v := entry.get("volume")) else ""
        pages = f", {esc(p)}" if (p := entry.get("pages")) else ""
        parts.append(f"{title}. <em>{esc(entry.get('journal') or '')}</em>{volume}{pages}.")
    elif entry.type in _SOFTWARE:
        parts.append(f"<em>{title}</em> [Computer software].")
        if publisher := entry.get("publisher"):
            parts.append(f"{esc(publisher)}.")
    elif entry.type in _ONLINE:
        parts.append(f"<em>{title}</em>.")
        if publisher := entry.get("publisher"):
            parts.append(f"{esc(publisher)}.")
    elif eprint:
        parts.append(f"<em>{title}</em> (arXiv:{esc(eprint)}). arXiv.")
    else:
        parts.append(f"<em>{title}</em>.")
        if publisher := entry.get("publisher") or entry.get("howpublished"):
            parts.append(f"{esc(publisher)}.")
    if url := _link(entry):
        parts.append(f'<a href="{esc(url)}">{esc(url)}</a>')
    return " ".join(parts)


# --------------------------------------------------------------------------------------------------------------------
# Pages


@dataclass(frozen=True)
class _Item:
    key: str
    prefix: str = ""
    locator: str = ""


@dataclass(frozen=True)
class _Citation:
    """A citation found on a page: its span in the page and its items (``narrative`` for ``@key``)."""

    start: int
    end: int
    items: tuple[_Item, ...]
    narrative: bool


def _unprotected(markdown: str) -> list[tuple[int, int]]:
    """The spans of ``markdown`` outside fenced code blocks, inline code and HTML comments."""
    spans: list[tuple[int, int]] = []
    fence: str | None = None
    pos = start = 0
    for line in markdown.splitlines(keepends=True):
        match = _FENCE.match(line)
        if fence is None and match:
            spans.append((start, pos))
            fence = match.group("fence")
        elif fence is not None and match and match.group("fence")[0] == fence[0]:
            if len(match.group("fence")) >= len(fence) and not line[match.end() :].strip():
                fence = None
                start = pos + len(line)
        pos += len(line)
    if fence is None:
        spans.append((start, pos))
    result = []
    for begin, end in spans:
        last = begin
        for match in _INLINE_PROTECTED.finditer(markdown, begin, end):
            result.append((last, match.start()))
            last = match.end()
        result.append((last, end))
    return [(a, b) for a, b in result if b > a]


def find_citations(markdown: str) -> list[_Citation]:
    """The citations of a page, in order."""
    found: list[_Citation] = []
    for begin, end in _unprotected(markdown):
        chunk = markdown[begin:end]
        taken: list[tuple[int, int]] = []
        for match in BRACKETED.finditer(chunk):
            items = []
            for raw in match.group("body").split(";"):
                item = _ITEM.match(raw.strip())
                if item is None:
                    raise CitationError(f"cannot read the citation item {raw.strip()!r} in {match.group()!r}")
                locator = item.group("locator").lstrip(",").strip()
                items.append(_Item(item.group("key"), item.group("prefix").strip(), locator))
            found.append(_Citation(begin + match.start(), begin + match.end(), tuple(items), narrative=False))
            taken.append(match.span())
        for match in NARRATIVE.finditer(chunk):
            if any(a <= match.start() < b for a, b in taken):
                continue
            found.append(_Citation(begin + match.start(), begin + match.end(), (_Item(match.group("key")),), True))
    return sorted(found, key=lambda c: c.start)


def _anchor(key: str) -> str:
    return REF_PREFIX + re.sub(r"[^\w-]", "-", key)


def render_page(markdown: str, bibliography: dict[str, Entry], page: str = "page") -> str:
    """``markdown`` with its citations rendered and the reference list appended (unchanged if it cites nothing).

    Raises :class:`CitationError` naming ``page`` and every key that is not in ``bibliography``.
    """
    citations = find_citations(markdown)
    if not citations:
        return markdown
    keys = list(dict.fromkeys(item.key for c in citations for item in c.items))
    unknown = [k for k in keys if k not in bibliography]
    if unknown:
        listed = ", ".join(f"@{k}" for k in unknown)
        raise CitationError(f"{page}: unknown citation key(s) {listed} (not in docs/{BIBLIOGRAPHY})")
    entries = [bibliography[k] for k in keys]
    label, year = labels(entries)

    def link(key: str, text: str) -> str:
        return f'<a class="citation" href="#{_anchor(key)}">{text}</a>'

    out, last = [], 0
    for citation in citations:
        out.append(markdown[last : citation.start])
        last = citation.end
        if citation.narrative:
            key = citation.items[0].key
            out.append(link(key, f"{html.escape(label[key].render('and'))} ({year[key]})"))
            continue
        rendered: list[str] = []
        previous: Label | None = None
        for item in citation.items:
            when = year[item.key] + (f", {html.escape(item.locator)}" if item.locator else "")
            if rendered and previous == label[item.key] and not item.prefix:
                # Works of the same authors share one author label: "(Li et al., 2024a, 2024b)".
                rendered[-1] += ", " + link(item.key, when)
            else:
                prefix = f"{html.escape(item.prefix)} " if item.prefix else ""
                rendered.append(prefix + link(item.key, f"{html.escape(label[item.key].render('&'))}, {when}"))
            previous = label[item.key]
        out.append("(" + "; ".join(rendered) + ")")
    out.append(markdown[last:])
    listing = "\n".join(
        f'<li id="{_anchor(e.key)}">{reference(e, year[e.key])}</li>' for e in sorted(entries, key=sort_key)
    )
    body = "".join(out).rstrip("\n")
    return f'{body}\n\n{HEADING}\n\n<ul class="bibliography">\n{listing}\n</ul>\n'


# --------------------------------------------------------------------------------------------------------------------
# MkDocs events

_cache: dict[str, Any] = {"path": None, "mtime": None, "entries": {}}


def load_bibliography(path: Path) -> dict[str, Entry]:
    """The entries of the bibliography at ``path``, re-read when the file changes (``mkdocs serve``)."""
    mtime = path.stat().st_mtime
    if _cache["path"] != path or _cache["mtime"] != mtime:
        _cache.update(path=path, mtime=mtime, entries=parse_bibtex(path.read_text(encoding="utf-8")))
    return _cache["entries"]


def on_page_markdown(markdown: str, page: Any = None, config: Any = None, **kwargs: Any) -> str:
    """Render the citations of the page and append its reference list."""
    from mkdocs.exceptions import PluginError

    docs_dir = Path(config["docs_dir"]) if config is not None else Path(__file__).resolve().parents[1]
    name = page.file.src_uri if page is not None else "page"
    try:
        return render_page(markdown, load_bibliography(docs_dir / BIBLIOGRAPHY), name)
    except CitationError as error:
        raise PluginError(str(error)) from error
