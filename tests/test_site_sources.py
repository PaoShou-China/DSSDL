"""Validate static assets and archived-data exports without opening a browser."""

import csv
import json
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []
        self.ids = []
        self.anchors = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.append(attributes["id"])
        if "src" in attributes:
            self.sources.append(attributes["src"])
        href = attributes.get("href", "")
        if tag == "link" and not href.startswith("http"):
            self.sources.append(href)
        if href.startswith("#") and len(href) > 1:
            self.anchors.append(href[1:])


def test_local_assets_and_navigation_exist():
    parser = Assets()
    parser.feed((DOCS / "index.html").read_text(encoding="utf-8"))
    assert len(parser.ids) == len(set(parser.ids))
    assert all((DOCS / source).is_file() for source in parser.sources)
    assert set(parser.anchors) <= set(parser.ids)


def test_website_numbers_match_the_saved_metric_table():
    text = (DOCS / "assets/results.js").read_text(encoding="utf-8")
    data = json.loads(text.removeprefix("window.DSSDL_RESULTS = ").removesuffix(";\n"))
    with (ROOT / "revision_2026_11/results/wasserstein/reference.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        records = list(csv.DictReader(handle))
    for system, result in data.items():
        assert len(result["reference"]) == (351 if system == "lorenz" else 501)
        for method, points in result["methods"].items():
            assert len(points) == len(result["reference"]), method
        for item in result["wasserstein"]:
            original = next(
                r for r in records if r["system"] == system and r["label"] == item["method"]
            )
            expected = (
                float(original["sliced_wasserstein_1"])
                if original["sliced_wasserstein_1"]
                else None
            )
            assert item["value"] == expected
