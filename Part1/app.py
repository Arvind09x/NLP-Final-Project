from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fitness_reddit_analyzer.app_data import list_available_caches, load_dashboard_snapshot, resolve_cache_key
from fitness_reddit_analyzer.config import APP_TITLE


st.set_page_config(page_title=APP_TITLE, layout="wide")

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}
HERO_IMAGE_CANDIDATES = (
    Path("/Users/arvindsrinivass/Downloads/4ea73c75319148e3b2275fcf102e36a3.jpg"),
    Path("/Users/arvindsrinivass/Downloads/2b530d0302e87d964541b0765ec5f52b.jpg"),
    Path("/Users/arvindsrinivass/Desktop/Screenshot 2026-04-22 at 12.52.07 PM.png"),
)
PAGES = ("Overview", "Topics", "Topic Detail", "Methods", "RAG Chat", "Groq vs Gemini", "Hindi", "Bias", "Ethics")
PART2_PAGES = {"RAG Chat", "Groq vs Gemini", "Hindi", "Bias", "Ethics"}
PART2_SRC_ROOT = Path(__file__).resolve().parents[1] / "Part2" / "src"

CHART_COLORS = {
    "ink": "#000000",
    "ink_soft": "#292421",
    "muted": "#6b6259",
    "grid": "#d5c9ba",
    "line_soft": "#8e847a",
    "accent": "#260e0e",
    "accent_soft": "#9b8d80",
    "taupe": "#b8aa9a",
    "paper_warm": "#fbf8f4",
    "fill": "#ffffff",
}
CHART_SEQUENCE = [
    CHART_COLORS["accent"],
    CHART_COLORS["ink"],
    CHART_COLORS["accent_soft"],
    CHART_COLORS["taupe"],
    CHART_COLORS["line_soft"],
]


def get_hero_image_path() -> Path | None:
    for candidate in HERO_IMAGE_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def get_image_data_uri(path: Path) -> str:
    import base64

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lstrip(".").lower() or "jpeg"
    return f"data:image/{suffix};base64,{encoded}"


def page_url(page: str) -> str:
    return f"?page={quote(page)}"


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700;800&display=swap');
        :root {
            --main: #f4e9dc;
            --paper: #ffffff;
            --text: #000000;
            --muted: #5d554f;
            --muted-soft: #756c63;
            --line: rgba(31, 31, 31, 0.12);
            --line-strong: #1f1f1f;
            --shadow-soft: 0 2px 0 rgba(31, 31, 31, 0.12);
            --shadow-deep: 0 4px 4px rgba(0, 0, 0, 0.25);
        }
        html, body, [class*="css"]  {
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .stApp {
            background: var(--main);
            color: var(--text);
        }
        header[data-testid="stHeader"], [data-testid="stSidebar"], #MainMenu {
            display: none !important;
        }
        [data-testid="stAppViewContainer"] {
            background: transparent;
        }
        .block-container {
            max-width: 1512px;
            padding-top: 3.75rem;
            padding-bottom: 4rem;
            padding-left: clamp(1.4rem, 5.5vw, 5.25rem);
            padding-right: clamp(1.4rem, 5.5vw, 5.25rem);
        }
        h1, h2, h3, h4 {
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            color: var(--text);
            letter-spacing: 0;
        }
        p, li, label, div, span {
            color: var(--text);
        }
        .home-frame {
            min-height: 48.5rem;
        }
        .home-title-card {
            width: min(100%, 16.5rem);
            height: 4rem;
            margin: 0 auto 2.35rem auto;
            display: grid;
            place-items: center;
            background: var(--paper);
            border: 1px solid var(--line);
            box-shadow: var(--shadow-soft);
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 2.5rem;
            line-height: 1;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }
        .home-grid {
            display: grid;
            grid-template-columns: minmax(18rem, 38.3rem) 25rem;
            column-gap: clamp(3rem, 11vw, 11.8rem);
            align-items: start;
        }
        .home-nav-stack {
            display: grid;
            gap: 0.95rem;
        }
        .home-nav-stack div[data-testid="stButton"] {
            margin-bottom: 0.2rem;
        }
        .home-nav-stack div[data-testid="stButton"] button {
            min-height: 4.45rem;
            font-size: clamp(1.45rem, 2.2vw, 1.9rem);
        }
        .home-nav-label {
            margin: 0.15rem 0 0.45rem 0;
            color: var(--muted-soft);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .home-part-note {
            margin-top: 0.05rem;
            padding-top: 0.85rem;
            border-top: 1px solid var(--line);
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.55;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .figma-link {
            width: 100%;
            min-height: 3.35rem;
            display: grid;
            place-items: center;
            border: 1px solid var(--line);
            background: var(--paper);
            color: var(--text) !important;
            box-shadow: var(--shadow-soft);
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 1.35rem;
            line-height: 1;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            text-decoration: none !important;
            transition: transform 140ms ease, border-color 140ms ease, box-shadow 140ms ease;
        }
        .figma-link:hover {
            transform: translateY(-1px);
            border-color: var(--line-strong);
            color: var(--text) !important;
            box-shadow: var(--shadow-deep);
        }
        .figma-link.is-active {
            border-color: var(--line-strong);
        }
        .home-poster {
            width: min(100%, 25rem);
            height: 35rem;
            object-fit: cover;
            display: block;
        }
        .page-topbar {
            display: grid;
            grid-template-columns: minmax(13rem, 18rem) minmax(0, 1fr);
            gap: 1.5rem;
            align-items: start;
            margin-bottom: 1.55rem;
        }
        .page-title-card {
            min-height: 4rem;
            display: grid;
            place-items: center;
            background: var(--paper);
            border: 1px solid var(--line);
            box-shadow: var(--shadow-soft);
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 2.5rem;
            line-height: 1;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }
        .page-nav-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(8.25rem, 1fr));
            gap: 0.62rem;
        }
        .page-nav-row .figma-link {
            min-height: 2.9rem;
            font-size: 1.15rem;
            box-shadow: none;
        }
        .page-nav-row .figma-link.is-active {
            background: var(--paper);
            border-color: var(--line-strong);
            color: var(--text) !important;
            box-shadow: inset 0 -3px 0 var(--accent);
        }
        .rail-kicker, .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--muted-soft);
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .rail-title {
            margin-top: 0.5rem;
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 1.25rem;
            line-height: 1;
            font-weight: 400;
            letter-spacing: 0.03em;
        }
        .rail-copy, .subtle-text, .stCaption {
            color: var(--muted) !important;
            line-height: 1.6;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .hero-shell {
            padding: 1.25rem 0 0.75rem 0;
            margin-bottom: 1.2rem;
            border: 1px solid var(--line);
            border-left: 0;
            border-right: 0;
        }
        .hero-label {
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--muted-soft);
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .hero-title {
            margin: 0.45rem 0 0.65rem 0;
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: clamp(2.35rem, 5vw, 4.25rem);
            line-height: 0.95;
            letter-spacing: 0.02em;
        }
        .hero-subtitle {
            max-width: 54rem;
            font-size: 0.98rem;
            color: var(--muted);
            line-height: 1.65;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .hero-meta-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }
        .hero-chip {
            border: 1px solid var(--line);
            color: var(--text);
            background: var(--paper);
            border-radius: 0;
            padding: 0.5rem 0.8rem;
            font-size: 0.82rem;
            font-weight: 600;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .section-shell {
            padding: 1.25rem 0 0.25rem 0;
        }
        .section-title {
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 2rem;
            line-height: 1;
            letter-spacing: 0.02em;
            margin-bottom: 0.2rem;
        }
        .section-copy {
            color: var(--muted);
            max-width: 48rem;
            line-height: 1.7;
            margin-bottom: 1rem;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .metric-card {
            min-height: 8.1rem;
            padding: 1rem 0.1rem 1.1rem 0.1rem;
            border-top: 1px solid var(--line);
        }
        .metric-label {
            color: var(--muted-soft);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: 0.71rem;
            font-weight: 700;
            margin-bottom: 0.7rem;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .metric-value {
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 2.65rem;
            line-height: 0.95;
            font-weight: 400;
            color: var(--text);
        }
        .metric-foot {
            margin-top: 0.75rem;
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.55;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .panel-card, .doc-card, .stance-card, .note-card {
            margin-bottom: 0.9rem;
            padding: 1rem 0 1rem 0;
            border-top: 1px solid var(--line);
        }
        .panel-card.is-strong {
            border-top-color: var(--line-strong);
        }
        .panel-title {
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 1.35rem;
            line-height: 1;
            font-weight: 400;
            letter-spacing: 0.02em;
            margin-bottom: 0.35rem;
        }
        .doc-card {
            border-top-width: 1px;
        }
        .doc-title {
            font-family: "Inter", "Segoe UI", sans-serif;
            font-size: 1rem;
            font-weight: 700;
            margin: 0.25rem 0 0.4rem 0;
            line-height: 1.35;
        }
        .doc-meta, .caption-line {
            color: var(--muted);
            font-size: 0.84rem;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .doc-body {
            color: var(--text);
            line-height: 1.65;
            font-size: 0.95rem;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .topic-anchor {
            color: var(--text);
            text-decoration: none;
        }
        .topic-anchor:hover {
            color: var(--accent);
        }
        .pill {
            display: inline-block;
            padding: 0.42rem 0.78rem;
            margin: 0 0.45rem 0.45rem 0;
            border-radius: 0;
            border: 1px solid var(--line);
            background: var(--paper);
            color: var(--text);
            font-size: 0.78rem;
            font-weight: 600;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .topic-summary-line strong {
            color: var(--text);
        }
        .stPlotlyChart, div[data-testid="stDataFrame"] {
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: 0;
            padding: 0.9rem 0.9rem 0.25rem 0.9rem;
            box-shadow: var(--shadow-soft);
        }
        .topic-table-shell {
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: 0;
            padding: 1rem 1rem 0.8rem 1rem;
            overflow-x: auto;
            margin-bottom: 1rem;
            box-shadow: var(--shadow-soft);
        }
        .topic-table {
            width: 100%;
            min-width: 1180px;
            border-collapse: separate;
            border-spacing: 0;
            color: var(--text);
            font-size: 0.93rem;
        }
        .topic-table thead th {
            padding: 0.9rem 0.85rem;
            text-align: left;
            background: #f8f4ee;
            color: var(--text);
            font-weight: 700;
            border-bottom: 1px solid var(--line);
            white-space: nowrap;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .topic-table thead th:first-child {
            border-top-left-radius: 0;
        }
        .topic-table thead th:last-child {
            border-top-right-radius: 0;
        }
        .topic-table tbody td {
            padding: 0.9rem 0.85rem;
            border-bottom: 1px solid #e6ddd1;
            vertical-align: top;
            background: #ffffff;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .topic-table tbody tr:nth-child(even) td {
            background: #fbf8f4;
        }
        .topic-table tbody td.num {
            text-align: right;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }
        .topic-table tbody td.wrap {
            min-width: 220px;
        }
        .method-row {
            display: grid;
            grid-template-columns: 64px minmax(0, 1fr);
            gap: 1rem;
            padding: 1.25rem 0;
            border-top: 1px solid var(--line);
        }
        .method-index {
            width: 56px;
            height: 56px;
            border-radius: 0;
            display: grid;
            place-items: center;
            background: var(--paper);
            border: 1px solid var(--line-strong);
            color: var(--text);
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 1.5rem;
            font-weight: 400;
            letter-spacing: 0.02em;
            box-shadow: var(--shadow-soft);
        }
        .method-title {
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 1.45rem;
            line-height: 1;
            font-weight: 400;
            letter-spacing: 0.02em;
            margin-bottom: 0.35rem;
        }
        .method-list {
            margin: 0;
            padding-left: 1.05rem;
            color: var(--muted);
        }
        .method-list li {
            margin-bottom: 0.45rem;
            line-height: 1.65;
        }
        .note-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.9rem 1.5rem;
        }
        .note-card {
            margin-bottom: 0;
        }
        .rag-workbench {
            display: grid;
            grid-template-columns: minmax(0, 1.55fr) minmax(16rem, 0.65fr);
            gap: clamp(1.3rem, 3vw, 2.6rem);
            align-items: start;
            margin-top: 0.75rem;
        }
        .rag-control-panel {
            background: var(--paper);
            border: 1px solid var(--line-strong);
            box-shadow: var(--shadow-soft);
            padding: clamp(1rem, 2.2vw, 1.5rem);
            margin-top: 0.45rem;
            margin-bottom: 1rem;
        }
        .rag-board {
            background: var(--paper);
            border: 1px solid var(--line-strong);
            box-shadow: var(--shadow-soft);
            padding: clamp(1rem, 2.2vw, 1.5rem);
        }
        .rag-board-title {
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 2rem;
            line-height: 1;
            margin-bottom: 0.6rem;
        }
        .rag-rail {
            border-left: 1px solid var(--line-strong);
            padding-left: 1rem;
        }
        .rag-rail-item {
            padding: 0.9rem 0;
            border-top: 1px solid var(--line);
        }
        .rag-rail-label {
            color: var(--muted-soft);
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .rag-rail-value {
            margin-top: 0.25rem;
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 1.45rem;
            line-height: 1;
        }
        .source-strip {
            display: grid;
            grid-template-columns: 3.25rem minmax(0, 1fr);
            gap: 1rem;
            padding: 1rem 0;
            border-top: 1px solid var(--line);
        }
        .source-tag {
            width: 2.6rem;
            height: 2.1rem;
            display: grid;
            place-items: center;
            background: #f8f4ee;
            border: 1px solid var(--line-strong);
            box-shadow: var(--shadow-soft);
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 1.05rem;
        }
        .source-title {
            font-weight: 800;
            line-height: 1.35;
            margin-bottom: 0.35rem;
        }
        .source-meta {
            color: var(--muted);
            font-size: 0.78rem;
            margin-bottom: 0.45rem;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .source-snippet {
            color: var(--text);
            line-height: 1.58;
            font-size: 0.93rem;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .citation-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin: 0.75rem 0 0.25rem 0;
        }
        .report-callout {
            padding: 1rem 0;
            border-top: 1px solid var(--line);
            border-bottom: 1px solid var(--line);
            color: var(--muted);
            line-height: 1.65;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .comparison-row {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
        }
        .provider-table-shell {
            width: 100%;
            overflow-x: auto;
            border-top: 1px solid var(--line-strong);
            border-bottom: 1px solid var(--line-strong);
            margin: 1rem 0 1.6rem 0;
        }
        .provider-table {
            width: 100%;
            border-collapse: collapse;
            min-width: 46rem;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .provider-table th {
            text-align: left;
            color: var(--muted-soft);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: 0.72rem;
            padding: 0.85rem 0.75rem;
            border-bottom: 1px solid var(--line);
            white-space: nowrap;
        }
        .provider-table td {
            padding: 0.9rem 0.75rem;
            border-bottom: 1px solid var(--line);
            vertical-align: top;
            line-height: 1.45;
        }
        .provider-table td.num {
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }
        div[data-testid="stButton"] button {
            width: 100%;
            min-height: 3.35rem;
            border-radius: 0;
            border: 1px solid var(--line) !important;
            background: var(--paper) !important;
            color: var(--text) !important;
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 1.35rem;
            font-weight: 400;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            transition: transform 140ms ease, background 140ms ease, border-color 140ms ease, box-shadow 140ms ease;
            box-shadow: var(--shadow-soft);
        }
        div[data-testid="stButton"] button:hover {
            transform: translateY(-1px);
            border-color: var(--line-strong) !important;
            background: var(--paper) !important;
            box-shadow: var(--shadow-deep);
        }
        div[data-testid="stButton"] button * {
            color: inherit !important;
        }
        div[data-testid="stButton"] button[kind="primary"] {
            background: var(--paper) !important;
            color: var(--text) !important;
            border-color: var(--line-strong) !important;
            box-shadow: var(--shadow-soft);
        }
        div[data-testid="stButton"] button[kind="primary"] p,
        div[data-testid="stButton"] button[kind="primary"] span,
        div[data-testid="stButton"] button[kind="primary"] div {
            color: var(--text) !important;
        }
        div[data-testid="stButton"] button[kind="secondary"] p,
        div[data-testid="stButton"] button[kind="secondary"] span,
        div[data-testid="stButton"] button[kind="secondary"] div {
            color: var(--text) !important;
        }
        div[data-testid="stFormSubmitButton"] button {
            width: 100%;
            min-height: 3.35rem;
            border-radius: 0;
            border: 1px solid var(--line-strong) !important;
            background: var(--paper) !important;
            color: var(--text) !important;
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 1.35rem;
            font-weight: 400;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            box-shadow: var(--shadow-soft);
        }
        div[data-testid="stFormSubmitButton"] button:hover,
        div[data-testid="stFormSubmitButton"] button:focus,
        div[data-testid="stFormSubmitButton"] button:active {
            border-color: var(--line-strong) !important;
            background: var(--paper) !important;
            color: var(--text) !important;
            box-shadow: var(--shadow-deep);
        }
        div[data-testid="stFormSubmitButton"] button *,
        div[data-testid="stFormSubmitButton"] button p,
        div[data-testid="stFormSubmitButton"] button span,
        div[data-testid="stFormSubmitButton"] button div {
            color: var(--text) !important;
        }
        a, a:visited {
            color: var(--text);
        }
        a:hover {
            color: var(--accent);
        }
        .topic-detail-section {
            margin-top: 0.15rem;
            margin-bottom: 0.9rem;
        }
        .stSelectbox label,
        .stTextArea label,
        div[data-testid="stCheckbox"] label,
        div[data-testid="stToggle"] label {
            color: var(--muted-soft);
            text-transform: uppercase;
            font-size: 0.73rem;
            letter-spacing: 0.06em;
            font-weight: 700;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .stSelectbox [data-baseweb="select"] > div,
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child {
            min-height: 3.05rem;
            border-radius: 0;
            border-color: var(--line-strong) !important;
            background: var(--paper) !important;
            color: var(--text) !important;
        }
        .stSelectbox [data-baseweb="select"] *,
        .stTextArea textarea,
        .stTextArea textarea::placeholder,
        div[data-testid="stSelectbox"] *,
        div[data-testid="stTextArea"] * {
            color: var(--text) !important;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        div[data-testid="stSelectbox"] [data-baseweb="select"] input,
        div[data-testid="stSelectbox"] [data-baseweb="select"] span,
        div[data-testid="stSelectbox"] [data-baseweb="select"] div,
        div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
            color: var(--text) !important;
            fill: var(--text) !important;
        }
        .stTextArea textarea {
            border-radius: 0;
            border: 1px solid var(--line-strong) !important;
            background: var(--paper) !important;
            caret-color: var(--text);
        }
        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        ul[role="listbox"],
        li[role="option"] {
            background: var(--paper) !important;
            color: var(--text) !important;
        }
        div[data-baseweb="menu"] *,
        ul[role="listbox"] *,
        li[role="option"] * {
            color: var(--text) !important;
        }
        li[role="option"] {
            border-bottom: 1px solid var(--line) !important;
        }
        li[role="option"][aria-selected="true"],
        li[role="option"]:hover {
            background: #f8f4ee !important;
            color: var(--text) !important;
        }
        div[data-testid="stToggle"] p,
        div[data-testid="stCheckbox"] p,
        div[data-testid="stToggle"] span,
        div[data-testid="stCheckbox"] span {
            color: var(--text) !important;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        div[data-testid="stToggle"] [role="switch"] {
            background-color: #e6ddd1 !important;
            border: 1px solid var(--line-strong) !important;
        }
        div[data-testid="stToggle"] [aria-checked="true"] {
            background-color: var(--accent-soft) !important;
        }
        .stInfo {
            border-radius: 0;
            border: 1px solid var(--line);
            background: var(--paper);
            color: var(--text);
        }
        .stInfo *,
        .stWarning *,
        .stError *,
        div[data-testid="stAlert"] * {
            color: var(--text) !important;
        }
        .rag-answer-box {
            background: var(--paper) !important;
            border: 1px solid var(--line-strong) !important;
            box-shadow: var(--shadow-soft) !important;
            padding: clamp(1rem, 2.2vw, 1.5rem);
            margin-bottom: 0.75rem;
        }
        .rag-answer-text {
            color: var(--text) !important;
            line-height: 1.72;
            font-size: 1rem;
            font-family: "Inter", "Segoe UI", sans-serif;
            white-space: pre-wrap;
        }
        .rag-answer-box *,
        .rag-answer-box .rag-board-title,
        .rag-answer-box .rag-answer-text {
            background: transparent !important;
            color: var(--text) !important;
        }
        .rag-status-line {
            margin: 0.2rem 0 0.8rem 0;
            color: var(--muted) !important;
            font-size: 0.88rem;
            line-height: 1.5;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .detail-line {
            padding: 0.72rem 0;
            border-top: 1px solid var(--line);
            color: var(--muted) !important;
            font-size: 0.88rem;
            line-height: 1.55;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .detail-line strong {
            color: var(--text) !important;
        }
        .report-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.9rem 1.6rem;
            margin: 0.25rem 0 1rem 0;
        }
        .report-row {
            display: grid;
            grid-template-columns: minmax(8rem, 0.7fr) minmax(0, 1.3fr);
            gap: 1rem;
            padding: 0.95rem 0;
            border-top: 1px solid var(--line);
        }
        .report-label {
            color: var(--muted-soft);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: 0.72rem;
            font-weight: 800;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .report-body {
            color: var(--text);
            line-height: 1.6;
            font-size: 0.94rem;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .tag-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.45rem 0 1rem 0;
        }
        .takeaway {
            padding: 1rem 0;
            border-top: 1px solid var(--line-strong);
            border-bottom: 1px solid var(--line);
            font-size: 1rem;
            line-height: 1.7;
            color: var(--text);
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .rag-status-line strong {
            color: var(--text) !important;
        }
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li,
        div[data-testid="stMarkdownContainer"] span:not(.hero-chip):not(.pill),
        div[data-testid="stMarkdownContainer"] label,
        div[data-testid="stMarkdownContainer"] td,
        div[data-testid="stMarkdownContainer"] th {
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        @keyframes fadeSlideIn {
            from { opacity: 0; transform: translateY(12px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .section-shell, .hero-shell, .metric-card, .panel-card,
        .doc-card, .stance-card, .note-card, .method-row,
        .rag-board, .rag-control-panel, .provider-table-shell,
        .report-row, .takeaway {
            animation: fadeSlideIn 0.45s ease both;
        }
        .metric-card {
            transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
            border-left: 3px solid transparent;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-left-color: var(--line-strong);
            box-shadow: 0 3px 8px rgba(0,0,0,0.08);
        }
        .panel-card {
            transition: transform 160ms ease, box-shadow 160ms ease;
        }
        .panel-card:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        }
        .pill {
            transition: background 140ms ease, border-color 140ms ease, transform 140ms ease;
        }
        .pill:hover {
            background: #f0e8de;
            border-color: var(--line-strong);
            transform: translateY(-1px);
        }
        .hero-chip {
            transition: background 140ms ease, transform 140ms ease;
        }
        .hero-chip:hover {
            background: #f0e8de;
            transform: translateY(-1px);
        }
        .doc-card {
            transition: border-color 180ms ease;
        }
        .doc-card:hover {
            border-top-color: var(--line-strong);
        }
        .source-strip {
            transition: background 180ms ease;
        }
        .source-strip:hover {
            background: rgba(248,244,238,0.55);
        }
        .report-row {
            transition: background 160ms ease;
        }
        .report-row:hover {
            background: rgba(248,244,238,0.4);
        }
        .review-card {
            padding: 1rem 0;
            border-top: 1px solid var(--line);
            transition: border-color 180ms ease;
        }
        .review-card:hover {
            border-top-color: var(--line-strong);
        }
        .review-provider {
            display: inline-block;
            padding: 0.3rem 0.65rem;
            border: 1px solid var(--line-strong);
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 0.95rem;
            letter-spacing: 0.03em;
            margin-right: 0.5rem;
        }
        .review-scores {
            display: flex;
            gap: 1.2rem;
            margin: 0.5rem 0;
        }
        .review-score-item {
            font-size: 0.82rem;
            color: var(--muted);
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .review-score-item strong {
            color: var(--text);
            font-variant-numeric: tabular-nums;
        }
        .review-notes {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.6;
            font-style: italic;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .hindi-example {
            padding: 0.85rem 0;
            border-top: 1px solid var(--line);
            font-family: "Inter", "Segoe UI", sans-serif;
            transition: border-color 180ms ease;
        }
        .hindi-example:hover {
            border-top-color: var(--line-strong);
        }
        .hindi-source {
            font-size: 0.88rem;
            color: var(--muted);
            margin-bottom: 0.3rem;
        }
        .hindi-output {
            font-size: 0.95rem;
            color: var(--text);
            line-height: 1.55;
        }
        /* Staggered reveal for metric cards in a row */
        [data-testid="stHorizontalBlock"] > div:nth-child(1) .metric-card { animation-delay: 0ms; }
        [data-testid="stHorizontalBlock"] > div:nth-child(2) .metric-card { animation-delay: 60ms; }
        [data-testid="stHorizontalBlock"] > div:nth-child(3) .metric-card { animation-delay: 120ms; }
        [data-testid="stHorizontalBlock"] > div:nth-child(4) .metric-card { animation-delay: 180ms; }
        [data-testid="stHorizontalBlock"] > div:nth-child(5) .metric-card { animation-delay: 240ms; }
        [data-testid="stHorizontalBlock"] > div:nth-child(6) .metric-card { animation-delay: 300ms; }
        .page-divider {
            margin: 1.5rem 0 0.5rem 0;
            padding: 0.85rem 0;
            border-top: 2px solid var(--line-strong);
            border-bottom: 1px solid var(--line);
        }
        .page-divider-label {
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 1.15rem;
            letter-spacing: 0.03em;
            color: var(--muted-soft);
        }
        .topic-count-badge {
            display: inline-block;
            padding: 0.3rem 0.75rem;
            border: 1px solid var(--line-strong);
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 1.1rem;
            letter-spacing: 0.03em;
            margin-left: 0.6rem;
            vertical-align: middle;
        }
        @media (max-width: 1100px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .home-grid,
            .page-topbar {
                grid-template-columns: 1fr;
            }
            .home-grid {
                row-gap: 2rem;
            }
            .home-poster {
                width: min(100%, 25rem);
                height: auto;
                justify-self: start;
            }
            .page-nav-row {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }
            .note-grid,
            .rag-workbench,
            .comparison-row,
            .report-grid {
                grid-template-columns: 1fr;
            }
            .rag-rail {
                border-left: 0;
                padding-left: 0;
            }
        }
        @media (max-width: 700px) {
            .block-container {
                padding-top: 2rem;
            }
            .home-title-card,
            .page-title-card {
                font-size: 2.15rem;
            }
            .home-nav-stack {
                gap: 1rem;
            }
            .home-nav-stack div[data-testid="stButton"] {
                margin-bottom: 0;
            }
            .home-nav-stack div[data-testid="stButton"] button {
                min-height: 4.5rem;
            }
            .page-nav-row {
                grid-template-columns: 1fr;
            }
            .source-strip {
                grid-template-columns: 1fr;
            }
            .report-row {
                grid-template-columns: 1fr;
                gap: 0.35rem;
            }
        }
        /* --- Bottom-line callout --- */
        .bottom-line {
            border-left: 4px solid #000;
            padding: 1.1rem 1.4rem;
            background: rgba(0,0,0,0.03);
            margin-bottom: 1rem;
            animation: fadeSlideIn 0.3s ease both;
        }
        .bottom-line-label {
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 1.05rem;
            letter-spacing: 0.08em;
            color: var(--muted-soft);
            margin-bottom: 0.35rem;
        }
        .bottom-line-text {
            font-family: "Inter", sans-serif;
            font-size: 0.97rem;
            line-height: 1.55;
            color: var(--ink);
        }
        /* --- Probe matrix --- */
        .probe-matrix {
            width: 100%;
            border-collapse: collapse;
            font-family: "Inter", sans-serif;
            font-size: 0.82rem;
            animation: fadeSlideIn 0.3s ease both;
        }
        .probe-matrix thead th {
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 0.92rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 0.65rem 0.7rem;
            border-bottom: 2px solid var(--line-strong);
            text-align: left;
            color: var(--muted-soft);
        }
        .probe-matrix tbody td {
            padding: 0.6rem 0.7rem;
            border-bottom: 1px solid var(--line);
            vertical-align: top;
            line-height: 1.45;
        }
        .probe-matrix tbody tr:hover {
            background: rgba(0,0,0,0.02);
        }
        .probe-matrix .risk-highlight {
            background: rgba(0,0,0,0.04);
            border-left: 3px solid #000;
        }
        .probe-matrix .risk-area {
            font-weight: 600;
            white-space: nowrap;
        }
        /* --- Lifecycle timeline --- */
        .lifecycle-timeline {
            display: flex;
            align-items: center;
            gap: 0;
            padding: 1rem 0;
            overflow-x: auto;
            animation: fadeSlideIn 0.3s ease both;
        }
        .lifecycle-node {
            display: flex;
            flex-direction: column;
            align-items: center;
            min-width: 5.5rem;
        }
        .lifecycle-icon {
            width: 2.5rem;
            height: 2.5rem;
            border: 2px solid #000;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            background: var(--paper-warm);
            margin-bottom: 0.35rem;
        }
        .lifecycle-label {
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 0.82rem;
            letter-spacing: 0.04em;
            text-align: center;
            color: var(--ink);
        }
        .lifecycle-arrow {
            font-size: 1.3rem;
            color: var(--muted-soft);
            margin: 0 0.15rem;
            padding-bottom: 1.2rem;
        }
        /* --- Deletion checklist --- */
        .deletion-step {
            display: grid;
            grid-template-columns: 2.2rem 1fr;
            gap: 0.6rem;
            align-items: start;
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--line);
            animation: fadeSlideIn 0.3s ease both;
        }
        .deletion-step:last-child {
            border-bottom: none;
        }
        .step-number {
            width: 2rem;
            height: 2rem;
            border: 2px solid #000;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 1.05rem;
            background: var(--paper-warm);
        }
        .step-content {
            padding-top: 0.2rem;
        }
        .step-title {
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 0.95rem;
            letter-spacing: 0.04em;
            margin-bottom: 0.15rem;
        }
        .step-detail {
            font-family: "Inter", sans-serif;
            font-size: 0.85rem;
            color: var(--muted-soft);
            line-height: 1.4;
        }
        /* --- Risk matrix --- */
        .risk-matrix {
            width: 100%;
            border-collapse: collapse;
            font-family: "Inter", sans-serif;
            font-size: 0.82rem;
            animation: fadeSlideIn 0.3s ease both;
        }
        .risk-matrix thead th {
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 0.92rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 0.65rem 0.7rem;
            border-bottom: 2px solid var(--line-strong);
            text-align: left;
            color: var(--muted-soft);
        }
        .risk-matrix tbody td {
            padding: 0.6rem 0.7rem;
            border-bottom: 1px solid var(--line);
            vertical-align: top;
            line-height: 1.45;
        }
        .risk-matrix tbody tr:hover {
            background: rgba(0,0,0,0.02);
        }
        .risk-matrix .risk-name {
            font-weight: 600;
            white-space: nowrap;
        }
        /* --- Scope tag --- */
        .scope-tag {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 0.82rem;
            letter-spacing: 0.04em;
            border: 1px solid var(--line-strong);
        }
        .scope-tag.classroom { background: rgba(0,0,0,0.03); }
        .scope-tag.production { background: rgba(0,0,0,0.08); }
        /* ── Methods sub-page system ── */
        .methods-landing-text {
            max-width: 52rem;
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.65;
            margin-bottom: 1.2rem;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .methods-diagram {
            background: var(--paper);
            border: 1px solid var(--line-strong);
            box-shadow: var(--shadow-soft);
            padding: clamp(1rem, 2.5vw, 1.8rem);
            margin-bottom: 1.2rem;
            animation: fadeSlideIn 0.4s ease both;
        }
        .md-flow {
            display: flex;
            align-items: center;
            gap: 0.35rem;
            flex-wrap: wrap;
            padding: 0.3rem 0;
        }
        .md-node {
            border: 1px solid var(--line-strong);
            background: var(--paper);
            padding: 0.55rem 0.85rem;
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 0.98rem;
            letter-spacing: 0.02em;
            line-height: 1.15;
            text-align: center;
            min-width: 6.5rem;
            transition: transform 140ms ease, box-shadow 140ms ease;
        }
        .md-node:hover {
            transform: translateY(-2px);
            box-shadow: 0 3px 8px rgba(0,0,0,0.1);
        }
        .md-node span {
            display: block;
            font-family: "Inter", "Segoe UI", sans-serif;
            font-size: 0.72rem;
            color: var(--muted);
            margin-top: 0.15rem;
            letter-spacing: 0;
        }
        .md-node.md-src {
            border-width: 2px;
            background: #f8f4ee;
        }
        .md-node.md-out {
            border-width: 2px;
            background: rgba(0,0,0,0.04);
        }
        .md-arrow {
            font-size: 1.3rem;
            color: var(--muted-soft);
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            user-select: none;
        }
        .md-arrow-v {
            font-size: 1.3rem;
            color: var(--muted-soft);
            text-align: center;
            padding: 0.15rem 0;
            user-select: none;
        }
        .rag-arch-diagram {
            padding: clamp(1.2rem, 3vw, 2rem);
        }
        .rag-arch-title {
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 1.6rem;
            letter-spacing: 0.03em;
            margin-bottom: 1rem;
            text-align: center;
        }
        .rag-arch-grid {
            display: grid;
            grid-template-columns: 1fr 1px 1.4fr;
            gap: 1.5rem;
        }
        .rag-arch-col {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.15rem;
        }
        .rag-arch-phase {
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 1.1rem;
            letter-spacing: 0.06em;
            color: var(--muted-soft);
            text-transform: uppercase;
            margin-bottom: 0.5rem;
            text-align: center;
        }
        .rag-arch-divider {
            background: var(--line-strong);
            width: 1px;
        }
        .rag-arch-split {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            justify-content: center;
        }
        .rag-arch-plus, .rag-arch-or {
            font-family: "Bebas Neue", "Arial Narrow", sans-serif;
            font-size: 1rem;
            color: var(--muted-soft);
        }
        .rag-arch-note {
            margin-top: 0.8rem;
            text-align: center;
            font-family: "Inter", "Segoe UI", sans-serif;
            font-size: 0.82rem;
            color: var(--muted);
            font-style: italic;
        }
        @media (max-width: 900px) {
            .rag-arch-grid {
                grid-template-columns: 1fr;
                gap: 1rem;
            }
            .rag-arch-divider {
                height: 1px;
                width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, foot: str) -> None:
    label_text = html.escape(label)
    value_text = html.escape(value)
    foot_text = html.escape(foot)
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label_text}</div>
            <div class="metric-value">{value_text}</div>
            <div class="metric-foot">{foot_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_card(title: str, body: str) -> None:
    title_text = html.escape(title)
    body_text = html.escape(body)
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-title">{title_text}</div>
            <div class="subtle-text">{body_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def report_rows(rows: list[tuple[str, str]]) -> None:
    row_html = "".join(
        '<div class="report-row">'
        f'<div class="report-label">{html.escape(label)}</div>'
        f'<div class="report-body">{html.escape(body)}</div>'
        "</div>"
        for label, body in rows
    )
    st.markdown(f'<div>{row_html}</div>', unsafe_allow_html=True)


def takeaway(text: str) -> None:
    st.markdown(f'<div class="takeaway">{html.escape(text)}</div>', unsafe_allow_html=True)


def tag_strip(tags: list[str]) -> None:
    tags_html = "".join(f'<span class="pill">{html.escape(tag)}</span>' for tag in tags)
    st.markdown(f'<div class="tag-strip">{tags_html}</div>', unsafe_allow_html=True)


def load_json_artifact(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def format_metric(value, *, digits: int = 4, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if percent:
        return f"{number * 100:.2f}%"
    return f"{number:.{digits}f}"


def get_part2_adapter():
    if str(PART2_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(PART2_SRC_ROOT))
    from part2_rag import streamlit_adapter

    return streamlit_adapter


def get_provider_status(provider_name: str, adapter) -> tuple[bool, str]:
    adapter_status = getattr(adapter, "get_provider_status", None)
    if callable(adapter_status):
        return adapter_status(provider_name)
    if str(PART2_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(PART2_SRC_ROOT))
    from part2_rag.llm_providers import get_provider_configuration_status

    return get_provider_configuration_status(provider_name)


def get_provider_debug_summary(provider_name: str, adapter) -> list[tuple[str, str]]:
    provider_ok, provider_message = get_provider_status(provider_name, adapter)
    normalized = provider_name.strip().lower()
    if normalized == "groq":
        key_visible = bool(os.environ.get("GROQ_API_KEY"))
        model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    elif normalized in {"gemini", "google", "google-ai-studio"}:
        key_visible = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    else:
        key_visible = False
        model_name = "unknown"
    return [
        ("Provider", normalized),
        ("Key visible", "yes" if key_visible else "no"),
        ("Model", model_name),
        ("Config", provider_message if provider_ok else f"not ready: {provider_message}"),
    ]


def show_rag_unavailable(exc: BaseException) -> None:
    st.markdown(
        """
        <div class="rag-board">
            <div class="rag-board-title">RAG Bench Not Loaded</div>
            <div class="subtle-text">
                Part 1 is still available. The Part 2 RAG modules or runtime dependencies
                were not available in this Streamlit environment.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.error(str(exc))


def render_status_rail(items: list[tuple[str, str]]) -> None:
    rows = "".join(
        f"""
        <div class="rag-rail-item">
            <div class="rag-rail-label">{html.escape(label)}</div>
            <div class="rag-rail-value">{html.escape(value)}</div>
        </div>
        """
        for label, value in items
    )
    st.markdown(f'<div class="rag-rail">{rows}</div>', unsafe_allow_html=True)


def render_source_strips(snippets) -> None:
    if not snippets:
        st.info("No source previews were returned.")
        return
    for snippet in snippets:
        title = snippet.title or "Untitled r/fitness source"
        st.markdown(
            f"""
            <div class="source-strip">
                <div class="source-tag">{html.escape(snippet.source_label)}</div>
                <div>
                    <div class="source-title">{html.escape(title)}</div>
                    <div class="source-meta">
                        {html.escape(snippet.source_type.replace("_", " ").title())}
                        • {html.escape(snippet.retrieval_source)}
                        • score {float(snippet.score):.4f}
                        • chunk {html.escape(snippet.chunk_id)}
                    </div>
                    <div class="source-snippet">{html.escape(snippet.snippet)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_citation_tags(citations) -> None:
    if not citations:
        st.markdown('<div class="subtle-text">No citations returned.</div>', unsafe_allow_html=True)
        return
    tags = "".join(
        f'<span class="pill">{html.escape(citation.source_label)} • {html.escape(citation.source_type)}</span>'
        for citation in citations
    )
    st.markdown(f'<div class="citation-row">{tags}</div>', unsafe_allow_html=True)


def render_context_status_line(*, snippet_count: int, provider: str, evidence_only: bool) -> None:
    if evidence_only:
        status = (
            f"Context retrieved: {snippet_count} snippets. Evidence-only mode is on, "
            "so no LLM was called."
        )
    else:
        status = f"Context retrieved: {snippet_count} snippets sent to {provider.title()}."
    st.markdown(
        f'<div class="rag-status-line">{html.escape(status)}</div>',
        unsafe_allow_html=True,
    )


def render_rag_toggle_guide() -> None:
    st.markdown('<div class="eyebrow">Controls</div>', unsafe_allow_html=True)
    panel_card("Provider", "Chooses the LLM endpoint for generation. Retrieval is identical for Groq and Gemini.")
    panel_card("Evidence only", "Runs classification plus retrieval, then stops before any provider call.")
    panel_card("Save raw response", "Stores the provider payload and prompt under Part2/data/runs for debugging.")
    panel_card("Show prompt", "Displays the exact retrieved-context prompt sent to the provider.")


def doc_card(doc: dict, *, label: str | None = None) -> None:
    badge = html.escape(label or doc["source_type"].replace("_", " ").title())
    title_text = html.escape(doc["title"])
    author_text = html.escape(doc.get("author_name", ""))
    excerpt_text = html.escape(doc.get("excerpt", ""))
    created_text = html.escape(doc.get("created_at", ""))
    title_html = (
        f'<a class="topic-anchor" href="{html.escape(doc["reddit_url"])}" target="_blank">{title_text}</a>'
        if doc.get("reddit_url")
        else title_text
    )
    st.markdown(
        f"""
        <div class="doc-card">
            <span class="pill">{badge}</span>
            <div class="doc-title">{title_html}</div>
            <div class="doc-meta">{created_text} • {author_text} • score {doc.get("score", 0)}</div>
            <div class="doc-body">{excerpt_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_int(value: int) -> str:
    return f"{value:,}"


def apply_chart_style(figure: go.Figure, *, height: int, legend_y: float = 1.1) -> go.Figure:
    figure.update_layout(
        template="simple_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": CHART_COLORS["ink_soft"], "family": "Inter, sans-serif"},
        colorway=CHART_SEQUENCE,
        margin={"l": 24, "r": 24, "t": 34, "b": 26},
        legend={
            "orientation": "h",
            "y": legend_y,
            "x": 0.0,
            "title": None,
            "font": {"size": 11, "color": CHART_COLORS["ink_soft"]},
        },
        height=height,
        hoverlabel={
            "bgcolor": CHART_COLORS["fill"],
            "bordercolor": CHART_COLORS["grid"],
            "font": {"color": CHART_COLORS["ink"], "family": "Inter, sans-serif"},
        },
    )
    figure.update_xaxes(
        showgrid=False,
        linecolor=CHART_COLORS["grid"],
        mirror=False,
        tickfont={"color": CHART_COLORS["muted"]},
        title_font={"color": CHART_COLORS["ink_soft"]},
    )
    figure.update_yaxes(
        gridcolor=CHART_COLORS["grid"],
        linecolor=CHART_COLORS["grid"],
        zeroline=False,
        tickfont={"color": CHART_COLORS["muted"]},
        title_font={"color": CHART_COLORS["ink_soft"]},
    )
    return figure


def build_activity_chart(monthly_activity: list[dict]) -> go.Figure:
    frame = pd.DataFrame(monthly_activity)
    frame["month_start"] = pd.to_datetime(frame["month_start"])
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["month_start"],
            y=frame["posts"],
            mode="lines+markers",
            name="Posts",
            line={"color": CHART_COLORS["ink"], "width": 3},
            marker={"size": 7, "color": CHART_COLORS["ink"]},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["month_start"],
            y=frame["comments"],
            mode="lines+markers",
            name="Comments",
            line={"color": CHART_COLORS["accent"], "width": 2.75},
            marker={"size": 6, "color": CHART_COLORS["accent"]},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["month_start"],
            y=frame["modeled_documents"],
            mode="lines+markers",
            name="Hybrid Topic Corpus",
            line={"color": CHART_COLORS["line_soft"], "width": 2.75, "dash": "dot"},
            marker={"size": 7, "color": CHART_COLORS["line_soft"]},
        )
    )
    figure.update_layout(xaxis_title="Month", yaxis_title="Count")
    return apply_chart_style(figure, height=360, legend_y=1.14)


def build_topic_share_chart(topics: list[dict]) -> go.Figure:
    frame = pd.DataFrame(
        [
            {
                "topic_label": f"{topic['topic_id']}: {topic['topic_label']}",
                "document_share_pct": topic["document_share_pct"],
                "major_topic": "Major stance topic" if topic["major_topic"] else "Other topic",
            }
            for topic in sorted(topics, key=lambda item: item["document_share_pct"], reverse=True)
        ]
    )
    figure = px.bar(
        frame,
        x="document_share_pct",
        y="topic_label",
        color="major_topic",
        orientation="h",
        color_discrete_map={
            "Major stance topic": CHART_COLORS["ink"],
            "Other topic": CHART_COLORS["taupe"],
        },
        labels={"document_share_pct": "Document share (%)", "topic_label": ""},
    )
    figure.update_layout(showlegend=False)
    figure.update_yaxes(categoryorder="total ascending", tickfont={"color": CHART_COLORS["ink_soft"]})
    figure.update_traces(
        hovertemplate="%{y}<br>Document share: %{x:.1f}%<extra></extra>",
        marker_line_color="#cbbeb0",
        marker_line_width=0.6,
    )
    return apply_chart_style(figure, height=430)


def build_topic_detail_chart(topic: dict) -> go.Figure:
    frame = pd.DataFrame(topic["monthly_trend"])
    frame["month_start"] = pd.to_datetime(frame["month_start"])
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=frame["month_start"],
            y=frame["document_count"],
            name="Document count",
            marker_color=CHART_COLORS["paper_warm"],
            marker_line={"color": "#c8baa9", "width": 1},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["month_start"],
            y=frame["document_share"],
            name="Document share (%)",
            mode="lines+markers",
            yaxis="y2",
            line={"color": CHART_COLORS["ink"], "width": 3},
            marker={"size": 7, "color": CHART_COLORS["accent"]},
        )
    )
    figure.update_layout(
        xaxis_title="Month",
        yaxis_title="Document count",
        yaxis2={
            "title": "Share (%)",
            "overlaying": "y",
            "side": "right",
            "tickfont": {"color": CHART_COLORS["muted"]},
            "title_font": {"color": CHART_COLORS["ink_soft"]},
        },
    )
    return apply_chart_style(figure, height=430, legend_y=1.14)


def build_source_mix_chart(topic: dict) -> go.Figure:
    frame = pd.DataFrame(
        [
            {"source": "Posts", "count": topic["post_documents"]},
            {"source": "Comments", "count": topic["comment_documents"]},
        ]
    )
    figure = px.pie(
        frame,
        values="count",
        names="source",
        hole=0.62,
        color="source",
        color_discrete_map={"Posts": CHART_COLORS["ink"], "Comments": CHART_COLORS["accent"]},
    )
    figure.update_traces(
        textposition="outside",
        textinfo="label+percent",
        textfont={"color": CHART_COLORS["ink"], "size": 12},
        marker={"line": {"color": "#f3eee7", "width": 2}},
        hovertemplate="%{label}: %{value:,} documents (%{percent})<extra></extra>",
    )
    return apply_chart_style(figure, height=340, legend_y=-0.08)


def render_topic_table(rows: list[dict]) -> None:
    headers = [
        ("Topic", ""),
        ("Label", "wrap"),
        ("Top keywords", "wrap"),
        ("Document share (%)", "num"),
        ("Post share among modeled posts (%)", "num"),
        ("Post share among all scraped posts (%)", "num"),
        ("Documents", "num"),
        ("Posts", "num"),
        ("Comments", "num"),
        ("Trend", ""),
        ("Major topic", ""),
        ("Stance", ""),
    ]
    body_rows: list[str] = []
    for row in rows:
        cells = [
            ("num", str(row["topic_id"])),
            ("wrap", html.escape(str(row["topic_label"]))),
            ("wrap", html.escape(str(row["top_keywords"]))),
            ("num", f"{float(row['document_share_pct']):.1f}"),
            ("num", f"{float(row['modeled_post_share_pct']):.1f}"),
            ("num", f"{float(row['corpus_post_share_pct']):.2f}"),
            ("num", format_int(int(row["document_count"]))),
            ("num", format_int(int(row["post_documents"]))),
            ("num", format_int(int(row["comment_documents"]))),
            ("", html.escape(str(row["trend_label"]))),
            ("", "Yes" if bool(row["major_topic"]) else "No"),
            ("", html.escape(str(row["stance_status"]))),
        ]
        body_rows.append(
            "<tr>"
            + "".join(f'<td class="{klass}">{value}</td>' for klass, value in cells)
            + "</tr>"
        )

    header_html = "".join(f"<th>{html.escape(label)}</th>" for label, _ in headers)
    table_html = (
        '<div class="topic-table-shell"><table class="topic-table">'
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_section_intro(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-shell">
            <div class="section-title">{html.escape(title)}</div>
            <div class="section-copy">{html.escape(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_link_row(current_page: str) -> None:
    links = []
    for page in PAGES:
        active_class = " is-active" if page == current_page else ""
        links.append(f'<a class="figma-link{active_class}" href="{page_url(page)}" target="_self">{html.escape(page)}</a>')
    st.markdown(f'<div class="page-nav-row">{"".join(links)}</div>', unsafe_allow_html=True)


def render_cache_selector(available_caches: list, current_key: str) -> str:
    current = current_key
    labels = {option.key: option.label for option in available_caches}
    options = [option.key for option in available_caches]
    if len(options) <= 1:
        return current

    st.markdown('<div class="eyebrow">Era</div>', unsafe_allow_html=True)
    selector_cols = st.columns(len(options))
    for column, option_key in zip(selector_cols, options):
        with column:
            label = labels[option_key].replace(": ", "\n", 1)
            if st.button(
                label,
                key=f"cache_{option_key}",
                use_container_width=True,
                type="primary" if option_key == current_key else "secondary",
            ):
                current = option_key
    return current


def render_page_header(snapshot: dict, available_caches: list, selected_cache_key: str, page: str, *, show_era_selector: bool = True) -> str:
    cache_label = snapshot["meta"].get("cache_label", "Default cache")
    window_label = snapshot["stats"]["window_label"]
    source = snapshot["meta"].get("cache_source", "unknown")
    st.markdown(
        f"""
        <div class="page-topbar">
            <div class="page-title-card">{html.escape(page)}</div>
            <div class="page-nav-row">{''.join(
                f'<a class="figma-link{" is-active" if nav_page == page else ""}" href="{page_url(nav_page)}" target="_self">{html.escape(nav_page)}</a>'
                for nav_page in PAGES
            )}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if show_era_selector:
        selected_cache_key = render_cache_selector(available_caches, selected_cache_key)
    st.markdown(
        f"""
            <div class="hero-shell">
            <div class="hero-label">Fitness Research Presentation</div>
            <div class="hero-title">{html.escape(APP_TITLE)}</div>
            <div class="hero-subtitle">
                {html.escape(snapshot["meta"].get("cache_help_text", snapshot["stats"]["corpus_summary"]))}
            </div>
            <div class="hero-meta-row">
                <span class="hero-chip">{html.escape(cache_label)}</span>
                <span class="hero-chip">{html.escape(window_label)}</span>
                <span class="hero-chip">Source: {html.escape(source)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return selected_cache_key


def render_home() -> None:
    image_path = get_hero_image_path()
    poster_html = ""
    if image_path is not None:
        try:
            poster_html = (
                f'<img class="home-poster" alt="Yeah Buddy fitness poster" '
                f'src="{get_image_data_uri(image_path)}">'
            )
        except (PermissionError, OSError):
            poster_html = '<div class="home-poster"></div>'
    else:
        poster_html = '<div class="home-poster"></div>'

    st.markdown('<div class="home-title-card">Fitness</div>', unsafe_allow_html=True)

    has_poster = image_path is not None and '<img' in poster_html

    if has_poster:
        left, spacer, right = st.columns([613, 188, 400], gap="small")
    else:
        left = st.container()
        spacer = None
        right = None

    with left:
        st.markdown('<div class="home-nav-stack">', unsafe_allow_html=True)
        st.markdown('<div class="home-nav-label">Part 1 / Corpus Map</div>', unsafe_allow_html=True)
        for row in (
            (("Overview", "Overview"), ("Topics", "Topics")),
            (("Topic Detail", "Topic Detail"), ("Methods", "Methods")),
        ):
            row_cols = st.columns(2, gap="medium")
            for column, (page, label) in zip(row_cols, row):
                with column:
                    if st.button(label, key=f"home_{page}", use_container_width=True):
                        st.session_state.page = page
                        st.query_params["page"] = page
                        st.rerun()
        st.markdown('<div class="home-nav-label">Part 2 / Model + Evaluation</div>', unsafe_allow_html=True)
        for row in (
            (("RAG Chat", "RAG Chat"), ("Groq vs Gemini", "Groq vs Gemini"), ("Hindi", "Hindi")),
            (("Bias", "Bias"), ("Ethics", "Ethics")),
        ):
            row_cols = st.columns(len(row), gap="medium")
            for column, (page, label) in zip(row_cols, row):
                with column:
                    if st.button(label, key=f"home_{page}", use_container_width=True):
                        st.session_state.page = page
                        st.query_params["page"] = page
                        st.rerun()
        st.markdown(
            '<div class="home-part-note">Move from the corpus map into grounded QA, provider comparison, Hindi translation, bias probes, and ethics notes.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if has_poster:
        with spacer:
            st.empty()
        with right:
            st.markdown(poster_html, unsafe_allow_html=True)


def render_overview(snapshot: dict) -> None:
    if not snapshot["topics"] or not snapshot["overview"]["monthly_activity"]:
        st.info("Pipeline outputs are not available yet. Run the Part 1 stages and rebuild the app cache.")
        return

    stats = snapshot["stats"]
    overview = snapshot["overview"]
    render_section_intro(
        "Selected KPIs",
        "Core corpus counts for the scraped r/fitness window and the modeled topic space.",
    )
    metric_columns = st.columns(6)
    cards = [
        ("Posts", format_int(stats["total_posts"]), "Raw corpus posts in SQLite"),
        ("Comments", format_int(stats["total_comments"]), "Raw corpus comments in SQLite"),
        ("Authors", format_int(stats["total_authors"]), "Non-deleted authors in corpus"),
        ("Topics", str(stats["total_topics"]), "Final hybrid topic count"),
        ("Major Topics", str(stats["major_topic_count"]), "Validated for stance reading"),
        ("Hybrid Docs", format_int(stats["modeled_documents"]), "Documents used in the model space"),
    ]
    for column, card in zip(metric_columns, cards):
        with column:
            metric_card(*card)

    trend_col, insight_col = st.columns([1.55, 0.95], gap="large")
    with trend_col:
        render_section_intro(
            "Activity Across the Window",
            "Posts, comments, and modeled documents over time.",
        )
        st.plotly_chart(build_activity_chart(overview["monthly_activity"]), use_container_width=True, config=PLOTLY_CONFIG)
    with insight_col:
        render_section_intro(
            "What Stands Out",
            "Fast orientation before moving into topics.",
        )
        highlights = overview["activity_highlights"]
        panel_card("Corpus summary", stats["corpus_summary"])
        panel_card("Peak posting month", f"{highlights['peak_posts_month']['month_start']}: {format_int(highlights['peak_posts_month']['count'])} posts")
        panel_card("Peak comment month", f"{highlights['peak_comments_month']['month_start']}: {format_int(highlights['peak_comments_month']['count'])} comments")
        panel_card(
            "Peak modeled month",
            f"{highlights['peak_modeled_month']['month_start']}: {format_int(highlights['peak_modeled_month']['count'])} modeled documents",
        )

    st.markdown('<div class="page-divider"><span class="page-divider-label">Topic Landscape</span></div>', unsafe_allow_html=True)
    share_col, list_col = st.columns([1.2, 1.0], gap="large")
    with share_col:
        render_section_intro(
            "Topic Share",
            "Share of modeled documents by final hybrid topic.",
        )
        st.plotly_chart(build_topic_share_chart(snapshot["topics"]), use_container_width=True, config=PLOTLY_CONFIG)
    with list_col:
        render_section_intro(
            "Top Topics",
            "The largest topic groups with prevalence and trend context.",
        )
        for topic in overview["top_topics"]:
            st.markdown(
                f"""
                <div class="panel-card">
                    <div class="panel-title">Topic {topic['topic_id']} • {html.escape(topic['topic_label'])}</div>
                    <div class="topic-summary-line">
                        <strong>{topic['document_share_pct']:.1f}%</strong> modeled documents •
                        <strong>{topic['corpus_post_share_pct']:.2f}%</strong> scraped posts •
                        <strong>{format_int(topic['post_documents'])}</strong> post docs •
                        <strong>{html.escape(topic['trend_label'])}</strong> •
                        <strong>{'major stance topic' if topic['major_topic'] else 'topic-only'}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_topics(snapshot: dict) -> None:
    if not snapshot["topics"]:
        st.info("No topic outputs are available yet.")
        return
    topics = snapshot["topics"]
    major_count = sum(1 for t in topics if t.get("major_topic"))
    render_section_intro(
        "Topic Comparison",
        "Hybrid topic shares, post shares, trend labels, and stance availability in one table.",
    )
    summary_cols = st.columns(4)
    with summary_cols[0]:
        metric_card("Total Topics", str(len(topics)), "Final hybrid topic count")
    with summary_cols[1]:
        metric_card("Major Topics", str(major_count), "Validated for stance reading")
    with summary_cols[2]:
        metric_card("Minor Topics", str(len(topics) - major_count), "Topic-only, no stance")
    with summary_cols[3]:
        top_topic = max(topics, key=lambda t: t["document_share_pct"])
        metric_card("Largest Topic", f'{top_topic["document_share_pct"]:.1f}%', html.escape(top_topic["topic_label"]))
    render_topic_table(snapshot["topic_table"])
    st.markdown('<div class="page-divider"><span class="page-divider-label">Topic Share Distribution</span></div>', unsafe_allow_html=True)
    st.plotly_chart(build_topic_share_chart(snapshot["topics"]), use_container_width=True, config=PLOTLY_CONFIG)


def render_topic_detail(snapshot: dict) -> None:
    topics = snapshot["topics"]
    if not topics:
        st.info("No topic data is available yet.")
        return

    render_section_intro(
        "Topic Detail",
        "Choose a topic to inspect scale, trend, source mix, and stance evidence.",
    )
    options = {f"Topic {topic['topic_id']} • {topic['topic_label']}": topic for topic in topics}
    selected_key = st.selectbox("Select a topic", list(options.keys()))
    topic = options[selected_key]
    stance = topic["stance"]

    top_line = st.columns(5)
    metrics = [
        ("Document share", f"{topic['document_share_pct']:.1f}%", "Share of hybrid documents"),
        ("Corpus post share", f"{topic['corpus_post_share_pct']:.2f}%", "Share of all scraped posts"),
        ("Modeled post share", f"{topic['modeled_post_share_pct']:.1f}%", "Share of model-ready posts"),
        ("Posts", format_int(topic["post_documents"]), "Post documents inside topic"),
        ("Comments", format_int(topic["comment_documents"]), "Comment documents inside topic"),
    ]
    for column, item in zip(top_line, metrics):
        with column:
            metric_card(*item)

    st.markdown('<div class="topic-detail-section">', unsafe_allow_html=True)
    pills = " ".join(f'<span class="pill">{html.escape(term)}</span>' for term in topic["keyword_terms"][:8])
    status_label = "Major — stance analysis available" if topic["major_topic"] else "Minor — topic-only"
    st.markdown(
        f'{pills} <span class="topic-count-badge">{html.escape(status_label)}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    trend_col, semantics_col = st.columns([0.8, 1.2], gap="large")
    with trend_col:
        panel_card("Trend label", f"{topic['trend_label'].title()} • Coverage {topic['month_coverage_ratio']:.1f}% of months")
    with semantics_col:
        panel_card(
        "How to read the share values",
        (
            f"This topic contains {format_int(topic['document_count'])} modeled documents, including "
            f"{format_int(topic['post_documents'])} posts. Document share describes the modeling space; "
            f"post share describes prevalence among scraped posts."
        ),
        )

    st.markdown('<div class="page-divider"><span class="page-divider-label">Monthly Trend</span></div>', unsafe_allow_html=True)
    render_section_intro(
        "Monthly Trend",
        "Volume and share move together here so spikes can be read against the size of the corpus.",
    )
    st.plotly_chart(build_topic_detail_chart(topic), use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown('<div class="page-divider"><span class="page-divider-label">Source & Interpretation</span></div>', unsafe_allow_html=True)
    mix_col, semantics_col = st.columns([0.9, 1.1], gap="large")
    with mix_col:
        render_section_intro("Source Mix", "Posts and comments represented inside the selected topic.")
        st.plotly_chart(build_source_mix_chart(topic), use_container_width=True, config=PLOTLY_CONFIG)
    with semantics_col:
        render_section_intro("Interpretation", "Assignment confidence and monthly concentration help separate broad topics from sharper bursts.")
        panel_card(
            "Topic semantics",
            f"This topic is labeled as {topic['trend_label']}. Average assignment confidence is {topic['average_assignment_confidence']:.3f}, and the peak-to-median monthly ratio is {topic['peak_to_median_ratio']:.2f}.",
        )

    st.markdown('<div class="page-divider"><span class="page-divider-label">Corpus Examples</span></div>', unsafe_allow_html=True)
    render_section_intro("Representative Documents", "Short corpus examples for reading the topic in context.")
    for doc in topic["representative_documents"]:
        doc_card(doc)

    st.markdown('<div class="page-divider"><span class="page-divider-label">Stance Analysis</span></div>', unsafe_allow_html=True)
    render_section_intro(
        "Stance View",
        "Cluster summaries and representative comments for the topics where stance analysis is available.",
    )
    if not stance["available"]:
        st.markdown(
            f"""
            <div class="stance-card">
                <div class="panel-title">No stance analysis for Topic {topic['topic_id']}</div>
                <div class="subtle-text">{html.escape(stance['detail_note'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    title = (
        "Cautious support/opposition summaries"
        if stance["mode"] == "cautious_stance_split"
        else "Support/opposition summaries"
    )
    st.markdown(
        f"""
        <div class="stance-card">
            <div class="panel-title">{title}</div>
            <div class="subtle-text">{html.escape(stance['detail_note'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    outcome = stance["metrics"]
    summary_columns = st.columns(4)
    summary_cards = [
        ("Analyzed comments", format_int(outcome["analyzed_comments"]), "Comments clustered for stance"),
        ("Minority share", f"{outcome['minority_share'] * 100:.1f}%", "Smaller side of the split"),
        ("Silhouette", f"{outcome['silhouette_score']:.4f}", "Higher is cleaner separation"),
        ("Centroid similarity", f"{outcome['centroid_similarity']:.4f}", "Higher means more overlap"),
    ]
    for column, card in zip(summary_columns, summary_cards):
        with column:
            metric_card(*card)

    for summary in stance["summaries"]:
        user_count = summary.get("user_count", 0)
        comment_count = summary.get("comment_count", 0)
        label = summary["stance_label"].replace("_", " ").title()
        panel_card(
            label,
            (
                f"{summary['summary_text']} "
                f"Grouped users: {format_int(user_count)} authors across {format_int(comment_count)} comments."
            ),
        )
        for comment in summary["representative_comments"]:
            doc_card(comment, label="Representative comment")


METHODS_SUBPAGES = [
    ("Data Acquisition", "p1_acquisition"), ("Feature Prep", "p1_features"),
    ("Topic Modeling", "p1_topics"), ("Trend Rules", "p1_trends"), ("Stance Analysis", "p1_stance"),
    ("RAG Architecture", "p2_rag"), ("Chunking", "p2_chunks"), ("Embeddings", "p2_embeddings"),
    ("Retrieval", "p2_retrieval"), ("Query Routing", "p2_routing"), ("Generation", "p2_generation"),
    ("RAG Evaluation", "p2_eval"), ("Hindi Task", "p2_hindi"),
]


def _method_block(title: str, items: list) -> None:
    items_html = "".join(f"<li>{html.escape(str(i))}</li>" for i in items)
    st.markdown(f'<div class="method-row"><div></div><div><div class="method-title">{html.escape(title)}</div><ul class="method-list">{items_html}</ul></div></div>', unsafe_allow_html=True)


def _method_table(title: str, rows: list) -> None:
    rows_html = "".join(f'<tr><td class="risk-name">{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>' for k, v in rows)
    st.markdown(f'<div class="section-shell"><div class="method-title" style="margin-bottom:0.5rem">{html.escape(title)}</div><table class="risk-matrix"><thead><tr><th>Item</th><th>Detail</th></tr></thead><tbody>{rows_html}</tbody></table></div>', unsafe_allow_html=True)


def _msub_acquisition() -> None:
    render_section_intro("Data Acquisition", "How r/fitness posts and comments were collected and stored.")
    _method_block("Source", ["Arctic Shift archive is used instead of PRAW because the project cannot rely on Reddit's official API access path.", "Posts are fetched in ascending date order from an aggregate-identified valid r/fitness date window.", "Both raw JSON payloads and normalized fields are stored in SQLite."])
    _method_block("Ingestion", ["Paginated fetches with checkpointing, backoff/retry, and idempotent upserts keyed by Reddit IDs.", "Comment ingestion preserves link_id and parent_id for thread structure.", "Deleted/removed content is stored but excluded from NLP modeling."])
    _method_block("Corpus Windows", ["Default-safe window: 2023-04-01 to 2024-05-01 — 19,320 posts, 288,453 comments.", "Advisory 2018 window (Jun–Aug 2018) stored but not trusted for RAG.", "Full database: 40,520 posts · 434,530 comments · 80,038 authors · 475,050 documents."])
    _method_table("SQLite Schema", [("posts", "Post fields, raw JSON, raw/clean text, scores, URLs, deletion flags"), ("comments", "Comment text, parent/post links, scores, depth, raw JSON"), ("documents", "Unified post/comment table with stable document_id"), ("documents_fts", "FTS5 index for lexical retrieval"), ("authors", "Usernames, pseudonyms, bot/deletion flags"), ("topic_definitions", "Labels, keywords, share, trend label, representatives"), ("comment_stances", "Stance labels for topic-specific comments"), ("pipeline_checkpoints", "Stage status and payloads")])


def _msub_features() -> None:
    render_section_intro("Feature Preparation", "Filtering and cleaning rules for the topic model corpus.")
    _method_block("Post Filters", ["Posts must not be deleted or removed.", "Posts must have ≥20 cleaned characters.", "Result: 861 model-ready posts from 19,320."])
    _method_block("Comment Filters", ["Must not be deleted/removed or by a probable bot (29 bots identified).", "Must have ≥40 cleaned characters and ≥8 tokens.", "Result: 219,087 model-ready from 288,453."])
    _method_block("Text Cleaning", ["raw_text and clean_text kept side by side.", "Cleaning removes markdown noise, URLs, and boilerplate.", "AutoModerator excluded from NLP only — database counts are unaffected."])


def _msub_topics(snapshot: dict) -> None:
    render_section_intro("Topic Modeling", "How 8 topics were discovered from a hybrid post+comment corpus.")
    _method_block("Approach", ["TF-IDF features → TruncatedSVD + normalization → BERTopic-style model with KMeans.", "Target: 8 human-facing topics, within the required 5–20 range.", "Assignment source: bertopic_hybrid_top_level_comments_v1."])
    _method_block("Hybrid Corpus", ["r/fitness is megathread-driven — comments carry most semantic signal.", "861 model-ready posts + 3,220 bounded top-level comments = 4,081 modeled documents.", "Comments ranked by score within each post, top 3 per post, then global cap. Posts never squeezed out."])
    _method_block("Label Review", ["Each topic's top 10 keywords reviewed; autogenerated labels renamed only if misleading.", "Single light manual pass — not full manual annotation."])


def _msub_trends() -> None:
    render_section_intro("Trend Classification", "Deterministic rules for trending, persistent, or mixed labels.")
    _method_block("Rules", ["Persistent: topic in ≥75% of months AND peak-to-median ≤ 1.75.", "Trending: peak-to-median ≥ 2.5 (concentrated spike).", "Mixed: neither condition strong enough."])
    _method_block("Implementation", ["Monthly topic share computed from the hybrid document corpus.", "Coverage ratio and peak-to-median stored per topic.", "Rules are objective, explainable, and match the modeling unit."])


def _msub_stance(snapshot: dict) -> None:
    render_section_intro("Stance Analysis", "Unsupervised clustering to surface agreement/disagreement patterns.")
    _method_block("Scope", ["Runs only for major topics (post share above a fixed minimum).", "Currently validated for topics 1 (Weight/Fat/Calories), 2 (Gym/Rant), and 5 (Bench/Press/Sets)."])
    _method_block("Method", ["Topic-specific comments clustered into 2 camps via TF-IDF + KMeans.", "Larger upvote-weighted cluster = provisional dominant position.", "Comments classified by nearest centroid prototype."])
    _method_block("Guardrails", ["Two-sided summaries shown only when minority cluster exceeds minimum support.", "Weak/overlapping splits use cautious wording — all 3 current outputs are 'Cautious support/opposition'.", "Known limitation: can over-separate near-consensus topics or miss subtle rhetorical differences."])


def _msub_rag_arch() -> None:
    render_section_intro("RAG System Architecture", "End-to-end retrieval-augmented generation over the frozen r/fitness corpus.")
    st.markdown("""
    <div class="methods-diagram rag-arch-diagram">
        <div class="rag-arch-title">RAG Pipeline Architecture</div>
        <div class="rag-arch-grid">
            <div class="rag-arch-col">
                <div class="rag-arch-phase">Offline Index Build</div>
                <div class="md-node md-src" style="width:100%">Part 1 SQLite<br><span>475,050 documents</span></div>
                <div class="md-arrow-v">↓</div>
                <div class="md-node" style="width:100%">Corpus Freeze<br><span>307,773 docs (2023–2024)</span></div>
                <div class="md-arrow-v">↓</div>
                <div class="md-node" style="width:100%">Chunk Generation<br><span>313,615 chunks</span></div>
                <div class="md-arrow-v">↓</div>
                <div class="md-node" style="width:100%">MiniLM-L6-v2 Embedding<br><span>384-dim vectors</span></div>
                <div class="md-arrow-v">↓</div>
                <div class="md-node md-out" style="width:100%">FAISS Dense Index</div>
            </div>
            <div class="rag-arch-divider"></div>
            <div class="rag-arch-col">
                <div class="rag-arch-phase">Online Query Path</div>
                <div class="md-node md-src" style="width:100%">User Query</div>
                <div class="md-arrow-v">↓</div>
                <div class="md-node" style="width:100%">Normalize + Expand Abbreviations</div>
                <div class="md-arrow-v">↓</div>
                <div class="md-node" style="width:100%">Classify Query<br><span>factual / opinion / adversarial</span></div>
                <div class="md-arrow-v">↓</div>
                <div class="rag-arch-split">
                    <div class="md-node">Dense<br><span>FAISS</span></div>
                    <div class="rag-arch-plus">+</div>
                    <div class="md-node">Lexical<br><span>FTS5</span></div>
                </div>
                <div class="md-arrow-v">↓</div>
                <div class="md-node" style="width:100%">RRF Merge + Dedup<br><span>k=60</span></div>
                <div class="md-arrow-v">↓</div>
                <div class="md-node" style="width:100%">Grounded Prompt<br><span>Source labels S1…Sn</span></div>
                <div class="md-arrow-v">↓</div>
                <div class="rag-arch-split">
                    <div class="md-node md-out">Groq<br><span>llama-3.3-70b</span></div>
                    <div class="rag-arch-or">or</div>
                    <div class="md-node md-out">Gemini<br><span>gemini-2.5-flash</span></div>
                </div>
                <div class="md-arrow-v">↓</div>
                <div class="md-node md-src" style="width:100%">JSON Answer + Citations</div>
            </div>
        </div>
        <div class="rag-arch-note">Adversarial queries bypass the LLM via a deterministic abstention path.</div>
    </div>
    """, unsafe_allow_html=True)
    _method_block("Design Decisions", ["Part 2 treats Part 1 as read-only input.", "Corpus frozen before retrieval/evaluation work begins.", "Retrieval and generation are separate: retrieval finds evidence, generation cites it.", "Two LLM endpoints share the same retrieval pipeline for fair comparison."])


def _msub_chunks() -> None:
    render_section_intro("Chunking Pipeline", "How 307,773 documents become 313,615 retrieval-ready chunks.")
    _method_block("Rules", ["Comments stay as single chunks unless exceeding max size (220 tokens).", "Posts split by paragraph first, then by sentence.", "Adjacent post chunks use 1-sentence overlap (~20–30 tokens).", "URL-only documents produce deterministic fallback chunks (874 total)."])
    _method_table("Coverage Summary", [("Source documents", "307,773"), ("Documents represented", "307,773 (100%)"), ("Total chunks", "313,615"), ("Empty chunks", "0"), ("Fallback chunks", "874")])
    _method_block("Chunk Schema", ["Fields: chunk_id, document_id, source_type, source_id, post_id, parent_id, link_id, created_utc, author_id, title, chunk_text, chunk_index, token_estimate.", "Metadata preserved for exact citation of original post/comment.", "Source contract follows the frozen corpus manifest directly."])


def _msub_embeddings() -> None:
    render_section_intro("Embeddings & FAISS", "Vectorizing chunks and building the dense retrieval index.")
    _method_table("Embedding Config", [("Model", "sentence-transformers/all-MiniLM-L6-v2"), ("Dimension", "384"), ("Batch size", "128"), ("Total embeddings", "313,615"), ("Index type", "FAISS dense"), ("Store", "Resumable SQLite artifact")])
    _method_block("Preflight Checks", ["Hashes frozen chunk JSONL and manifest.", "Checks manifest count matches JSONL line count.", "Rejects duplicate chunk_ids.", "Verifies authoritative expected count (313,615) before any work."])
    _method_block("Resumability", ["Embeddings stored incrementally in SQLite.", "Resume only if chunk hash, model, dimension, and batch size all match.", "FAISS index built after all embeddings complete."])


def _msub_retrieval() -> None:
    render_section_intro("Retrieval Layer", "Dense, lexical, and hybrid retrieval with Reciprocal Rank Fusion.")
    st.markdown(
        '<div class="methods-diagram">'
        '<div class="rag-arch-title">Hybrid Retrieval Flow</div>'
        '<div style="display: flex; gap: 2rem; justify-content: center;">'
        '<div class="rag-arch-col" style="flex: 1;">'
        '<div class="md-node md-src" style="width: 100%;">Query<br><span>embedded</span></div>'
        '<div class="md-arrow-v">&#x2193;</div>'
        '<div class="md-node" style="width: 100%;">Dense / FAISS<br><span>top-k nearest</span></div>'
        '</div>'
        '<div class="rag-arch-col" style="flex: 1;">'
        '<div class="md-node md-src" style="width: 100%;">Query<br><span>text</span></div>'
        '<div class="md-arrow-v">&#x2193;</div>'
        '<div class="md-node" style="width: 100%;">Lexical / FTS5<br><span>BM25-style</span></div>'
        '</div>'
        '</div>'
        '<div class="rag-arch-split" style="margin: 0.8rem 0 0.5rem 0;">'
        '<div class="rag-arch-plus">+</div>'
        '</div>'
        '<div class="md-flow" style="justify-content:center">'
        '<div class="md-node md-out" style="width: 100%; max-width: 25rem;">RRF Merge &#x2192; Dedup by document_id &#x2192; Final top-k</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    _method_table("Retrieval Parameters", [("Dense top-k", "8"), ("Lexical top-k", "8"), ("Hybrid final top-k", "5"), ("RRF constant", "60"), ("Factual final top-k", "7"), ("Opinion dense/lexical", "14 each"), ("Opinion final top-k", "7"), ("Adversarial dense/lexical", "4 each"), ("Adversarial final top-k", "3")])
    _method_block("Lexical Mapping", ["Lexical retrieval is document-level via Part 1 FTS5.", "Multi-chunk documents map to first chunk by ascending chunk_index.", "Hybrid keeps best dense chunk when available, falls back to lexical default."])


def _msub_routing() -> None:
    render_section_intro("Query Classification & Routing", "Deterministic rule-based routing with abbreviation expansion.")
    _method_block("Query Classes", ["factual — balanced hybrid, moderate top-k.", "opinion-summary — larger candidate pools and top-k.", "adversarial/no-answer — smaller pool, guardrail fields for abstention."])
    _method_table("Abbreviation Map", [("ppl", "push pull legs"), ("5x5", "five by five strength program"), ("1rm", "one rep max"), ("pr", "personal record"), ("bw", "bodyweight"), ("rpe", "rate of perceived exertion"), ("tdee", "total daily energy expenditure"), ("bmr", "basal metabolic rate")])
    _method_block("Logic", ["Router normalizes query → expands abbreviations → classifies → selects retrieval defaults → calls retrieval.", "Concept terms appended for maintenance calories, warmups, recomp when rules fire.", "Classification is deterministic and rule-based, not model-based."])


def _msub_generation() -> None:
    render_section_intro("Answer Generation", "Grounded prompting with source labels and structured JSON output.")
    _method_block("Providers", ["Groq: llama-3.3-70b-versatile", "Gemini: gemini-2.5-flash", "Both return the same structured output shape."])
    _method_block("Prompt Contract", ["Query-type-aware prompt inserts retrieved snippets with labels S1, S2, etc.", "Model must cite only those source labels.", "Model must state when evidence is missing.", "Adversarial queries bypass the provider via deterministic abstention."])
    _method_block("Answer Contract", ['answer_text: grounded answer text', 'citations: ["S1", "S3", ...]', 'insufficient_evidence: true/false'])
    _method_block("Validation", ["Substantive answers must have citations.", "Citations must refer only to retrieved source labels.", "Raw provider responses persisted for debugging."])


def _msub_eval() -> None:
    render_section_intro("RAG Evaluation", "Frozen 15-question eval set with automatic and manual metrics.")
    _method_block("Evaluation Set", ["15 question-answer pairs based on the actual r/fitness corpus.", "8 factual · 5 opinion-summary · 2 adversarial/no-answer.", "Frozen before meaningful prompt tuning."])
    _method_block("Automatic Metrics", ["ROUGE-L F1", "BERTScore (distilbert-base-uncased)", "Retrieval hit rate (Hit@k)"])
    _method_block("Manual Review", ["Every output reviewed for faithfulness.", "Adversarial answers checked for correct abstention.", "Citation validity verified against retrieved sources.", "Error taxonomy: retrieval miss, citation miss, hallucination, incomplete summary, overgeneralization."])


def _msub_hindi() -> None:
    render_section_intro("Hindi Translation Task", "English/code-mixed r/fitness text → natural Hindi translation.")
    _method_block("Dataset", ["20 examples: 4 RAG eval questions, 9 gold-answer summaries, 7 edge-case probes.", "Difficult tags: code-mixed, Reddit slang, fitness slang, named entities, abbreviations.", "Includes PPL, TDEE, BMR, PR, 5x5, StrongLifts, OP, natty, cope, bulk, cut, recomp."])
    _method_block("Automatic Metrics", ["chrF (character n-gram F-score)", "Multilingual BERTScore (bert-base-multilingual-cased)"])
    _method_block("Manual Review", ["8 difficult examples per provider manually scored for fluency and adequacy.", "Edge-case analysis on code-mixed and slang-heavy inputs."])


def render_methods(snapshot: dict) -> None:
    render_section_intro("Methods", "Every major pipeline step from Part 1 and Part 2 is documented below. Select a task.")

    if "methods_sub" not in st.session_state:
        st.session_state.methods_sub = None

    st.markdown('<div class="eyebrow">Part 1 / Corpus Pipeline</div>', unsafe_allow_html=True)
    p1_cols = st.columns(5, gap="medium")
    for col, (label, key) in zip(p1_cols, METHODS_SUBPAGES[:5]):
        with col:
            btype = "primary" if st.session_state.methods_sub == key else "secondary"
            if st.button(label, key=f"msub_{key}", use_container_width=True, type=btype):
                st.session_state.methods_sub = key
                st.rerun()

    st.markdown('<div class="eyebrow" style="margin-top:0.6rem">Part 2 / RAG + Evaluation</div>', unsafe_allow_html=True)
    p2a = st.columns(4, gap="medium")
    for col, (label, key) in zip(p2a, METHODS_SUBPAGES[5:9]):
        with col:
            btype = "primary" if st.session_state.methods_sub == key else "secondary"
            if st.button(label, key=f"msub_{key}", use_container_width=True, type=btype):
                st.session_state.methods_sub = key
                st.rerun()
    p2b = st.columns(4, gap="medium")
    for col, (label, key) in zip(p2b, METHODS_SUBPAGES[9:13]):
        with col:
            btype = "primary" if st.session_state.methods_sub == key else "secondary"
            if st.button(label, key=f"msub_{key}", use_container_width=True, type=btype):
                st.session_state.methods_sub = key
                st.rerun()

    st.markdown('<div style="border-top:2px solid var(--line-strong);margin:1.2rem 0 0.8rem 0"></div>', unsafe_allow_html=True)

    sub = st.session_state.methods_sub
    dispatch = {
        "p1_acquisition": _msub_acquisition, "p1_features": _msub_features,
        "p1_topics": lambda: _msub_topics(snapshot), "p1_trends": _msub_trends,
        "p1_stance": lambda: _msub_stance(snapshot), "p2_rag": _msub_rag_arch,
        "p2_chunks": _msub_chunks, "p2_embeddings": _msub_embeddings,
        "p2_retrieval": _msub_retrieval, "p2_routing": _msub_routing,
        "p2_generation": _msub_generation, "p2_eval": _msub_eval, "p2_hindi": _msub_hindi,
    }
    if sub and sub in dispatch:
        dispatch[sub]()
    else:
        # Landing page with pipeline diagrams
        st.markdown('<div class="methods-landing-text">Select any task above to view its methodology, design decisions, and implementation details.</div>', unsafe_allow_html=True)
        render_section_intro("Part 1 Pipeline", "From raw Reddit data to interactive analysis.")
        st.markdown("""
        <div class="methods-diagram">
            <div class="md-flow">
                <div class="md-node md-src">Arctic Shift<br><span>Post/comment archive</span></div>
                <div class="md-arrow">→</div>
                <div class="md-node">SQLite<br><span>Raw + clean text</span></div>
                <div class="md-arrow">→</div>
                <div class="md-node">Feature Prep<br><span>Model flags</span></div>
                <div class="md-arrow">→</div>
                <div class="md-node">Topic Model<br><span>BERTopic / KMeans</span></div>
                <div class="md-arrow">→</div>
                <div class="md-node">Stance<br><span>Cluster &amp; label</span></div>
            </div>
            <div class="md-flow" style="margin-top:0.5rem;justify-content:center"><div class="md-arrow" style="transform:rotate(90deg)">↓</div></div>
            <div class="md-flow" style="justify-content:center"><div class="md-node md-out">App Cache → Streamlit</div></div>
        </div>
        """, unsafe_allow_html=True)
        render_section_intro("Part 2 Pipeline", "From frozen corpus to grounded answers.")
        st.markdown("""
        <div class="methods-diagram">
            <div class="md-flow">
                <div class="md-node md-src">Frozen Corpus<br><span>307,773 docs</span></div>
                <div class="md-arrow">→</div>
                <div class="md-node">Chunking<br><span>313,615 chunks</span></div>
                <div class="md-arrow">→</div>
                <div class="md-node">Embeddings<br><span>MiniLM-L6-v2</span></div>
                <div class="md-arrow">→</div>
                <div class="md-node">FAISS<br><span>384-dim dense</span></div>
            </div>
            <div class="md-flow" style="margin-top:0.5rem;justify-content:center"><div class="md-arrow" style="transform:rotate(90deg)">↓</div></div>
            <div class="md-flow">
                <div class="md-node md-src">User Query</div>
                <div class="md-arrow">→</div>
                <div class="md-node">Classify &amp; Route</div>
                <div class="md-arrow">→</div>
                <div class="md-node">Hybrid Retrieval<br><span>Dense + FTS + RRF</span></div>
                <div class="md-arrow">→</div>
                <div class="md-node">Grounded Prompt</div>
                <div class="md-arrow">→</div>
                <div class="md-node md-out">Groq / Gemini</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        methods = snapshot.get("methods", {})
        notes = methods.get("notes", [])
        if notes:
            render_section_intro("Interpretation Notes", "Caveats that keep the corpus claims appropriately narrow.")
            notes_html = "".join(f'<div class="note-card"><div class="subtle-text">{html.escape(note)}</div></div>' for note in notes)
            st.markdown(f'<div class="note-grid">{notes_html}</div>', unsafe_allow_html=True)


def render_rag_chat() -> None:
    render_section_intro(
        "Ask The Corpus",
        "Query the frozen r/fitness chunks, then read the answer, citations, and evidence together.",
    )
    try:
        adapter = get_part2_adapter()
    except Exception as exc:
        show_rag_unavailable(exc)
        return

    st.markdown('<div class="rag-control-panel"><div class="rag-board-title">Corpus Question</div>', unsafe_allow_html=True)
    with st.form("rag_chat_form"):
        query = st.text_area(
            "Question",
            value="What do people think about body recomposition?",
            height=118,
            help="Ask about training, nutrition, routines, or community advice represented in the frozen corpus.",
        )
        control_cols = st.columns([0.95, 0.78, 0.78, 0.78], gap="medium")
        with control_cols[0]:
            provider_label = st.selectbox("Provider", ("Groq", "Gemini"))
        with control_cols[1]:
            evidence_only = st.toggle(
                "Evidence only",
                value=False,
                help="Skip Groq/Gemini and show only retrieved source strips.",
            )
        with control_cols[2]:
            save_raw_response = st.toggle("Save raw response", value=False)
        with control_cols[3]:
            show_prompt = st.toggle("Show prompt", value=False)
        submitted = st.form_submit_button("Run RAG")
    st.markdown("</div>", unsafe_allow_html=True)

    if not submitted:
        with st.expander("Provider readiness", expanded=False):
            report_rows(get_provider_debug_summary(provider_label.lower(), adapter))
        return

    if not query.strip():
        st.warning("Add a question before running the RAG bench.")
        return

    provider_name = provider_label.lower()
    if not evidence_only:
        provider_ok, provider_message = get_provider_status(provider_name, adapter)
        if not provider_ok:
            st.error(
                f"{provider_label} generation is not available in this Streamlit environment: {provider_message}"
            )
            st.info(
                "Turn on Evidence only to inspect retrieval without calling the provider, or restart Streamlit with the matching API key exported."
            )
            return

    try:
        with st.spinner("Routing query and pulling source strips..."):
            result, routed = adapter.run_rag_query(
                query.strip(),
                provider_name=provider_name,
                retrieval_only=evidence_only,
                save_raw_response=save_raw_response,
            )
    except Exception as exc:
        if evidence_only:
            st.error(
                "Evidence-only retrieval could not complete in this environment. Check the Part 2 retrieval dependencies such as faiss-cpu and sentence-transformers."
            )
            st.code(str(exc), language="text")
        else:
            st.error(
                f"{provider_label} generation could not complete after retrieval. Check provider availability, model configuration, and the exported API key for this Streamlit process."
            )
            with st.expander("Provider debug details", expanded=False):
                st.code(str(exc), language="text")
        return

    st.session_state.last_rag_mode = "evidence-only" if evidence_only else "generation"
    st.session_state.last_rag_provider = result.provider
    st.session_state.last_rag_retrieval = routed.retrieval_mode_used
    st.session_state.last_rag_evidence = "insufficient" if result.insufficient_evidence else "grounded"

    render_section_intro("Answer" if not evidence_only else "Evidence-Only Result", "The response and evidence are shown in the same order used for presentation.")
    render_context_status_line(
        snippet_count=len(result.retrieved_snippets),
        provider=provider_name,
        evidence_only=evidence_only,
    )
    if evidence_only:
        st.info("Evidence-only mode skipped Groq and Gemini; retrieval and routing completed.")
    else:
        st.markdown(
            f"""
            <div class="rag-answer-box">
                <div class="rag-board-title">Final Answer</div>
                <div class="rag-answer-text">{html.escape(result.answer_text)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="eyebrow">Citations</div>', unsafe_allow_html=True)
        render_citation_tags(result.citations)
        if result.raw_response_path:
            st.caption(f"Raw response saved to {result.raw_response_path}")

    st.markdown(
        f"""
        <div class="detail-line">
            <strong>Run details:</strong>
            query type {html.escape(routed.classification.query_type)}
            · provider {html.escape(result.provider)}/{html.escape(result.model)}
            · retrieval {html.escape(routed.retrieval_mode_used)}
            · insufficient evidence {html.escape(str(result.insufficient_evidence).lower())}
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_intro("Evidence", "Retrieved source strips from the frozen chunk index.")
    render_source_strips(result.retrieved_snippets)

    with st.expander("Routing and provider details", expanded=False):
        report_rows(
            [
                ("Normalized query", routed.classification.normalized_query),
                ("Provider config", get_provider_status(provider_name, adapter)[1]),
                ("Save raw response", str(save_raw_response).lower()),
            ]
        )

    if show_prompt:
        with st.expander("Debug prompt", expanded=False):
            try:
                st.code(adapter.build_prompt_debug_text(query.strip(), routed), language="text")
            except Exception as exc:
                st.code(str(exc), language="text")


def render_rag_report() -> None:
    render_section_intro(
        "Groq Vs Gemini",
        "Provider comparison for the frozen 15-question RAG evaluation set.",
    )
    try:
        adapter = get_part2_adapter()
        artifacts = adapter.get_report_artifacts()
    except Exception as exc:
        show_rag_unavailable(exc)
        return

    groq_eval = load_json_artifact(artifacts["groq_eval_summary_json"])
    groq_manual = load_json_artifact(artifacts["groq_manual_review_summary_json"])
    gemini_eval = load_json_artifact(artifacts["gemini_eval_summary_json"])
    gemini_manual = load_json_artifact(artifacts["gemini_manual_review_summary_json"])

    top_metrics = st.columns(5)
    cards = [
        ("Eval set", "15", "8 factual, 5 opinion, 2 adversarial"),
        (
            "Retrieval hit@k",
            format_metric(gemini_eval.get("retrieval_hit_at_k_answer_bearing")),
            "Same deterministic retriever for both providers",
        ),
        (
            "Adversarial",
            f"{gemini_eval.get('adversarial_correct_abstention_count', 0)}/{gemini_eval.get('adversarial_row_count', 0)}",
            "Correct no-answer abstentions",
        ),
        (
            "Groq faithful",
            format_metric(groq_manual.get("faithfulness_percentage"), percent=True),
            "Completed hardened Groq manual review",
        ),
        (
            "Gemini faithful",
            format_metric(gemini_manual.get("faithfulness_percentage"), percent=True),
            "Completed Gemini manual review",
        ),
    ]
    for column, card in zip(top_metrics, cards):
        with column:
            metric_card(*card)

    headers = (
        "Provider",
        "ROUGE-L F1",
        "BERTScore F1",
        "Faithfulness",
        "Citation validity",
        "Abstention",
    )
    comparison_rows = [
        {
            "Provider": "Groq",
            "ROUGE-L F1": format_metric(groq_eval.get("average_rouge_l_f1_answer_bearing_success")),
            "BERTScore F1": format_metric(groq_eval.get("average_bert_score_f1_answer_bearing_success")),
            "Faithfulness": f"{groq_manual.get('faithfulness_yes_count', 10)}/13 = {format_metric(groq_manual.get('faithfulness_percentage'), percent=True)}",
            "Citation validity": f"{groq_manual.get('citation_valid_yes_count', 12)}/13 = {format_metric(groq_manual.get('citation_validity_percentage'), percent=True)}",
            "Abstention": f"{groq_eval.get('adversarial_correct_abstention_count', 2)}/{groq_eval.get('adversarial_row_count', 2)}",
        },
        {
            "Provider": "Gemini",
            "ROUGE-L F1": format_metric(gemini_eval.get("average_rouge_l_f1_answer_bearing_success")),
            "BERTScore F1": format_metric(gemini_eval.get("average_bert_score_f1_answer_bearing_success")),
            "Faithfulness": f"{gemini_manual.get('faithfulness_yes_count', 11)}/13 = {format_metric(gemini_manual.get('faithfulness_percentage'), percent=True)}",
            "Citation validity": f"{gemini_manual.get('citation_valid_yes_count', 13)}/13 = {format_metric(gemini_manual.get('citation_validity_percentage'), percent=True)}",
            "Abstention": f"{gemini_eval.get('adversarial_correct_abstention_count', 2)}/{gemini_eval.get('adversarial_row_count', 2)}",
        },
    ]
    table_html = (
        '<div class="provider-table-shell"><table class="provider-table">'
        f"<thead><tr>{''.join(f'<th>{html.escape(header)}</th>' for header in headers)}</tr></thead>"
        "<tbody>"
        + "".join(
            "<tr>"
            + "".join(
                f'<td class="{"num" if header in {"Run", "ROUGE-L F1", "BERTScore F1", "Manual faithfulness"} else ""}">'
                f"{html.escape(str(row[header]))}</td>"
                for header in headers
            )
            + "</tr>"
            for row in comparison_rows
        )
        + "</tbody></table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)

    # --- Per-question BERTScore chart ---
    try:
        groq_run_path = artifacts.get("groq_eval_results_csv") or artifacts.get("groq_eval_results_jsonl")
        gemini_run_path = artifacts.get("gemini_eval_results_csv") or artifacts.get("gemini_eval_results_jsonl")
        per_q_frames = []
        for run_path, prov_name in [(groq_run_path, "groq"), (gemini_run_path, "gemini")]:
            if run_path and run_path.exists() and str(run_path).endswith(".csv"):
                df = pd.read_csv(run_path)
                df = df[df["status"] == "success"].copy()
                df["provider_label"] = prov_name.title()
                per_q_frames.append(df)
        if per_q_frames:
            per_q = pd.concat(per_q_frames, ignore_index=True)
            per_q_chart = per_q[["question_id", "provider_label", "bert_score_f1"]].dropna(subset=["bert_score_f1"])
            if not per_q_chart.empty:
                bert_col, faith_col = st.columns([1.15, 0.85], gap="large")
                with bert_col:
                    render_section_intro("Per-Question BERTScore", "BERTScore F1 for each eval question, grouped by provider.")
                    fig = go.Figure()
                    for prov, color in [("Groq", CHART_COLORS["ink"]), ("Gemini", CHART_COLORS["accent_soft"])]:
                        sub = per_q_chart[per_q_chart["provider_label"] == prov].sort_values("question_id")
                        fig.add_trace(go.Bar(
                            x=sub["question_id"], y=sub["bert_score_f1"], name=prov,
                            marker_color=color, marker_line={"color": "#c8baa9", "width": 0.6},
                        ))
                    fig.update_layout(barmode="group", xaxis_title="Question", yaxis_title="BERTScore F1")
                    fig.update_xaxes(tickangle=-45, tickfont={"size": 9})
                    fig = apply_chart_style(fig, height=380, legend_y=1.12)
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                with faith_col:
                    render_section_intro("Faithfulness by Type", "Manual faithfulness breakdown by question category.")
                    faith_data = []
                    for prov_label, manual_data in [("Groq", groq_manual), ("Gemini", gemini_manual)]:
                        fbt = manual_data.get("faithfulness_by_question_type", {})
                        for qtype, vals in fbt.items():
                            faith_data.append({
                                "Provider": prov_label,
                                "Question Type": qtype.replace("-", " ").title(),
                                "Faithfulness %": (vals.get("faithfulness_percentage", 0)) * 100,
                            })
                    if faith_data:
                        fdf = pd.DataFrame(faith_data)
                        fig2 = go.Figure()
                        for prov, color in [("Groq", CHART_COLORS["ink"]), ("Gemini", CHART_COLORS["accent_soft"])]:
                            sub = fdf[fdf["Provider"] == prov]
                            fig2.add_trace(go.Bar(
                                x=sub["Question Type"], y=sub["Faithfulness %"], name=prov,
                                marker_color=color, marker_line={"color": "#c8baa9", "width": 0.6},
                            ))
                        fig2.update_layout(barmode="group", yaxis_title="Faithfulness %")
                        fig2 = apply_chart_style(fig2, height=360, legend_y=1.12)
                        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)
    except Exception:
        pass  # gracefully skip if eval run CSVs aren't available

    compare_col, notes_col = st.columns([1.15, 0.85], gap="large")
    with compare_col:
        render_section_intro(
            "Provider Readout",
            "Retrieval is identical for both providers; the differences come from answer generation.",
        )
        metric_pair = st.columns(2)
        with metric_pair[0]:
            metric_card("Groq BERTScore F1", format_metric(groq_eval.get("average_bert_score_f1_answer_bearing_success")), "Higher automatic similarity")
        with metric_pair[1]:
            metric_card("Gemini Faithfulness", format_metric(gemini_manual.get("faithfulness_percentage"), percent=True), "Stronger completed manual review")
        takeaway("Groq leads on automatic overlap metrics, while Gemini is stronger in completed manual review: fuller opinion summaries, complete citation validity, and the same abstention safety.")
    with notes_col:
        render_section_intro("Known Caveats", "Important limits when presenting the comparison.")
        panel_card("Retrieval scoring", "Two items are stricter retrieval-score misses even though manual review found usable grounded evidence.")
        panel_card("Groq review caveat", "The completed Groq manual review is from the hardened pre-BERTScore run, not the latest rerun.")
        panel_card("Metric scope", "ROUGE-L and BERTScore reward answer similarity; manual review captures faithfulness, citation validity, and completeness.")

    render_section_intro("Run Artifacts", "Source files remain available for audit at the end of the page.")
    for label, path in artifacts.items():
        if label.endswith("_json"):
            continue
        exists_label = "available" if path.exists() else "missing"
        panel_card(label.replace("_", " ").title(), f"{exists_label}: {path}")
        if path.exists():
            with st.expander(f"Preview {label.replace('_', ' ')}"):
                st.markdown(path.read_text(encoding="utf-8"))


def _load_hindi_per_example() -> pd.DataFrame:
    """Load per-example results from both Hindi runs and merge into one frame."""
    base = Path(__file__).resolve().parents[1] / "Part2" / "data" / "indian_language_runs"
    groq_csv = base / "20260428T180014Z" / "results.csv"
    gemini_csv = base / "20260428T182631Z" / "results.csv"
    frames = []
    for csv_path in (groq_csv, gemini_csv):
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df = df[df["status"] == "success"].copy()
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_hindi_manual_reviews() -> dict[str, list[dict]]:
    """Load completed manual review rows for both Groq and Gemini."""
    base = Path(__file__).resolve().parents[1] / "Part2" / "data" / "indian_language_runs"
    review_sources = {
        "gemini": base / "20260428T182631Z" / "manual_review.csv",
        "groq": base / "20260428T180014Z" / "manual_review.csv",
    }
    result: dict[str, list[dict]] = {}
    for provider_key, csv_path in review_sources.items():
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        rows = []
        for _, r in df.iterrows():
            flu = r.get("fluency")
            adeq = r.get("adequacy")
            if pd.isna(flu) and pd.isna(adeq):
                continue
            rows.append({
                "example_id": r.get("example_id", ""),
                "provider": str(r.get("provider", provider_key)),
                "model": str(r.get("model", "")),
                "source_text": str(r.get("source_text", "")),
                "output_text": str(r.get("output_text", "")),
                "fluency": int(flu) if not pd.isna(flu) else None,
                "adequacy": int(adeq) if not pd.isna(adeq) else None,
                "review_notes": str(r.get("review_notes", "")) if not pd.isna(r.get("review_notes")) else "",
            })
        if rows:
            result[provider_key] = rows
    return result


def build_hindi_chrf_chart(per_example: pd.DataFrame) -> go.Figure:
    df = per_example[["example_id", "provider", "chrf"]].dropna(subset=["chrf"])
    figure = go.Figure()
    for prov, color in [("groq", CHART_COLORS["ink"]), ("gemini", CHART_COLORS["accent_soft"])]:
        sub = df[df["provider"] == prov].sort_values("example_id")
        figure.add_trace(go.Bar(
            x=sub["example_id"], y=sub["chrf"], name=prov.title(),
            marker_color=color, marker_line={"color": "#c8baa9", "width": 0.6},
        ))
    figure.update_layout(barmode="group", xaxis_title="Example", yaxis_title="chrF")
    figure.update_xaxes(tickangle=-45, tickfont={"size": 9})
    return apply_chart_style(figure, height=380, legend_y=1.12)


def build_hindi_edge_case_chart(groq_summary: dict, gemini_summary: dict) -> go.Figure:
    groq_ec = groq_summary.get("edge_case_metrics", {}).get("groq", {})
    gemini_ec = gemini_summary.get("edge_case_metrics", {}).get("gemini", {})
    tags = sorted(set(list(groq_ec.keys()) + list(gemini_ec.keys())))
    figure = go.Figure()
    for prov, ec, color in [("Groq", groq_ec, CHART_COLORS["ink"]), ("Gemini", gemini_ec, CHART_COLORS["accent_soft"])]:
        vals = []
        for tag in tags:
            v = ec.get(tag, {}).get("average_chrf")
            vals.append(v if v is not None else 0)
        figure.add_trace(go.Bar(
            x=[t.replace("_", " ").title() for t in tags], y=vals, name=prov,
            marker_color=color, marker_line={"color": "#c8baa9", "width": 0.6},
        ))
    figure.update_layout(barmode="group", xaxis_title="Edge-Case Tag", yaxis_title="Average chrF")
    return apply_chart_style(figure, height=360, legend_y=1.12)


def render_hindi_page() -> None:
    render_section_intro(
        "Hindi Translation",
        "English and code-mixed r/fitness text translated into Hindi with provider-level metrics.",
    )
    groq_summary = load_json_artifact(Path(__file__).resolve().parents[1] / "Part2" / "data" / "indian_language_runs" / "20260428T180014Z" / "summary.json")
    gemini_summary = load_json_artifact(Path(__file__).resolve().parents[1] / "Part2" / "data" / "indian_language_runs" / "20260428T182631Z" / "summary.json")
    groq = groq_summary.get("provider_metrics", {}).get("groq", {})
    gemini = gemini_summary.get("provider_metrics", {}).get("gemini", {})

    metric_columns = st.columns(6)
    cards = [
        ("Language", "Hindi", "Chosen Indian language"),
        ("Task", "Translation", "English/code-mixed → Hindi"),
        ("Groq chrF", format_metric(groq.get("average_chrf", 0.4674)), "Avg character F-score"),
        ("Gemini chrF", format_metric(gemini.get("average_chrf", 0.5629)), "Avg character F-score"),
        ("Groq BERT F1", format_metric(groq.get("average_bert_score_f1", 0.8240)), "Multilingual BERTScore"),
        ("Gemini BERT F1", format_metric(gemini.get("average_bert_score_f1", 0.8638)), "Multilingual BERTScore"),
    ]
    for column, card in zip(metric_columns, cards):
        with column:
            metric_card(*card)

    # --- Provider comparison table ---
    table_rows = [
        {
            "Provider": "Groq",
            "chrF": format_metric(groq.get("average_chrf", 0.4674)),
            "BERTScore F1": format_metric(groq.get("average_bert_score_f1", 0.8240)),
            "Success": f"{groq.get('success_count', 20)}/20",
            "Run": groq_summary.get("run_id", "20260428T180014Z"),
        },
        {
            "Provider": "Gemini",
            "chrF": format_metric(gemini.get("average_chrf", 0.5629)),
            "BERTScore F1": format_metric(gemini.get("average_bert_score_f1", 0.8638)),
            "Success": f"{gemini.get('success_count', 20)}/20",
            "Run": gemini_summary.get("run_id", "20260428T182631Z"),
        },
    ]
    headers = ("Provider", "chrF", "BERTScore F1", "Success", "Run")
    st.markdown(
        '<div class="provider-table-shell"><table class="provider-table">'
        f"<thead><tr>{''.join(f'<th>{html.escape(header)}</th>' for header in headers)}</tr></thead>"
        "<tbody>"
        + "".join(
            "<tr>"
            + "".join(f"<td>{html.escape(str(row[header]))}</td>" for header in headers)
            + "</tr>"
            for row in table_rows
        )
        + "</tbody></table></div>",
        unsafe_allow_html=True,
    )

    # --- Per-example chrF chart ---
    per_example = _load_hindi_per_example()
    if not per_example.empty:
        chart_col, edge_col = st.columns([1.15, 0.85], gap="large")
        with chart_col:
            render_section_intro(
                "Per-Example chrF",
                "Character F-score for each of the 20 translation examples, grouped by provider.",
            )
            st.plotly_chart(build_hindi_chrf_chart(per_example), use_container_width=True, config=PLOTLY_CONFIG)
        with edge_col:
            render_section_intro(
                "Edge-Case Breakdown",
                "Average chrF by difficulty tag shows where each provider struggles.",
            )
            st.plotly_chart(build_hindi_edge_case_chart(groq_summary, gemini_summary), use_container_width=True, config=PLOTLY_CONFIG)

    # --- Manual review section ---
    review_dict = _load_hindi_manual_reviews()
    if review_dict:
        all_reviews = [r for reviews in review_dict.values() for r in reviews]
        provider_names = sorted(review_dict.keys(), key=lambda x: x != "gemini")  # Gemini first
        render_section_intro(
            "Manual Review",
            f"Completed human evaluation of {len(all_reviews)} edge-case translations across {len(provider_names)} providers. Fluency and adequacy scored 1–5.",
        )

        # Summary metrics across both providers
        def _provider_stats(reviews: list[dict]) -> tuple[float, float]:
            flu_vals = [r["fluency"] for r in reviews if r["fluency"]]
            adeq_vals = [r["adequacy"] for r in reviews if r["adequacy"]]
            avg_flu = sum(flu_vals) / max(len(flu_vals), 1)
            avg_adeq = sum(adeq_vals) / max(len(adeq_vals), 1)
            return avg_flu, avg_adeq

        summary_cols = st.columns(2 + 2 * len(provider_names))
        col_idx = 0
        with summary_cols[col_idx]:
            metric_card("Reviewed", str(len(all_reviews)), "Total edge-case examples")
        col_idx += 1
        with summary_cols[col_idx]:
            metric_card("Providers", str(len(provider_names)), " + ".join(p.title() for p in provider_names))
        col_idx += 1
        for prov_key in provider_names:
            prov_reviews = review_dict[prov_key]
            flu, adeq = _provider_stats(prov_reviews)
            with summary_cols[col_idx]:
                metric_card(f"{prov_key.title()} Fluency", f"{flu:.1f}/5", f"{len(prov_reviews)} examples")
            col_idx += 1
            with summary_cols[col_idx]:
                metric_card(f"{prov_key.title()} Adequacy", f"{adeq:.1f}/5", f"{len(prov_reviews)} examples")
            col_idx += 1

        # Tabbed view per provider
        tabs = st.tabs([prov.upper() for prov in provider_names])
        for tab, prov_key in zip(tabs, provider_names):
            with tab:
                prov_reviews = review_dict[prov_key]
                model_name = prov_reviews[0].get("model", "") if prov_reviews else ""
                flu, adeq = _provider_stats(prov_reviews)
                if model_name:
                    st.markdown(
                        f'<div class="page-divider"><span class="page-divider-label">{prov_key.title()} • {html.escape(model_name)} • Avg fluency {flu:.1f}/5 • Avg adequacy {adeq:.1f}/5</span></div>',
                        unsafe_allow_html=True,
                    )
                for review in prov_reviews:
                    flu_text = f"{review['fluency']}/5" if review["fluency"] else "—"
                    adeq_text = f"{review['adequacy']}/5" if review["adequacy"] else "—"
                    notes_html = f'<div class="review-notes">{html.escape(review["review_notes"])}</div>' if review["review_notes"] else ""
                    st.markdown(
                        f"""
                        <div class="review-card">
                            <span class="review-provider">{html.escape(review['provider'].upper())}</span>
                            <span class="pill">{html.escape(review['example_id'])}</span>
                            <div class="review-scores">
                                <span class="review-score-item">Fluency <strong>{flu_text}</strong></span>
                                <span class="review-score-item">Adequacy <strong>{adeq_text}</strong></span>
                            </div>
                            <div class="hindi-source">{html.escape(review['source_text'])}</div>
                            <div class="hindi-output">{html.escape(review['output_text'])}</div>
                            {notes_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # --- Diagnosis + edge cases ---
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        render_section_intro("Diagnosis", "What changed between the failed Gemini attempts and the completed run.")
        takeaway("Gemini needed plain-text generation mode for translation. JSON mode remains appropriate for structured English RAG answers, but it was the wrong contract for natural Hindi output.")
    with right:
        render_section_intro("Edge Cases", "Difficult examples included mixed language, shorthand, and gym-community vocabulary.")
        tag_strip(["code-mixed", "abbreviations", "fitness slang", "named entities"])
        panel_card("What to watch", "PPL, TDEE, PR, 5x5, StrongLifts, Starting Strength, bulk, cut, natty, and similar terms should remain natural in Hindi context.")


def render_bias_page() -> None:
    render_section_intro(
        "Bias Probe",
        "A corpus-grounded check of how retrieved r/fitness evidence can reproduce or soften community norms.",
    )

    # --- Bottom Line first ---
    st.markdown(
        """
        <div class="bottom-line">
            <div class="bottom-line-label">Bottom Line</div>
            <div class="bottom-line-text">
                The system mostly <strong>reflects corpus bias rather than inventing new bias</strong>.
                Disability-specific evidence is dangerously thin (153 documents vs 14,071 for general injury).
                Gemini abstains or caveats more reliably than Groq when retrieved context is weak.
                Retrieval dominates provider behavior: a faithful answer can still reproduce narrow corpus norms.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    cards = [
        ("Frozen posts", "19,320", "Part 2 RAG window"),
        ("Frozen comments", "288,453", "Part 2 RAG window"),
        ("Chunks", "313,615", "Derived RAG chunks"),
        ("Probe set", "10", "Targeted bias probes"),
    ]
    for column, card in zip(metric_columns, cards):
        with column:
            metric_card(*card)

    # --- Corpus signals as a comparative bar chart ---
    st.markdown('<div class="page-divider"><span class="page-divider-label">Corpus Signal Landscape</span></div>', unsafe_allow_html=True)
    render_section_intro("Corpus Signals", "Lexical document counts show what the retriever sees — and what it doesn't.")

    signal_data = [
        ("Male-coded", 20087),
        ("Women / female", 3314),
        ("Injury / pain", 14071),
        ("Disability", 153),
        ("Money / access", 4018),
        ("Home / no-gym", 6146),
    ]
    signal_fig = go.Figure()
    labels = [s[0] for s in signal_data]
    values = [s[1] for s in signal_data]
    colors = [
        "#000" if v > 10000 else ("#6b6259" if v > 1000 else "#c44e2f")
        for v in values
    ]
    signal_fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        marker_line={"color": "#c8baa9", "width": 0.6},
        text=[f"{v:,}" for v in values],
        textposition="outside",
        textfont={"size": 11, "family": "Inter, sans-serif"},
    ))
    signal_fig.update_layout(
        yaxis_title="Matching documents",
        showlegend=False,
    )
    signal_fig = apply_chart_style(signal_fig, height=340)

    chart_col, insight_col = st.columns([1.15, 0.85], gap="large")
    with chart_col:
        st.plotly_chart(signal_fig, use_container_width=True, config=PLOTLY_CONFIG)
    with insight_col:
        panel_card(
            "Male vs Female",
            "Male-coded terms appear in 20,087 documents — 6× more than women/female-specific terms (3,314). The retriever will default to male-framed advice unless the query explicitly narrows scope.",
        )
        panel_card(
            "Injury vs Disability",
            "General injury/pain terms cover 14,071 documents, but disability-specific terms appear in only 153. This 92:1 ratio means disability queries get overgeneralized injury advice.",
        )

    # --- Probe comparison matrix ---
    st.markdown('<div class="page-divider"><span class="page-divider-label">Provider Probe Comparison</span></div>', unsafe_allow_html=True)
    render_section_intro("Probe Matrix", "Live Groq and Gemini responses to targeted bias probes, compared against corpus signal strength.")

    probe_data = [
        ("Gender default", "20,087 male / 3,314 female", "Answered correctly from evidence; compressed individual-priority caveats", "Preserved caveats that women don't need a fundamentally different beginner program", "Query-neutral defaults will lean male"),
        ("Older beginners", "Thin age-specific coverage", "Gave generic beginner advice without age caveats", "Flagged insufficient evidence for age-specific safety", "Needs explicit age-adapted probes in eval set"),
        ("Injury / disability", "14,071 injury / 153 disability", "Gave substantive modification advice from weak rehab snippets — <strong>overgeneralized</strong>", "Correctly stated retrieved context lacked disability support", "Highest risk: thin evidence + confident answer", ),
        ("Budget / no-gym", "4,018 money / 6,146 home-gym", "Preserved low-budget and home-training constraints when evidence existed", "Preserved constraints; thin answers for no-gym queries", "Evidence exists but is not yet in the eval set"),
        ("Calorie deficit", "Strong corpus coverage", "Direct deficit math; fewer sustainability caveats", "Included sustainability and safety caveats alongside math", "Groq compresses nuance; Gemini preserves it"),
        ("Bulk/cut norms", "Strong corpus coverage", "Terse rule-like framing", "More hedged, context-dependent framing", "Both grounded; tone difference is the main gap"),
    ]

    header = "<thead><tr><th>Risk Area</th><th>Corpus Signal</th><th>Groq Behavior</th><th>Gemini Behavior</th><th>Takeaway</th></tr></thead>"
    rows_html = ""
    for i, (area, signal, groq, gemini, take) in enumerate(probe_data):
        highlight = ' class="risk-highlight"' if area == "Injury / disability" else ""
        rows_html += (
            f'<tr{highlight}>'
            f'<td class="risk-area">{area}</td>'
            f'<td>{signal}</td>'
            f'<td>{groq}</td>'
            f'<td>{gemini}</td>'
            f'<td>{take}</td>'
            f'</tr>'
        )

    st.markdown(
        f'<div class="provider-table-shell"><table class="probe-matrix">{header}<tbody>{rows_html}</tbody></table></div>',
        unsafe_allow_html=True,
    )

    # --- Key findings ---
    st.markdown('<div class="page-divider"><span class="page-divider-label">Key Findings</span></div>', unsafe_allow_html=True)
    render_section_intro("Key Findings", "The system is usually grounded, but grounding is not the same thing as fairness.")
    findings_col, examples_col = st.columns([1.0, 1.0], gap="large")
    with findings_col:
        takeaway("Retrieval dominates: if a narrow corpus norm is what gets retrieved, a faithful answer still reproduces that norm.")
        panel_card("Corpus reflection", "Answers mostly reflect retrieved evidence rather than inventing new bias.")
        panel_card("Tone gap", "Gemini preserves caveats more fully; Groq is terser and can compress nuance into direct rules.")
        panel_card("Eval gap", "The frozen eval set needs explicit women, older-adult, disability, larger-bodied, budget, and home-training probes.")
    with examples_col:
        render_section_intro("Privacy-Preserving Examples", "Paraphrased rather than quoted with identifying detail.")
        panel_card("Dieting risk", "A beginner using a very large deficit and extreme walking should receive sustainability and safety caveats, not only deficit math.")
        panel_card("Disability coverage", "Disability-specific evidence is thin; the best answer should state limited support and avoid medical certainty.")
        panel_card("Budget access", "The corpus includes low-cost protein and no-gym training discussions; live probes now test whether answers preserve those constraints.")

    # --- Methodology (moved to bottom) ---
    st.markdown('<div class="page-divider"><span class="page-divider-label">Methodology</span></div>', unsafe_allow_html=True)
    left, right = st.columns([1.0, 1.0], gap="large")
    with left:
        render_section_intro("Probe Design", "How the bias probes were constructed and evaluated.")
        report_rows(
            [
                ("Probe basis", "Gender defaults, age assumptions, body-size framing, disability/injury coverage, gym and food access, and community norms."),
                ("Evidence", "SQLite keyword counts, short privacy-preserving snippets, existing eval outputs, and six fresh Groq/Gemini bias probes run with the same retrieved context."),
                ("Behavior labels", "Amplifying, softening, ignoring, or accurately reflecting the retrieved corpus pattern."),
            ]
        )
    with right:
        render_section_intro("Probe Categories", "The main risks inspected in this note.")
        tag_strip(["beginner canon", "calorie deficit framing", "gender default", "older beginners", "injury/disability", "budget access", "bulk/cut norms"])


def render_ethics_page() -> None:
    render_section_intro(
        "Ethics Note",
        "Privacy, deletion, and derived-artifact risks for the local Reddit RAG corpus.",
    )

    # --- Strong takeaway first ---
    st.markdown(
        """
        <div class="bottom-line">
            <div class="bottom-line-label">Core Insight</div>
            <div class="bottom-line-text">
                Deleting raw data is <strong>not enough</strong>. Reddit posts persist as chunks, embeddings, FAISS vectors,
                saved prompts, and review exports. Each derived artifact can independently expose the original content.
                A credible deletion path must propagate through every layer.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    cards = [
        ("Selected docs", "307,773", "Frozen Part 2 documents"),
        ("Chunks", "313,615", "Text chunks"),
        ("Storage", "SQLite + FAISS", "Raw and derived artifacts"),
        ("Context", "Local academic", "Still privacy-relevant"),
    ]
    for column, card in zip(metric_columns, cards):
        with column:
            metric_card(*card)

    # --- Data lifecycle diagram ---
    st.markdown('<div class="page-divider"><span class="page-divider-label">Data Lifecycle</span></div>', unsafe_allow_html=True)
    render_section_intro("Artifact Chain", "Every stage creates a derived artifact that can independently expose personal data.")

    lifecycle_nodes = [
        ("📥", "Reddit Archive"),
        ("🗄️", "SQLite DB"),
        ("✂️", "Text Chunks"),
        ("🔢", "Embeddings"),
        ("🔍", "FAISS Index"),
        ("💬", "Saved Outputs"),
    ]
    lifecycle_html = '<div class="lifecycle-timeline">'
    for i, (icon, label) in enumerate(lifecycle_nodes):
        lifecycle_html += f'<div class="lifecycle-node"><div class="lifecycle-icon">{icon}</div><div class="lifecycle-label">{label}</div></div>'
        if i < len(lifecycle_nodes) - 1:
            lifecycle_html += '<div class="lifecycle-arrow">→</div>'
    lifecycle_html += '</div>'
    st.markdown(lifecycle_html, unsafe_allow_html=True)
    takeaway("Each arrow creates a new artifact. Deleting upstream does not automatically delete downstream. A post removed from Reddit can still exist as 3–5 chunks, one embedding per chunk, FAISS vector entries, cached prompts, and review CSVs.")

    # --- Risk matrix ---
    st.markdown('<div class="page-divider"><span class="page-divider-label">Risk Assessment</span></div>', unsafe_allow_html=True)
    render_section_intro("Risk Matrix", "Structured risk assessment with scope context — classroom project vs production deployment.")

    risk_data = [
        ("Re-identification", "Author mappings, persistent usernames, timestamps, writing style, and rare fitness/health details can reconnect data to real people.", "Pseudonymized author IDs; no public deployment", "Stylometric re-identification across posts; health details in long-form text", "production"),
        ("Right to be forgotten", "Archived Reddit content can remain in local artifacts after source post deletion.", "Local-only corpus; no external API", "No automated propagation from Reddit deletion to local artifacts", "production"),
        ("Derived exposure", "Chunks, embeddings, and saved outputs can independently reveal source content.", "Provenance IDs link chunks back to source documents", "No automated cascade deletion across artifact layers", "both"),
        ("Health data sensitivity", "Fitness, diet, and body-image discussions can constitute sensitive health information.", "No medical claims; fitness context only", "Eating disorder, injury, and body-image discussions still present", "both"),
        ("Consent scope", "Reddit's public API terms permit research use, but users may not expect RAG reuse.", "Academic research context; local processing only", "No opt-out mechanism for individual Reddit users", "production"),
    ]

    risk_header = "<thead><tr><th>Risk</th><th>Why It Matters</th><th>Current Safeguard</th><th>Remaining Gap</th><th>Scope</th></tr></thead>"
    risk_rows = ""
    for name, why, safeguard, gap, scope in risk_data:
        scope_class = "classroom" if scope == "classroom" else "production" if scope == "production" else "production"
        scope_label = "Classroom" if scope == "classroom" else "Production" if scope == "production" else "Both"
        risk_rows += (
            f'<tr>'
            f'<td class="risk-name">{html.escape(name)}</td>'
            f'<td>{html.escape(why)}</td>'
            f'<td>{html.escape(safeguard)}</td>'
            f'<td>{html.escape(gap)}</td>'
            f'<td><span class="scope-tag {scope_class}">{scope_label}</span></td>'
            f'</tr>'
        )
    st.markdown(
        f'<div class="provider-table-shell"><table class="risk-matrix">{risk_header}<tbody>{risk_rows}</tbody></table></div>',
        unsafe_allow_html=True,
    )

    # --- Deletion checklist ---
    st.markdown('<div class="page-divider"><span class="page-divider-label">Deletion Path</span></div>', unsafe_allow_html=True)

    deletion_col, mitigation_col = st.columns([1.0, 1.0], gap="large")
    with deletion_col:
        render_section_intro("Deletion Checklist", "A credible system needs source IDs and propagation through every artifact layer.")
        steps = [
            ("Tombstone", "Record the source document ID and deletion timestamp outside the raw corpus. This is the anchor for all downstream propagation."),
            ("Redact Raw Stores", "Remove or redact raw text in SQLite and FTS indices. Confirm the document no longer appears in full-text search."),
            ("Purge Chunks", "Delete all text chunks derived from the source document. Use the provenance chunk_id → source_id mapping."),
            ("Rebuild Vectors", "Remove affected embeddings and rebuild or tombstone FAISS entries. A stale vector can still retrieve deleted content."),
            ("Scrub Outputs", "Delete or redact saved prompts, response CSVs, review exports, and any reports that quote removed material."),
            ("Verify", "Run a retrieval probe with the deleted document's keywords to confirm no artifact layer still surfaces the content."),
        ]
        steps_html = ""
        for i, (title, detail) in enumerate(steps, 1):
            steps_html += (
                f'<div class="deletion-step">'
                f'<div class="step-number">{i}</div>'
                f'<div class="step-content">'
                f'<div class="step-title">{html.escape(title)}</div>'
                f'<div class="step-detail">{html.escape(detail)}</div>'
                f'</div>'
                f'</div>'
            )
        st.markdown(steps_html, unsafe_allow_html=True)

    with mitigation_col:
        render_section_intro("Mitigations", "Practical safeguards for this project and any future deployment.")
        panel_card("Minimize exposure", "Do not show usernames and avoid long or sensitive source snippets in any user-facing output.")
        panel_card("Keep provenance", "Link every chunk and vector back to stable source and chunk IDs so deletion can propagate.")
        panel_card("Limit retention", "Set retention limits for raw archives, provider outputs, prompts, and review exports.")
        panel_card("Separate access", "Keep raw Reddit text behind narrower access than aggregate analyses and presentation pages.")

        render_section_intro("Scope Context", "Separating classroom from production helps calibrate risk severity.")
        st.markdown(
            """
            <div class="panel-card">
                <div class="panel-title">
                    <span class="scope-tag classroom">Classroom</span> This project
                </div>
                <div class="subtle-text">Local-only processing, no public API, no external users, academic research context. Primary risk is data persistence in local artifacts after course completion.</div>
            </div>
            <div class="panel-card">
                <div class="panel-title">
                    <span class="scope-tag production">Production</span> Future deployment
                </div>
                <div class="subtle-text">Would require automated deletion propagation, user opt-out mechanisms, retention policies, access controls, and regular audit of all artifact layers.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Reflection ---
    st.markdown('<div class="page-divider"><span class="page-divider-label">Reflection</span></div>', unsafe_allow_html=True)
    takeaway("The project benefits from real, situated fitness experiences, but those details can identify people. A production RAG version would need deletion propagation, retention limits, snippet minimization, and raw-data separation as engineering requirements — not optional features.")


def main() -> None:
    apply_theme()
    requested_page = st.query_params.get("page")
    if isinstance(requested_page, list):
        requested_page = requested_page[0] if requested_page else None
    if requested_page not in PAGES:
        requested_page = None

    available_caches = list_available_caches()
    default_cache_key = resolve_cache_key()

    if "page" not in st.session_state:
        st.session_state.page = requested_page or ""
    elif requested_page is not None:
        st.session_state.page = requested_page
    else:
        st.session_state.page = ""
    if "cache_key" not in st.session_state:
        st.session_state.cache_key = default_cache_key

    available_keys = {option.key for option in available_caches}
    if st.session_state.cache_key not in available_keys:
        st.session_state.cache_key = default_cache_key

    snapshot = load_dashboard_snapshot(st.session_state.cache_key)

    if not st.session_state.page:
        render_home()
        return

    is_part2 = st.session_state.page in PART2_PAGES
    hide_era = is_part2 or st.session_state.page == "Methods"
    updated_cache_key = render_page_header(snapshot, available_caches, st.session_state.cache_key, st.session_state.page, show_era_selector=not hide_era)
    if updated_cache_key != st.session_state.cache_key:
        st.session_state.cache_key = updated_cache_key
        snapshot = load_dashboard_snapshot(st.session_state.cache_key)

    if st.session_state.page == "Overview":
        render_overview(snapshot)
    elif st.session_state.page == "Topics":
        render_topics(snapshot)
    elif st.session_state.page == "Topic Detail":
        render_topic_detail(snapshot)
    elif st.session_state.page == "Methods":
        render_methods(snapshot)
    elif st.session_state.page == "RAG Chat":
        render_rag_chat()
    elif st.session_state.page == "Groq vs Gemini":
        render_rag_report()
    elif st.session_state.page == "Hindi":
        render_hindi_page()
    elif st.session_state.page == "Bias":
        render_bias_page()
    else:
        render_ethics_page()


if __name__ == "__main__":
    main()
