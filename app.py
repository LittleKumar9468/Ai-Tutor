"""Falcon Students - Phase 4: AI learning, revision, analytics, and mock interviews.

Configure credentials in .streamlit/secrets.toml or environment variables.
Never put API keys, MongoDB URIs, or email app passwords in source code.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import smtplib
import ssl
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from email.mime.text import MIMEText
from typing import Any

import pandas as pd
import streamlit as st
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collation import Collation
from pymongo.errors import ConfigurationError, DuplicateKeyError, PyMongoError

try:
    from google import genai as modern_genai
except ImportError:
    modern_genai = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Falcon Students",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.WARNING)
LOGGER = logging.getLogger("falcon_students")

APP_SESSION_VERSION = 3
PBKDF2_ITERATIONS = 600_000
OTP_TTL_SECONDS = 5 * 60
OTP_RESEND_WAIT_SECONDS = 60
OTP_MAX_ATTEMPTS = 5
LOGIN_MAX_FAILURES = 5
LOGIN_LOCK_SECONDS = 15 * 60
MAX_CHAT_MESSAGES_PER_SUBJECT = 200
MAX_CONTEXT_MESSAGES = 12
MAX_PROMPT_LENGTH = 4_000
MAX_QUIZ_HISTORY = 500
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
EMAIL_COLLATION = Collation(locale="en", strength=2)
SUBJECTS = ("DSA", "OS", "DBMS")
LEARNING_STYLES = ("Simple & Analogies", "Code Heavy", "Bullet Points")
LEVELS = ("Beginner", "Intermediate", "Advanced")
MODEL_NAME = "gemini-3.5-flash-lite"

SUBJECT_TOPICS: dict[str, list[str]] = {
    "DSA": [
        "Arrays", "Strings", "Searching", "Sorting", "Linked Lists",
        "Stacks and Queues", "Recursion", "Trees", "Graphs", "Hashing",
        "Dynamic Programming",
    ],
    "OS": [
        "Process Management", "CPU Scheduling", "Deadlocks", "Memory Management",
        "Virtual Memory", "Synchronization", "File Systems",
    ],
    "DBMS": [
        "Relational Model", "SQL Queries", "Joins", "Normalization", "Transactions",
        "Indexing", "ER Model", "Keys and Constraints",
    ],
}


def get_setting(name: str, default: str = "") -> str:
    """Read environment variables first, then Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value.strip()
    try:
        secret_value = st.secrets.get(name, default)
    except Exception:
        secret_value = default
    return str(secret_value).strip() if secret_value is not None else default


# Credentials supplied for this local project. Keep this file private; do not commit it to Git.
MONGO_URI = get_setting("MONGO_URI")
GEMINI_API_KEY = get_setting("GEMINI_API_KEY")
SENDER_EMAIL = get_setting("SENDER_EMAIL")
SENDER_PASSWORD = get_setting("SENDER_PASSWORD")
MODEL_NAME = get_setting("GEMINI_MODEL", "gemini-3.5-flash-lite")

# -----------------------------------------------------------------------------
# Theme: vibrant multi-color visual layer + 3-line hamburger sidebar toggle.
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800;900&display=swap');

      :root {
        --falcon-violet: #6366f1;
        --falcon-purple: #8b5cf6;
        --falcon-pink: #ec4899;
        --falcon-rose: #f43f5e;
        --falcon-amber: #f59e0b;
        --falcon-emerald: #10b981;
        --falcon-cyan: #06b6d4;
        --falcon-ink: #0f172a;
        --falcon-muted: #475569;
        --falcon-border: #e2e8f0;
        --falcon-surface: #ffffff;
        --falcon-radius: 20px;
      }

      html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      }

      /* 1. TRANSPARENT TOP HEADER & HIDE ONLY RIGHT-SIDE SHARE/STAR/EDIT/DEPLOY ICONS */
      header[data-testid="stHeader"] {
        background: transparent !important;
        box-shadow: none !important;
        backdrop-filter: none !important;
        border: none !important;
        height: 3.2rem !important;
      }
      [data-testid="stToolbarActions"],
      [data-testid="stAppDeployButton"],
      .stAppDeployButton,
      [data-testid="stMainMenu"],
      #MainMenu,
      [data-testid="stStatusWidget"],
      [data-testid="stDecoration"],
      footer {
        display: none !important;
        visibility: hidden !important;
      }
      /* Hide chain-link anchor icons next to headings */
      [data-testid="stHeaderActionElements"],
      h1 a, h2 a, h3 a, h4 a {
        display: none !important;
      }

      /* 2. 3-LINE HAMBURGER BUTTON (☰) FOR OPENING & CLOSING SIDEBAR */
      [data-testid="stSidebarCollapseButton"],
      [data-testid="stExpandSidebarButton"],
      [data-testid="collapsedControl"] {
        opacity: 1 !important;
        visibility: visible !important;
        display: flex !important;
        align-items: center !important;
        z-index: 999999 !important;
      }

      button[data-testid="stExpandSidebarButton"],
      [data-testid="collapsedControl"] button,
      button[data-testid="stSidebarCollapseButton"],
      [data-testid="stSidebarCollapseButton"] button {
        width: 42px !important;
        height: 42px !important;
        min-width: 42px !important;
        min-height: 42px !important;
        border-radius: 12px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        padding: 0 !important;
        transition: all 0.2s ease !important;
      }

      /* Hamburger style when sidebar is CLOSED (top-left of screen) */
      button[data-testid="stExpandSidebarButton"],
      [data-testid="collapsedControl"] button {
        background: linear-gradient(135deg, #4f46e5 0%, #9333ea 55%, #ec4899 100%) !important;
        border: 1.5px solid rgba(255, 255, 255, 0.45) !important;
        box-shadow: 0 8px 22px rgba(147, 51, 234, 0.42) !important;
        color: #ffffff !important;
        margin-top: 0.35rem !important;
        margin-left: 0.4rem !important;
      }

      /* Hamburger style when sidebar is OPEN (top-left inside sidebar) */
      button[data-testid="stSidebarCollapseButton"],
      [data-testid="stSidebarCollapseButton"] button {
        background: rgba(255, 255, 255, 0.12) !important;
        border: 1.5px solid rgba(255, 255, 255, 0.28) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
      }

      button[data-testid="stExpandSidebarButton"]:hover,
      [data-testid="collapsedControl"] button:hover,
      button[data-testid="stSidebarCollapseButton"]:hover,
      [data-testid="stSidebarCollapseButton"] button:hover {
        transform: scale(1.06) !important;
        filter: brightness(1.12) !important;
      }

      /* Replace default arrow SVG with clean 3-line Hamburger icon ☰ */
      button[data-testid="stExpandSidebarButton"] svg,
      [data-testid="collapsedControl"] button svg,
      button[data-testid="stSidebarCollapseButton"] svg,
      [data-testid="stSidebarCollapseButton"] button svg,
      button[data-testid="stExpandSidebarButton"] span,
      [data-testid="stSidebarCollapseButton"] button span {
        display: none !important;
      }

      button[data-testid="stExpandSidebarButton"]::after,
      [data-testid="collapsedControl"] button::after,
      button[data-testid="stSidebarCollapseButton"]::after,
      [data-testid="stSidebarCollapseButton"] button::after {
        content: "☰" !important;
        font-size: 1.45rem !important;
        font-weight: 900 !important;
        line-height: 1 !important;
        color: #ffffff !important;
        display: block !important;
      }

      /* 3. MULTI-COLOR AURORA MESH BACKGROUND */
      .stApp {
        background:
          radial-gradient(circle at 6% 8%, rgba(139, 92, 246, 0.14), transparent 32%),
          radial-gradient(circle at 94% 10%, rgba(236, 72, 153, 0.12), transparent 30%),
          radial-gradient(circle at 88% 88%, rgba(16, 185, 129, 0.12), transparent 34%),
          radial-gradient(circle at 12% 85%, rgba(245, 158, 11, 0.11), transparent 30%),
          linear-gradient(180deg, #f8faff 0%, #f1f5f9 100%);
        color: var(--falcon-ink);
      }

      [data-testid="stMainBlockContainer"] {
        width: 100%;
        max-width: 1440px;
        padding: 3.1rem clamp(1.2rem, 2.8vw, 2.8rem) 3rem !important;
      }
      [data-testid="stAppViewContainer"] > .main { background: transparent; }
      [data-testid="stVerticalBlock"] { gap: 0.95rem; }

      h1, h2, h3, h4 {
        font-family: 'Outfit', 'Plus Jakarta Sans', sans-serif;
        color: var(--falcon-ink);
        letter-spacing: -0.03em;
        font-weight: 800;
      }
      h1 { font-size: clamp(1.95rem, 2.8vw, 2.65rem); line-height: 1.15; }
      h2 { font-size: clamp(1.4rem, 2.1vw, 1.85rem); line-height: 1.25; }
      h3 { font-size: clamp(1.12rem, 1.6vw, 1.35rem); line-height: 1.3; }
      hr { border-color: rgba(148, 163, 184, 0.28); margin: 1.25rem 0; }

      /* 4. SLEEK COSMIC SIDEBAR WITH CUIMS-STYLE TOP HEADER BAR */
      [data-testid="stSidebar"] {
        background:
          radial-gradient(circle at 90% 10%, rgba(236, 72, 153, 0.22), transparent 35%),
          radial-gradient(circle at 10% 85%, rgba(6, 182, 212, 0.20), transparent 40%),
          linear-gradient(180deg, #0b0f19 0%, #17153b 52%, #1e1b4b 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.14) !important;
      }
      /* Top header inside sidebar: [ ☰ ]  🎓 Falcon Students */
      [data-testid="stSidebarHeader"] {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 0.75rem !important;
        height: auto !important;
        min-height: 3.8rem !important;
        padding: 0.85rem 1rem 0.65rem 1rem !important;
        margin: 0 !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12) !important;
      }
      [data-testid="stLogoSpacer"] {
        display: none !important;
      }
      [data-testid="stSidebarHeader"]::after {
        content: "🎓 Falcon Students";
        font-family: 'Outfit', sans-serif;
        font-size: 1.18rem;
        font-weight: 900;
        color: #ffffff;
        letter-spacing: -0.02em;
        white-space: nowrap;
      }
      [data-testid="stSidebarUserContent"] {
        padding-top: 0.75rem !important;
      }
      [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.55rem !important;
      }
      [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.14) !important;
        margin: 0.65rem 0 !important;
      }
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] span,
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #cbd5e1 !important;
      }

      /* Sidebar Navigation Buttons */
      [data-testid="stSidebar"] div.stButton > button {
        justify-content: flex-start !important;
        text-align: left !important;
        padding: 0.68rem 1rem !important;
        min-height: 2.75rem !important;
        border-radius: 13px !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        transition: all 0.22s ease !important;
      }
      /* Inactive Sidebar Buttons */
      [data-testid="stSidebar"] div.stButton > button[kind="secondary"],
      [data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-secondary"] {
        background: rgba(255, 255, 255, 0.07) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #f1f5f9 !important;
      }
      [data-testid="stSidebar"] div.stButton > button[kind="secondary"] *,
      [data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-secondary"] * {
        color: #f1f5f9 !important;
      }
      [data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover,
      [data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-secondary"]:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        border-color: rgba(167, 139, 250, 0.55) !important;
        transform: translateX(4px) !important;
      }
      /* Active Sidebar Button & Logout Button */
      [data-testid="stSidebar"] div.stButton > button[kind="primary"],
      [data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #f43f5e 0%, #8b5cf6 55%, #06b6d4 100%) !important;
        border: none !important;
        color: #ffffff !important;
        box-shadow: 0 8px 22px rgba(236, 72, 153, 0.35) !important;
      }
      [data-testid="stSidebar"] div.stButton > button[kind="primary"] *,
      [data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-primary"] * {
        color: #ffffff !important;
      }
      /* Sidebar Selectbox & Slider */
      [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: rgba(15, 23, 42, 0.85) !important;
        border-color: rgba(255, 255, 255, 0.18) !important;
        color: #ffffff !important;
      }
      [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] * {
        color: #ffffff !important;
      }

      /* 5. MAIN BUTTONS: HIGH-CONTRAST WHITE TEXT & VIBRANT GRADIENTS */
      div.stButton > button,
      div.stFormSubmitButton > button,
      [data-testid="stFormSubmitButton"] > button,
      [data-testid="stDownloadButton"] > button {
        min-height: 2.85rem !important;
        padding: 0.7rem 1.25rem !important;
        border-radius: 13px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.01em !important;
        transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
      }
      div.stButton > button[kind="primary"],
      div.stButton > button[data-testid="baseButton-primary"],
      div.stFormSubmitButton > button,
      [data-testid="stFormSubmitButton"] > button,
      [data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #4f46e5 0%, #9333ea 52%, #ec4899 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 10px 24px -6px rgba(147, 51, 234, 0.48) !important;
      }
      div.stButton > button[kind="primary"] *,
      div.stButton > button[data-testid="baseButton-primary"] *,
      div.stFormSubmitButton > button *,
      [data-testid="stFormSubmitButton"] > button *,
      [data-testid="stDownloadButton"] > button * {
        color: #ffffff !important;
        fill: #ffffff !important;
      }
      div.stButton > button[kind="primary"]:hover,
      div.stButton > button[data-testid="baseButton-primary"]:hover,
      div.stFormSubmitButton > button:hover,
      [data-testid="stFormSubmitButton"] > button:hover,
      [data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 14px 28px -6px rgba(236, 72, 153, 0.58) !important;
        filter: brightness(1.06) !important;
      }
      /* Main Area Secondary Buttons */
      [data-testid="stMainBlockContainer"] div.stButton > button[kind="secondary"],
      [data-testid="stMainBlockContainer"] div.stButton > button[data-testid="baseButton-secondary"] {
        background: #ffffff !important;
        color: #1e1b4b !important;
        border: 1.5px solid #cbd5e1 !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05) !important;
      }
      [data-testid="stMainBlockContainer"] div.stButton > button[kind="secondary"] *,
      [data-testid="stMainBlockContainer"] div.stButton > button[data-testid="baseButton-secondary"] * {
        color: #1e1b4b !important;
      }
      [data-testid="stMainBlockContainer"] div.stButton > button[kind="secondary"]:hover,
      [data-testid="stMainBlockContainer"] div.stButton > button[data-testid="baseButton-secondary"]:hover {
        border-color: #8b5cf6 !important;
        background: #f5f3ff !important;
        transform: translateY(-2px) !important;
      }

      /* 6. FORMS, INPUTS & PILL RADIO TABS */
      [data-testid="stForm"] {
        padding: clamp(1.25rem, 2.2vw, 1.85rem) !important;
        border: 1px solid rgba(139, 92, 246, 0.22) !important;
        border-top: 4px solid #8b5cf6 !important;
        border-radius: 22px !important;
        background: rgba(255, 255, 255, 0.95) !important;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.08) !important;
        backdrop-filter: blur(16px);
      }
      [data-testid="stTextInput"] input,
      [data-testid="stNumberInput"] input,
      [data-testid="stTextArea"] textarea,
      [data-testid="stMainBlockContainer"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #f8fafc !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 12px !important;
        color: #0f172a !important;
        font-weight: 500 !important;
        transition: all 0.18s ease !important;
      }
      [data-testid="stTextInput"] input:focus,
      [data-testid="stNumberInput"] input:focus,
      [data-testid="stTextArea"] textarea:focus {
        background: #ffffff !important;
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.15) !important;
      }
      [data-testid="stMainBlockContainer"] [data-testid="stTextInput"] label,
      [data-testid="stMainBlockContainer"] [data-testid="stTextArea"] label,
      [data-testid="stMainBlockContainer"] [data-testid="stSelectbox"] label,
      [data-testid="stMainBlockContainer"] [data-testid="stNumberInput"] label {
        color: #1e293b !important;
        font-size: 0.9rem !important;
        font-weight: 700 !important;
      }

      /* Pill-style Radio Switcher (Login / Sign Up / Forgot Password) */
      [data-testid="stRadio"] [role="radiogroup"] {
        background: #eef2ff;
        padding: 0.35rem;
        border-radius: 14px;
        border: 1px solid #c7d2fe;
        gap: 0.4rem !important;
        display: inline-flex;
      }
      [data-testid="stRadio"] label[data-baseweb="radio"] {
        border-radius: 10px !important;
        padding: 0.45rem 0.95rem !important;
        background: rgba(255, 255, 255, 0.75);
        font-weight: 700 !important;
        transition: all 0.18s ease;
      }
      [data-testid="stRadio"] label[data-baseweb="radio"]:hover {
        background: #ffffff;
      }

      /* 7. MULTI-COLOR METRIC CARDS, CONTAINERS, EXPANDERS */
      [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        border-top: 4px solid #8b5cf6 !important;
        padding: 1.2rem 1.35rem !important;
        border-radius: 18px !important;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
      }
      [data-testid="column"]:nth-of-type(1) [data-testid="stMetric"] {
        border-top-color: #8b5cf6 !important;
        background: linear-gradient(180deg, #ffffff 0%, #f5f3ff 100%) !important;
      }
      [data-testid="column"]:nth-of-type(2) [data-testid="stMetric"] {
        border-top-color: #10b981 !important;
        background: linear-gradient(180deg, #ffffff 0%, #ecfdf5 100%) !important;
      }
      [data-testid="column"]:nth-of-type(3) [data-testid="stMetric"] {
        border-top-color: #f59e0b !important;
        background: linear-gradient(180deg, #ffffff 0%, #fffbeb 100%) !important;
      }
      [data-testid="column"]:nth-of-type(4) [data-testid="stMetric"] {
        border-top-color: #ec4899 !important;
        background: linear-gradient(180deg, #ffffff 0%, #fdf2f8 100%) !important;
      }
      [data-testid="stMetric"]:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 16px 34px rgba(99, 102, 241, 0.14) !important;
      }
      [data-testid="stMetricLabel"] p {
        color: #475569 !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }
      [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 900 !important;
        font-size: 2rem !important;
      }

      /* Bordered Containers */
      [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 20px !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        background: rgba(255, 255, 255, 0.92) !important;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06) !important;
        padding: 1.25rem !important;
        transition: transform 0.22s ease, box-shadow 0.22s ease !important;
      }
      [data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 18px 40px rgba(99, 102, 241, 0.12) !important;
      }

      [data-testid="stExpander"] {
        border: 1px solid #cbd5e1 !important;
        border-radius: 16px !important;
        background: #ffffff !important;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04) !important;
        overflow: hidden;
      }
      [data-testid="stExpander"] summary {
        padding: 0.85rem 1.1rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #f8fafc, #f1f5f9);
      }
      [data-testid="stProgressBar"] > div { border-radius: 99px; overflow: hidden; }
      [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #6366f1, #ec4899, #f59e0b, #10b981) !important;
        border-radius: 99px;
      }
      [data-testid="stChatMessage"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 18px !important;
        padding: 1rem 1.15rem !important;
        background: #ffffff !important;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.04) !important;
        margin-bottom: 0.75rem !important;
      }

      /* 8. LOGIN HERO PANEL (NETFLIX / COURSERA CINEMATIC SHOWCASE) */
      .falcon-brand-panel {
        position: relative;
        isolation: isolate;
        overflow: hidden;
        padding: clamp(1.8rem, 3vw, 2.75rem);
        min-height: 560px;
        color: #fff;
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 28px;
        background:
          radial-gradient(circle at 88% 12%, rgba(236, 72, 153, 0.42), transparent 34%),
          radial-gradient(circle at 12% 90%, rgba(6, 182, 212, 0.35), transparent 38%),
          radial-gradient(circle at 50% 50%, rgba(139, 92, 246, 0.25), transparent 50%),
          linear-gradient(140deg, #090d16 0%, #17153b 45%, #2e1065 80%, #4c0519 100%);
        box-shadow: 0 28px 75px rgba(15, 23, 42, 0.32);
        animation: falconFadeUp 0.6s ease both;
      }
      .falcon-brand-panel h1 {
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif;
        font-weight: 900;
        line-height: 1.1;
        letter-spacing: -0.04em;
        margin: 1rem 0 0.85rem;
        font-size: clamp(2.1rem, 3vw, 3.2rem);
      }
      .falcon-brand-panel .accent {
        background: linear-gradient(90deg, #38bdf8 0%, #c084fc 38%, #f472b6 70%, #fbbf24 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
      }
      .falcon-brand-panel .tagline {
        color: #e2e8f0 !important;
        max-width: 34rem;
        line-height: 1.65;
        font-size: 1rem;
        margin-bottom: 1.35rem;
      }
      .falcon-badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.22);
        color: #fde047;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        backdrop-filter: blur(10px);
      }
      .falcon-feature-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin-top: 1rem;
      }
      .falcon-feature-card {
        padding: 0.95rem 1rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.14);
        backdrop-filter: blur(12px);
        transition: transform 0.22s ease, background 0.22s ease;
      }
      .falcon-feature-card:hover {
        transform: translateY(-3px);
        background: rgba(255, 255, 255, 0.14);
      }
      .falcon-feature-card.cyan { border-left: 4px solid #38bdf8; }
      .falcon-feature-card.emerald { border-left: 4px solid #34d399; }
      .falcon-feature-card.amber { border-left: 4px solid #fbbf24; }
      .falcon-feature-card.rose { border-left: 4px solid #fb7185; }
      .falcon-feature-card strong {
        color: #ffffff;
        font-size: 0.95rem;
        display: block;
        margin-bottom: 0.2rem;
      }
      .falcon-feature-card span {
        color: #cbd5e1;
        font-size: 0.82rem;
        line-height: 1.45;
      }

      /* 9. DASHBOARD HERO BANNERS & CLASSROOM CARDS */
      .falcon-hero-banner {
        position: relative;
        overflow: hidden;
        padding: 1.85rem 2.2rem;
        border-radius: 24px;
        color: #ffffff;
        margin-bottom: 1.2rem;
        background:
          radial-gradient(circle at 92% 15%, rgba(244, 63, 94, 0.42), transparent 36%),
          radial-gradient(circle at 70% 90%, rgba(16, 185, 129, 0.30), transparent 40%),
          linear-gradient(135deg, #0f172a 0%, #1e1b4b 48%, #3b0764 100%);
        box-shadow: 0 20px 45px rgba(30, 27, 75, 0.22);
      }
      .falcon-hero-banner h1 {
        color: #ffffff !important;
        margin: 0.35rem 0 0.45rem 0 !important;
        font-size: clamp(1.85rem, 2.8vw, 2.5rem) !important;
      }
      .falcon-hero-banner p {
        color: #e2e8f0 !important;
        margin: 0 !important;
        font-size: 1rem;
        max-width: 44rem;
      }
      .falcon-hero-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1rem;
      }
      .falcon-chip {
        padding: 0.32rem 0.8rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        background: rgba(255, 255, 255, 0.13);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #f8fafc;
      }

      .falcon-class-banner {
        padding: 1.2rem 1.25rem;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 0.9rem;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12);
      }
      .falcon-class-banner.dsa {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
      }
      .falcon-class-banner.os {
        background: linear-gradient(135deg, #059669 0%, #0d9488 55%, #0284c7 100%);
      }
      .falcon-class-banner.dbms {
        background: linear-gradient(135deg, #ea580c 0%, #e11d48 55%, #9333ea 100%);
      }
      .falcon-class-tag {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.22);
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        margin-bottom: 0.45rem;
      }
      .falcon-class-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.2rem;
        font-weight: 800;
        line-height: 1.3;
        color: #ffffff;
      }
      .falcon-topic-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.38rem;
        margin: 0.65rem 0 0.9rem;
      }
      .falcon-topic-pill {
        font-size: 0.74rem;
        font-weight: 700;
        padding: 0.22rem 0.6rem;
        border-radius: 8px;
        background: #f1f5f9;
        color: #334155;
        border: 1px solid #e2e8f0;
      }

      /* Sidebar User Profile Card */
      .falcon-sidebar-profile {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.28), rgba(236, 72, 153, 0.22));
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 14px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.5rem;
        color: #ffffff;
      }
      .falcon-sidebar-user {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.85rem;
        color: #e2e8f0;
      }
      .falcon-status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 8px #34d399;
        margin-right: 6px;
      }

      @keyframes falconFadeUp {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
      }

      @media (max-width: 900px) {
        .falcon-brand-panel { min-height: auto; padding: 1.5rem; margin-bottom: 1rem; }
        .falcon-feature-grid { grid-template-columns: 1fr; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# MongoDB setup and persistence
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_database(uri: str):
    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=10_000,
        connectTimeoutMS=10_000,
        appname="FalconStudentsPhase4",
    )
    client.admin.command("ping")
    database = client["ai_tutor_platform"]
    users = database["users"]
    plans = database["study_plans"]
    interviews = database["mock_interviews"]

    # Existing Phase 1 user documents remain compatible.
    interviews.create_index(
        [("user_email", ASCENDING), ("created_at", DESCENDING)],
        name="mock_interviews_by_user_date",
    )
    interviews.create_index(
        [("user_email", ASCENDING), ("session_id", ASCENDING)],
        unique=True,
        name="unique_mock_interview_per_user",
    )
    users.create_index(
        [("email", ASCENDING)],
        unique=True,
        collation=EMAIL_COLLATION,
        name="unique_email_case_insensitive",
    )
    plans.create_index([("user_email", ASCENDING), ("created_at", DESCENDING)])
    plans.create_index([("user_email", ASCENDING), ("plan_id", ASCENDING)], unique=True)
    return database


if not MONGO_URI:
    st.error("MongoDB is not configured. Add MONGO_URI to .streamlit/secrets.toml or an environment variable.")
    st.code(
        'MONGO_URI = "your-mongodb-connection-string"\n'
        'GEMINI_API_KEY = "your-gemini-api-key"\n'
        'SENDER_EMAIL = "your-sender@gmail.com"\n'
        'SENDER_PASSWORD = "your-gmail-app-password"\n'
        'GEMINI_MODEL = "gemini-3.5-flash-lite"',
        language="toml",
    )
    st.stop()

try:
    database = init_database(MONGO_URI)
    users_db = database["users"]
    plans_db = database["study_plans"]
    interviews_db = database["mock_interviews"]
except ConfigurationError as exc:
    LOGGER.exception("MongoDB configuration failed")
    st.error("MongoDB connection settings could not be parsed. Make sure you installed the updated requirements.txt (which includes PyMongo SRV/DNS support) and that MONGO_URI is copied correctly into .streamlit/secrets.toml.")
    st.stop()
except Exception as exc:
    LOGGER.error("MongoDB initialization failed: %s", type(exc).__name__)
    st.error("Could not connect to MongoDB. Check the Atlas database username/password, Network Access IP allowlist, database-user permissions, and whether existing email records conflict with the unique email index. Do not paste your connection URI into chat.")
    st.stop()

# -----------------------------------------------------------------------------
# Authentication helpers (retain Phase 1 features)
# -----------------------------------------------------------------------------
def normalize_email(email: str) -> str:
    return email.strip().lower()


def validate_email(email: str) -> bool:
    clean = normalize_email(email)
    return len(clean) <= 254 and bool(EMAIL_PATTERN.fullmatch(clean))


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(stored_hash: str, password: str) -> tuple[bool, bool]:
    """Return (matches, legacy_sha256_hash)."""
    if stored_hash.startswith("pbkdf2_sha256$"):
        try:
            _, rounds_text, salt_hex, digest_hex = stored_hash.split("$", 3)
            rounds = int(rounds_text)
            if not 100_000 <= rounds <= 2_000_000:
                return False, False
            actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), rounds)
            return hmac.compare_digest(actual, bytes.fromhex(digest_hex)), False
        except (ValueError, TypeError):
            return False, False
    if re.fullmatch(r"[0-9a-fA-F]{64}", stored_hash):
        legacy = hashlib.sha256(password.encode("utf-8")).hexdigest()
        return hmac.compare_digest(legacy.lower(), stored_hash.lower()), True
    return False, False


def find_user_by_email(email: str) -> dict[str, Any] | None:
    return users_db.find_one({"email": normalize_email(email)}, collation=EMAIL_COLLATION)


def create_user(name: str, email: str, password_hash: str) -> bool:
    email = normalize_email(email)
    if len(name.strip()) < 2 or not validate_email(email):
        return False
    now = datetime.now(timezone.utc)
    document = {
        "name": name.strip(),
        "email": email,
        "password": password_hash,
        "score": 0,
        "failed_login_count": 0,
        "login_lock_until_ts": 0,
        "chat_history": {subject: [] for subject in SUBJECTS},
        "quiz_history": [],
        "topic_stats": {},
        "topic_labels": {},
        "learning_style": LEARNING_STYLES[0],
        "student_level": LEVELS[0],
        "created_at": now,
        "updated_at": now,
    }
    try:
        users_db.insert_one(document)
        return True
    except DuplicateKeyError:
        return False


def authenticate_user(email: str, password: str) -> tuple[dict[str, Any] | None, str | None]:
    user = find_user_by_email(email)
    if not user:
        return None, "Email or password is incorrect."
    now = time.time()
    lock_until = float(user.get("login_lock_until_ts", 0) or 0)
    if lock_until > now:
        minutes = max(1, int((lock_until - now + 59) // 60))
        return None, f"Too many failed attempts. Try again in about {minutes} minute(s)."
    matches, legacy = verify_password(str(user.get("password", "")), password)
    if not matches:
        failures = int(user.get("failed_login_count", 0) or 0) + 1
        if failures >= LOGIN_MAX_FAILURES:
            users_db.update_one({"_id": user["_id"]}, {"$set": {"failed_login_count": 0, "login_lock_until_ts": now + LOGIN_LOCK_SECONDS}})
            return None, "Too many failed attempts. Try again in about 15 minutes."
        users_db.update_one({"_id": user["_id"]}, {"$set": {"failed_login_count": failures, "last_failed_login_ts": now}})
        return None, "Email or password is incorrect."
    updates: dict[str, Any] = {
        "failed_login_count": 0,
        "login_lock_until_ts": 0,
        "last_login_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    if legacy:
        updates["password"] = hash_password(password)
    users_db.update_one({"_id": user["_id"]}, {"$set": updates})
    user["email"] = normalize_email(user.get("email", email))
    return user, None


def update_password(email: str, password: str) -> bool:
    result = users_db.update_one(
        {"email": normalize_email(email)},
        {"$set": {
            "password": hash_password(password),
            "failed_login_count": 0,
            "login_lock_until_ts": 0,
            "updated_at": datetime.now(timezone.utc),
        }},
        collation=EMAIL_COLLATION,
    )
    return result.matched_count == 1


def save_preferences() -> None:
    email = st.session_state.get("user_email", "")
    if not email:
        return
    style = st.session_state.get("learning_style", LEARNING_STYLES[0])
    level = st.session_state.get("student_level", LEVELS[0])
    try:
        users_db.update_one(
            {"email": normalize_email(email)},
            {"$set": {"learning_style": style, "student_level": level, "updated_at": datetime.now(timezone.utc)}},
            collation=EMAIL_COLLATION,
        )
        st.session_state["preference_save_error"] = False
    except PyMongoError:
        LOGGER.warning("Could not save learning preferences")
        st.session_state["preference_save_error"] = True

# -----------------------------------------------------------------------------
# OTP delivery and state
# -----------------------------------------------------------------------------
def send_otp_email(receiver_email: str, otp: str, purpose: str = "signup") -> bool:
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        LOGGER.warning("SMTP is not configured")
        return False
    if purpose == "signup":
        subject = "Verify your Falcon Students account"
        body = f"Welcome to Falcon Students.\n\nYour verification code is: {otp}\n\nThis code expires in 5 minutes."
    else:
        subject = "Falcon Students password reset code"
        body = f"Your password reset code is: {otp}\n\nThis code expires in 5 minutes. If you did not request it, ignore this email."
    message = MIMEText(body, "plain", "utf-8")
    message["Subject"] = subject
    message["From"] = SENDER_EMAIL
    message["To"] = receiver_email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15, context=ssl.create_default_context()) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(message)
        return True
    except (smtplib.SMTPException, OSError, TimeoutError) as exc:
        LOGGER.warning("OTP email delivery failed: %s", type(exc).__name__)
        return False


def make_otp_state(email: str, otp: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    salt = secrets.token_hex(16)
    now = time.time()
    return {
        "email": normalize_email(email),
        "salt": salt,
        "digest": hashlib.sha256(f"{salt}:{otp}".encode()).hexdigest(),
        "created_at": now,
        "expires_at": now + OTP_TTL_SECONDS,
        "resend_after": now + OTP_RESEND_WAIT_SECONDS,
        "attempts_left": OTP_MAX_ATTEMPTS,
        "extra": extra or {},
        "verified": False,
    }


def request_otp(email: str, purpose: str, extra: dict[str, Any] | None = None) -> tuple[dict[str, Any], bool]:
    otp = f"{secrets.randbelow(1_000_000):06d}"
    state = make_otp_state(email, otp, extra)
    return state, send_otp_email(email, otp, purpose)


def verify_otp_state(state_key: str, entered: str) -> str:
    state = st.session_state.get(state_key)
    if not isinstance(state, dict):
        return "missing"
    if time.time() > float(state.get("expires_at", 0)):
        return "expired"
    if int(state.get("attempts_left", 0)) <= 0:
        return "locked"
    normalized = entered.strip()
    actual = hashlib.sha256(f"{state.get('salt', '')}:{normalized}".encode()).hexdigest()
    if re.fullmatch(r"\d{6}", normalized) and hmac.compare_digest(actual, str(state.get("digest", ""))):
        return "ok"
    updated = dict(state)
    updated["attempts_left"] = max(0, int(updated.get("attempts_left", 0)) - 1)
    st.session_state[state_key] = updated
    return "locked" if updated["attempts_left"] <= 0 else "invalid"


def resend_allowed(state: dict[str, Any]) -> tuple[bool, int]:
    seconds = max(0, int(float(state.get("resend_after", 0)) - time.time() + 0.999))
    return seconds == 0, seconds

# -----------------------------------------------------------------------------
# Gemini integration
# -----------------------------------------------------------------------------
modern_client = None
if GEMINI_API_KEY and modern_genai is not None:
    try:
        modern_client = modern_genai.Client(api_key=GEMINI_API_KEY)
    except Exception as exc:
        LOGGER.warning("Google GenAI client initialization failed: %s", type(exc).__name__)


def generate_ai_text(prompt: str, *, json_mode: bool = False) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("AI is not configured. Add GEMINI_API_KEY to .streamlit/secrets.toml or your environment variables.")
    try:
        if modern_client is not None:
            if json_mode:
                from google.genai import types
                response = modern_client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
            else:
                response = modern_client.models.generate_content(model=MODEL_NAME, contents=prompt)
            output = getattr(response, "text", None)
        else:
            raise RuntimeError("Install or update google-genai to enable AI features.")
        if not isinstance(output, str) or not output.strip():
            raise RuntimeError("The AI service returned an empty response. Try again.")
        return output.strip()
    except RuntimeError:
        raise
    except Exception as exc:
        LOGGER.exception("Gemini request failed")
        raise RuntimeError(
            "The AI service could not complete this request. "
            "Check the API key, model name, quota, and network connection."
        ) from None


def parse_json_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("The AI response was not valid JSON. Please retry.")
    payload = json.loads(text[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("The AI returned an unexpected response format. Please retry.")
    return payload


def parse_quiz_json(raw: str, subject: str, requested_topic: str) -> dict[str, Any]:
    payload = parse_json_response(raw)
    question = payload.get("question")
    options = payload.get("options")
    answer = str(payload.get("correct_answer", "")).strip().upper()
    if not isinstance(question, str) or not question.strip():
        raise ValueError("The generated quiz is missing its question. Generate another question.")
    if not isinstance(options, dict) or any(letter not in options for letter in "ABCD"):
        raise ValueError("The generated quiz did not include four options. Generate another question.")
    normalized_options = {letter: str(options[letter]).strip() for letter in "ABCD"}
    if any(not normalized_options[letter] for letter in "ABCD") or answer not in "ABCD":
        raise ValueError("The generated quiz options or answer key were invalid. Generate another question.")
    topic = requested_topic
    if requested_topic == "Mixed practice":
        model_topic = str(payload.get("topic", "General")).strip().casefold()
        matches = [item for item in SUBJECT_TOPICS[subject] if item.casefold() == model_topic]
        topic = matches[0] if matches else "General"
    return {
        "question": question.strip()[:3_000],
        "options": normalized_options,
        "correct_answer": answer,
        "explanation": str(payload.get("explanation", "")).strip()[:5_000],
        "topic": topic,
    }

# -----------------------------------------------------------------------------
# Phase 2: quiz history and topic-wise tracking
# -----------------------------------------------------------------------------
def topic_key(subject: str, topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")[:70] or "general"
    return f"{subject}__{slug}"


def save_quiz_attempt(email: str, attempt: dict[str, Any]) -> bool:
    """Atomically save an attempt and update topic stats/XP exactly once per attempt ID."""
    subject = attempt["subject"]
    topic = attempt["topic"]
    key = topic_key(subject, topic)
    record = {
        "attempt_id": attempt["attempt_id"],
        "subject": subject,
        "topic": topic,
        "difficulty": attempt["difficulty"],
        "question": attempt["question"][:3_000],
        "options": attempt["options"],
        "selected_answer": attempt["selected_answer"],
        "correct_answer": attempt["correct_answer"],
        "is_correct": bool(attempt["is_correct"]),
        "xp_awarded": 10 if attempt["is_correct"] else 0,
        "explanation": str(attempt.get("explanation", ""))[:5_000],
        "created_at": datetime.now(timezone.utc),
    }
    increments: dict[str, int] = {f"topic_stats.{key}.attempts": 1}
    if attempt["is_correct"]:
        increments["score"] = 10
        increments[f"topic_stats.{key}.correct"] = 1
    result = users_db.update_one(
        {
            "email": normalize_email(email),
            "quiz_history.attempt_id": {"$ne": attempt["attempt_id"]},
        },
        {
            "$push": {"quiz_history": {"$each": [record], "$slice": -MAX_QUIZ_HISTORY}},
            "$inc": increments,
            "$set": {
                f"topic_labels.{key}": topic,
                "updated_at": datetime.now(timezone.utc),
            },
        },
        collation=EMAIL_COLLATION,
    )
    return result.modified_count == 1


def get_user_history(email: str) -> list[dict[str, Any]]:
    user = find_user_by_email(email) or {}
    raw = user.get("quiz_history", [])
    return raw if isinstance(raw, list) else []


def get_topic_analytics(user: dict[str, Any]) -> list[dict[str, Any]]:
    stats = user.get("topic_stats", {}) or {}
    labels = user.get("topic_labels", {}) or {}
    rows = []
    if not isinstance(stats, dict):
        return rows
    for key, values in stats.items():
        if not isinstance(values, dict):
            continue
        subject = key.split("__", 1)[0] if "__" in key else "Other"
        attempts = int(values.get("attempts", 0) or 0)
        correct = int(values.get("correct", 0) or 0)
        rows.append({
            "Subject": subject,
            "Topic": labels.get(key, key.split("__", 1)[-1].replace("_", " ").title()),
            "Attempts": attempts,
            "Correct": correct,
            "Accuracy (%)": round((correct / attempts) * 100, 1) if attempts else 0.0,
        })
    return sorted(rows, key=lambda row: (row["Subject"], row["Topic"]))


def compute_recommendation(topic_rows: list[dict[str, Any]]) -> str:
    practiced = [row for row in topic_rows if row["Attempts"] >= 2]
    weak = [row for row in practiced if row["Accuracy (%)"] < 60]
    if weak:
        item = sorted(weak, key=lambda row: (row["Accuracy (%)"], -row["Attempts"]))[0]
        return f"Revise **{item['Topic']}** in {item['Subject']}. Accuracy is {item['Accuracy (%)']}% across {item['Attempts']} attempts. Review the concept in the AI Tutor, then try another quiz."
    if practiced:
        strong = max(practiced, key=lambda row: row["Accuracy (%)"])
        return f"Good progress in **{strong['Topic']}** ({strong['Accuracy (%)']}% accuracy). Continue with a new topic or try a higher difficulty level."
    return "Complete at least two quizzes in a topic to unlock a useful topic-level recommendation."


# -----------------------------------------------------------------------------
# Phase 3: adaptive revision, learning streaks, achievements, and export
# -----------------------------------------------------------------------------
APP_TIMEZONE = ZoneInfo("Asia/Kolkata")


def attempt_local_date(value: Any) -> date | None:
    """Convert an attempt timestamp to the app's local calendar date."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(APP_TIMEZONE).date()


def get_learning_streak(history: list[dict[str, Any]]) -> tuple[int, int]:
    """Return current quiz-activity streak and total distinct active days."""
    active_days = {
        day for item in history
        if isinstance(item, dict)
        for day in [attempt_local_date(item.get("created_at"))]
        if day is not None
    }
    if not active_days:
        return 0, 0
    today = datetime.now(APP_TIMEZONE).date()
    cursor = today if today in active_days else today - timedelta(days=1)
    if cursor not in active_days:
        return 0, len(active_days)
    streak = 0
    while cursor in active_days:
        streak += 1
        cursor -= timedelta(days=1)
    return streak, len(active_days)


def choose_revision_topic(topic_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose a weak topic first, otherwise the least-accurate practised topic."""
    practiced = [row for row in topic_rows if int(row.get("Attempts", 0)) > 0]
    if not practiced:
        return None
    weak = [row for row in practiced if int(row.get("Attempts", 0)) >= 2 and float(row.get("Accuracy (%)", 0)) < 60]
    pool = weak or practiced
    return min(pool, key=lambda row: (float(row.get("Accuracy (%)", 0)), -int(row.get("Attempts", 0))))


def get_achievements(history: list[dict[str, Any]], score: int, streak: int) -> list[dict[str, Any]]:
    total = len(history)
    correct = sum(1 for item in history if isinstance(item, dict) and item.get("is_correct"))
    accuracy = (correct / total * 100) if total else 0.0
    definitions = [
        ("First Steps", "Complete your first quiz", total >= 1),
        ("Practice Builder", "Complete 5 quiz attempts", total >= 5),
        ("Streak Starter", "Practise on 3 consecutive days", streak >= 3),
        ("Accuracy Ace", "Reach 80% accuracy over at least 5 attempts", total >= 5 and accuracy >= 80),
        ("Quiz Master", "Answer 25 questions correctly", correct >= 25),
        ("Century Club", "Earn 100 XP", score >= 100),
    ]
    return [{"name": name, "description": description, "unlocked": unlocked} for name, description, unlocked in definitions]


def build_quiz_export(history: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in history:
        if not isinstance(item, dict):
            continue
        stamp = item.get("created_at")
        local_day = attempt_local_date(stamp)
        rows.append({
            "Date": local_day.isoformat() if local_day else "",
            "Subject": item.get("subject", ""),
            "Topic": item.get("topic", ""),
            "Difficulty": item.get("difficulty", ""),
            "Question": item.get("question", ""),
            "Your answer": item.get("selected_answer", ""),
            "Correct answer": item.get("correct_answer", ""),
            "Result": "Correct" if item.get("is_correct") else "Incorrect",
            "XP": int(item.get("xp_awarded", 0) or 0),
            "Explanation": item.get("explanation", ""),
        })
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# Phase 4: saved mock interview practice
# -----------------------------------------------------------------------------
INTERVIEW_SUBJECTS = ("DSA", "OS", "DBMS", "Mixed CS")
INTERVIEW_FOCI = ("Core concepts", "Problem-solving", "Mixed technical")
INTERVIEW_QUESTION_COUNT = 5


def parse_interview_questions(raw: str, subject: str) -> list[dict[str, Any]]:
    """Validate and normalise generated mock-interview questions."""
    payload = parse_json_response(raw)
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list) or len(raw_questions) < INTERVIEW_QUESTION_COUNT:
        raise ValueError(
            f"The AI did not return {INTERVIEW_QUESTION_COUNT} interview questions. Please try again."
        )

    allowed_topics = (
        [topic for values in SUBJECT_TOPICS.values() for topic in values]
        if subject == "Mixed CS"
        else SUBJECT_TOPICS.get(subject, [])
    )
    questions: list[dict[str, Any]] = []
    for item in raw_questions[:INTERVIEW_QUESTION_COUNT]:
        if isinstance(item, str):
            question_text = item.strip()
            topic = "General"
        elif isinstance(item, dict):
            question_text = str(item.get("question", "")).strip()
            topic = str(item.get("topic", "General")).strip() or "General"
        else:
            continue

        if not question_text:
            continue
        if allowed_topics and topic.casefold() not in {t.casefold() for t in allowed_topics}:
            topic = "General"

        questions.append({
            "question_id": uuid.uuid4().hex,
            "question": question_text[:2_000],
            "topic": topic[:100],
            "answer": "",
            "evaluation": None,
        })

    if len(questions) != INTERVIEW_QUESTION_COUNT:
        raise ValueError("Some interview questions were incomplete. Please generate the session again.")
    return questions


def parse_answer_evaluation(raw: str) -> dict[str, Any]:
    """Validate AI feedback before displaying or saving it."""
    payload = parse_json_response(raw)
    raw_score = payload.get("score", 0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        raise ValueError("The AI returned an invalid score. Please evaluate again.") from None

    if not 0 <= score <= 10:
        raise ValueError("The AI returned a score outside the expected 0–10 range.")

    def clean_list(name: str) -> list[str]:
        values = payload.get(name, [])
        if not isinstance(values, list):
            return []
        return [str(value).strip()[:500] for value in values[:5] if str(value).strip()]

    feedback = str(payload.get("feedback", "")).strip()[:2_000]
    model_answer = str(payload.get("model_answer", "")).strip()[:5_000]
    if not feedback:
        feedback = "Review the suggested improvements and try explaining the concept again."

    return {
        "score": round(score, 1),
        "feedback": feedback,
        "strengths": clean_list("strengths"),
        "improvements": clean_list("improvements"),
        "model_answer": model_answer,
    }


def get_mock_interviews(email: str, limit: int = 10) -> list[dict[str, Any]]:
    return list(
        interviews_db.find({"user_email": normalize_email(email)})
        .sort("created_at", DESCENDING)
        .limit(limit)
    )


def get_mock_interview(email: str, session_id: str) -> dict[str, Any] | None:
    return interviews_db.find_one({
        "user_email": normalize_email(email),
        "session_id": session_id,
    })


def save_mock_interview(
    email: str,
    subject: str,
    difficulty: str,
    focus: str,
    questions: list[dict[str, Any]],
) -> str:
    session_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    interviews_db.insert_one({
        "session_id": session_id,
        "user_email": normalize_email(email),
        "subject": subject,
        "difficulty": difficulty,
        "focus": focus,
        "questions": questions,
        "status": "in_progress",
        "created_at": now,
        "updated_at": now,
        "completed_at": None,
    })
    return session_id


def evaluate_mock_interview_answer(
    email: str,
    session_id: str,
    question_index: int,
    answer: str,
    question: dict[str, Any],
    subject: str,
    difficulty: str,
) -> dict[str, Any]:
    """Evaluate one answer and persist the result for this user's own session."""
    clean_answer = answer.strip()[:3_000]
    if len(clean_answer) < 2:
        raise ValueError("Write an answer before requesting feedback.")

    prompt = f"""You are a fair technical interviewer and a learning coach.
Subject: {subject}
Difficulty: {difficulty}
Question topic: {question.get("topic", "General")}
Interview question: {question.get("question", "")}
Candidate answer: {clean_answer}
Evaluate understanding rather than writing style. Be fair to a beginner when difficulty is Beginner.
Return only JSON with this schema:
{{"score":7,"feedback":"Brief overall feedback","strengths":["specific strength"],"improvements":["specific improvement"],"model_answer":"A clear answer the candidate could give in an interview"}}
Rules: score must be a number from 0 to 10; strengths and improvements must be arrays of short strings; do not invent what the candidate said; the model answer should be accurate and appropriate for the stated level."""

    evaluation = parse_answer_evaluation(generate_ai_text(prompt, json_mode=True))
    now = datetime.now(timezone.utc)
    field_prefix = f"questions.{question_index}"

    # Only update a session owned by the currently authenticated account.
    session = get_mock_interview(email, session_id)
    if not session:
        raise ValueError("This interview session could not be found for your account.")

    questions = session.get("questions", [])
    if not isinstance(questions, list) or not 0 <= question_index < len(questions):
        raise ValueError("The selected interview question is unavailable. Refresh and retry.")

    questions[question_index]["answer"] = clean_answer
    questions[question_index]["evaluation"] = evaluation
    complete = all(
        isinstance(item, dict)
        and isinstance(item.get("evaluation"), dict)
        and bool(str(item.get("answer", "")).strip())
        for item in questions
    )

    updates: dict[str, Any] = {
        f"{field_prefix}.answer": clean_answer,
        f"{field_prefix}.evaluation": evaluation,
        "updated_at": now,
        "status": "completed" if complete else "in_progress",
    }
    if complete:
        updates["completed_at"] = now

    result = interviews_db.update_one(
        {"user_email": normalize_email(email), "session_id": session_id},
        {"$set": updates},
    )
    if result.matched_count != 1:
        raise ValueError("The interview session could not be saved. Please reopen it and retry.")
    return evaluation


# -----------------------------------------------------------------------------
# Phase 2: saved study plans and task progress
# -----------------------------------------------------------------------------
def parse_study_plan(raw: str, goal: str, timeline: str) -> dict[str, Any]:
    payload = parse_json_response(raw)
    title = str(payload.get("title", goal)).strip()[:120] or goal[:120]
    days = payload.get("days")
    if not isinstance(days, list) or not days:
        raise ValueError("The generated study plan did not include day-by-day tasks. Please generate it again.")
    expected_days = {"1 Week": 7, "2 Weeks": 14, "1 Month": 30}[timeline]
    if len(days) < expected_days:
        raise ValueError(
            f"The AI returned {len(days)} days instead of {expected_days}. Please generate the plan again."
        )
    cleaned_days = []
    for index, day in enumerate(days[:expected_days], start=1):
        if not isinstance(day, dict):
            continue
        focus = str(day.get("focus", f"Study day {index}")).strip()[:160] or f"Study day {index}"
        raw_tasks = day.get("tasks", [])
        if not isinstance(raw_tasks, list):
            raw_tasks = []
        tasks = []
        for task in raw_tasks[:3]:
            task_text = str(task).strip()[:280]
            if task_text:
                tasks.append({"task_id": uuid.uuid4().hex, "text": task_text, "completed": False})
        if not tasks:
            tasks = [{"task_id": uuid.uuid4().hex, "text": f"Review: {focus}", "completed": False}]
        cleaned_days.append({"day": index, "focus": focus, "tasks": tasks})
    if not cleaned_days:
        raise ValueError("The generated study plan was empty. Please try again.")
    return {"title": title, "days": cleaned_days}


def create_study_plan(email: str, goal: str, timeline: str, plan_data: dict[str, Any]) -> str:
    plan_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    plans_db.insert_one({
        "plan_id": plan_id,
        "user_email": normalize_email(email),
        "title": plan_data["title"],
        "goal": goal.strip(),
        "timeline": timeline,
        "days": plan_data["days"],
        "created_at": now,
        "updated_at": now,
    })
    return plan_id


def get_study_plans(email: str, limit: int = 10) -> list[dict[str, Any]]:
    return list(plans_db.find({"user_email": normalize_email(email)}).sort("created_at", DESCENDING).limit(limit))


def toggle_study_task(email: str, plan_id: str, task_id: str, completed: bool) -> bool:
    plan = plans_db.find_one({"user_email": normalize_email(email), "plan_id": plan_id})
    if not plan:
        return False
    for day_index, day in enumerate(plan.get("days", [])):
        for task_index, task in enumerate(day.get("tasks", [])):
            if task.get("task_id") == task_id:
                field = f"days.{day_index}.tasks.{task_index}.completed"
                result = plans_db.update_one(
                    {"user_email": normalize_email(email), "plan_id": plan_id},
                    {"$set": {field: bool(completed), "updated_at": datetime.now(timezone.utc)}},
                )
                return result.modified_count == 1 or result.matched_count == 1
    return False


def get_plan_progress(plan: dict[str, Any]) -> tuple[int, int, float]:
    total = 0
    done = 0
    for day in plan.get("days", []):
        for task in day.get("tasks", []):
            total += 1
            done += int(bool(task.get("completed")))
    return done, total, (done / total * 100) if total else 0.0

# -----------------------------------------------------------------------------
# Chat history and session helpers
# -----------------------------------------------------------------------------
def get_chat_history(user: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    raw_history = user.get("chat_history") or {}
    output: dict[str, list[dict[str, str]]] = {}
    for subject in SUBJECTS:
        messages = raw_history.get(subject, []) if isinstance(raw_history, dict) else []
        clean = []
        if isinstance(messages, list):
            for item in messages[-MAX_CHAT_MESSAGES_PER_SUBJECT:]:
                if isinstance(item, dict) and item.get("role") in ("user", "assistant") and isinstance(item.get("content"), str):
                    clean.append({"role": item["role"], "content": item["content"][:30_000]})
        output[subject] = clean
    return output


def save_chat_message(email: str, subject: str, message: dict[str, str]) -> None:
    if subject not in SUBJECTS:
        raise ValueError("Unknown subject")
    if message.get("role") not in ("user", "assistant"):
        raise ValueError("Invalid message role")
    content = str(message.get("content", ""))[:30_000]
    users_db.update_one(
        {"email": normalize_email(email)},
        {
            "$push": {f"chat_history.{subject}": {"$each": [{"role": message["role"], "content": content, "created_at": datetime.now(timezone.utc)}], "$slice": -MAX_CHAT_MESSAGES_PER_SUBJECT}},
            "$set": {"updated_at": datetime.now(timezone.utc)},
        },
        collation=EMAIL_COLLATION,
    )


def load_logged_in_user(user: dict[str, Any]) -> None:
    st.session_state["logged_in"] = True
    st.session_state["user_name"] = str(user.get("name", "Student"))
    st.session_state["user_email"] = normalize_email(str(user.get("email", "")))
    st.session_state["learning_style"] = user.get("learning_style") if user.get("learning_style") in LEARNING_STYLES else LEARNING_STYLES[0]
    st.session_state["student_level"] = user.get("student_level") if user.get("student_level") in LEVELS else LEVELS[0]
    chats = get_chat_history(user)
    for subject in SUBJECTS:
        st.session_state[f"messages_{subject}"] = chats[subject]


def logout() -> None:
    st.session_state.clear()
    st.session_state["_app_session_version"] = APP_SESSION_VERSION
    try:
        st.query_params.clear()
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Session state migration: URL parameters are never treated as authentication.
# -----------------------------------------------------------------------------
if "user_email" in st.query_params:
    try:
        del st.query_params["user_email"]
    except Exception:
        pass

if st.session_state.get("_app_session_version") != APP_SESSION_VERSION:
    st.session_state.clear()
    st.session_state["_app_session_version"] = APP_SESSION_VERSION

DEFAULT_STATE: dict[str, Any] = {
    "logged_in": False,
    "user_name": "",
    "user_email": "",
    "current_page": "Home",
    "auth_view": "Login",
    "signup_otp_state": None,
    "forgot_otp_state": None,
    "current_quiz": None,
    "quiz_answer_key": "Select an answer",
    "roadmap_notice": "",
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value
for subject in SUBJECTS:
    st.session_state.setdefault(f"messages_{subject}", [])

# -----------------------------------------------------------------------------
# Authentication UI
# -----------------------------------------------------------------------------
def render_auth_screen() -> None:
    left, right = st.columns([1.08, 0.92], gap="large", vertical_alignment="center")
    with left:
        st.markdown(
            """
            <div class="falcon-brand-panel">
              <div class="falcon-badge-pill">🚀 FALCON STUDENTS · AI CS PLATFORM</div>
              <h1>Master Computer Science.<br><span class="accent">Build Real Confidence.</span></h1>
              <p class="tagline">Your all-in-one interactive studio for Data Structures, Operating Systems, DBMS, adaptive quizzes, and AI mock interviews.</p>
              <div class="falcon-feature-grid">
                <div class="falcon-feature-card cyan">
                  <strong>🤖 24/7 AI Subject Tutor</strong>
                  <span>Context-aware explanations tailored to your level & style for DSA, OS & DBMS.</span>
                </div>
                <div class="falcon-feature-card emerald">
                  <strong>⚡ Adaptive XP Quizzes</strong>
                  <span>Instant MCQ generation, topic-level accuracy analytics, and daily streaks.</span>
                </div>
                <div class="falcon-feature-card amber">
                  <strong>🗺️ Smart Study Roadmaps</strong>
                  <span>Day-by-day personalized plans with interactive MongoDB task tracking.</span>
                </div>
                <div class="falcon-feature-card rose">
                  <strong>🎤 AI Mock Interviews</strong>
                  <span>Practise technical rounds with 0–10 scoring, strengths, and model answers.</span>
                </div>
              </div>
              <div class="falcon-hero-chips" style="margin-top:1.35rem;">
                <span class="falcon-chip">✨ Gemini AI Powered</span>
                <span class="falcon-chip">🔐 OTP Verified Security</span>
                <span class="falcon-chip">📊 Live Analytics</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div style="margin-bottom:0.6rem;">
              <span style="display:inline-block;padding:0.28rem 0.75rem;border-radius:999px;background:linear-gradient(90deg,#ede9fe,#fce7f3);color:#6d28d9;font-weight:800;font-size:0.75rem;letter-spacing:0.08em;">STUDENT PORTAL</span>
              <h2 style="margin:0.4rem 0 0.2rem 0;">Welcome to Falcon Students</h2>
              <p style="color:#475569;margin:0 0 0.6rem 0;font-size:0.95rem;">Sign in or create your verified student account to continue.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.radio("Account action", ("Login", "Sign Up", "Forgot Password"), horizontal=True, key="auth_view", label_visibility="collapsed")
        view = st.session_state["auth_view"]

        if view == "Login":
            st.subheader("🔑 Login to Your Workspace")
            with st.form("login_form"):
                email = st.text_input("Email", max_chars=254, key="login_email")
                password = st.text_input("Password", type="password", max_chars=128, key="login_password")
                submitted = st.form_submit_button("Login to Falcon Students →", type="primary", use_container_width=True)
            if submitted:
                if not email.strip() or not password:
                    st.error("Enter both your email and password.")
                elif not validate_email(email):
                    st.error("Enter a valid email address.")
                else:
                    try:
                        user, error_message = authenticate_user(email, password)
                        if user:
                            load_logged_in_user(user)
                            st.rerun()
                        st.error(error_message or "Email or password is incorrect.")
                    except PyMongoError:
                        st.error("Login is temporarily unavailable. Check the database connection.")

        elif view == "Sign Up":
            state = st.session_state.get("signup_otp_state")
            if not isinstance(state, dict):
                st.subheader("✨ Create a New Account")
                with st.form("signup_form"):
                    name = st.text_input("Full name", max_chars=100)
                    email = st.text_input("Email", max_chars=254, key="signup_email")
                    password = st.text_input("Password (at least 8 characters)", type="password", max_chars=128)
                    confirm = st.text_input("Confirm password", type="password", max_chars=128)
                    submitted = st.form_submit_button("Create account and send OTP →", type="primary", use_container_width=True)
                if submitted:
                    clean_name, clean_email = name.strip(), normalize_email(email)
                    if len(clean_name) < 2:
                        st.error("Enter a name containing at least 2 characters.")
                    elif not validate_email(clean_email):
                        st.error("Enter a valid email address.")
                    elif len(password) < 8:
                        st.error("Use a password with at least 8 characters.")
                    elif password != confirm:
                        st.error("The passwords do not match.")
                    else:
                        try:
                            if find_user_by_email(clean_email):
                                st.error("An account with this email already exists. Try logging in.")
                            else:
                                extra = {"name": clean_name, "password_hash": hash_password(password)}
                                otp_state, sent = request_otp(clean_email, "signup", extra)
                                if sent:
                                    st.session_state["signup_otp_state"] = otp_state
                                    st.success("Verification code sent. Check your inbox and spam folder.")
                                    st.rerun()
                                else:
                                    st.error("Verification email could not be sent. Check SMTP settings. No account has been created.")
                        except PyMongoError:
                            st.error("Registration is temporarily unavailable. Check the database connection.")
            else:
                st.subheader("📧 Verify your email")
                st.info(f"Enter the 6-digit code sent to {state['email']}. It expires in 5 minutes.")
                st.caption(f"Attempts remaining: {state.get('attempts_left', 0)}")
                with st.form("signup_verify_form"):
                    entered = st.text_input("6-digit verification code", max_chars=6)
                    submitted = st.form_submit_button("Verify and create account", type="primary", use_container_width=True)
                if submitted:
                    status = verify_otp_state("signup_otp_state", entered)
                    current = st.session_state.get("signup_otp_state") or state
                    if status == "ok":
                        details = current.get("extra", {})
                        try:
                            created = create_user(details.get("name", ""), current["email"], details.get("password_hash", ""))
                            if created:
                                st.session_state["signup_otp_state"] = None
                                st.success("Account created. Select Login to continue.")
                            else:
                                st.error("An account with this email already exists. Select Login instead.")
                                st.session_state["signup_otp_state"] = None
                        except PyMongoError:
                            st.error("Database could not save the account. Please try again.")
                    elif status == "expired":
                        st.error("This code has expired. Request a new code below.")
                    elif status in ("locked", "missing"):
                        st.error("The code is unavailable or too many attempts were made. Request a new code.")
                    else:
                        st.error("That code is incorrect.")
                current = st.session_state.get("signup_otp_state")
                if isinstance(current, dict):
                    if st.button("Resend verification code", key="resend_signup_otp", use_container_width=True):
                        allowed, seconds = resend_allowed(current)
                        if not allowed:
                            st.warning(f"Wait {seconds} second(s) before requesting another code.")
                        else:
                            new_state, sent = request_otp(current["email"], "signup", current.get("extra", {}))
                            if sent:
                                st.session_state["signup_otp_state"] = new_state
                                st.success("A new verification code was sent.")
                                st.rerun()
                            else:
                                st.error("The verification email could not be sent. Check SMTP settings.")
                    if st.button("Cancel sign up", key="cancel_signup", use_container_width=True):
                        st.session_state["signup_otp_state"] = None
                        st.rerun()

        else:  # Forgot Password
            state = st.session_state.get("forgot_otp_state")
            if not isinstance(state, dict):
                st.subheader("🛡️ Reset password")
                with st.form("forgot_request_form"):
                    email = st.text_input("Registered email", max_chars=254)
                    submitted = st.form_submit_button("Send reset code →", type="primary", use_container_width=True)
                if submitted:
                    clean_email = normalize_email(email)
                    if not validate_email(clean_email):
                        st.error("Enter a valid email address.")
                    else:
                        try:
                            exists = find_user_by_email(clean_email) is not None
                            if exists:
                                state, sent = request_otp(clean_email, "reset", {"account_exists": True})
                            else:
                                fake = f"{secrets.randbelow(1_000_000):06d}"
                                state, sent = make_otp_state(clean_email, fake, {"account_exists": False}), True
                            if exists and not sent:
                                st.error("Email service is unavailable. Check SMTP settings.")
                            else:
                                st.session_state["forgot_otp_state"] = state
                                st.info("If an account is registered with this address, a reset code has been sent. Check your inbox and spam folder.")
                                st.rerun()
                        except PyMongoError:
                            st.error("Password reset is temporarily unavailable. Check the database connection.")
            elif not state.get("verified") or time.time() > float(state.get("expires_at", 0)):
                st.subheader("Verify reset code")
                st.info(f"If registered, a code was sent to {state['email']}. It expires in 5 minutes.")
                st.caption(f"Attempts remaining: {state.get('attempts_left', 0)}")
                with st.form("forgot_verify_form"):
                    entered = st.text_input("6-digit reset code", max_chars=6)
                    submitted = st.form_submit_button("Verify code", type="primary", use_container_width=True)
                if submitted:
                    status = verify_otp_state("forgot_otp_state", entered)
                    current = st.session_state.get("forgot_otp_state")
                    if status == "ok" and isinstance(current, dict) and current.get("extra", {}).get("account_exists"):
                        current = dict(current)
                        current["verified"] = True
                        st.session_state["forgot_otp_state"] = current
                        st.rerun()
                    elif status == "expired":
                        st.error("This code has expired. Request another code below.")
                    elif status in ("locked", "missing"):
                        st.error("The code is unavailable or too many attempts were made. Request another code.")
                    else:
                        st.error("The code is incorrect or the account could not be verified.")
                current = st.session_state.get("forgot_otp_state")
                if isinstance(current, dict) and st.button("Resend reset code", key="resend_forgot_otp", use_container_width=True):
                    allowed, seconds = resend_allowed(current)
                    if not allowed:
                        st.warning(f"Wait {seconds} second(s) before requesting another code.")
                    elif current.get("extra", {}).get("account_exists"):
                        new_state, sent = request_otp(current["email"], "reset", {"account_exists": True})
                        if sent:
                            st.session_state["forgot_otp_state"] = new_state
                            st.success("If the account is registered, a new code has been sent.")
                            st.rerun()
                        else:
                            st.error("The verification email could not be sent. Check SMTP settings.")
                    else:
                        fake = f"{secrets.randbelow(1_000_000):06d}"
                        st.session_state["forgot_otp_state"] = make_otp_state(current["email"], fake, {"account_exists": False})
                        st.info("If the account is registered, a new code has been sent.")
                        st.rerun()
            else:
                st.subheader("Choose a new password")
                with st.form("reset_password_form"):
                    new_password = st.text_input("New password (at least 8 characters)", type="password", max_chars=128)
                    confirm = st.text_input("Confirm new password", type="password", max_chars=128)
                    submitted = st.form_submit_button("Update password", type="primary", use_container_width=True)
                if submitted:
                    if len(new_password) < 8:
                        st.error("Use a password with at least 8 characters.")
                    elif new_password != confirm:
                        st.error("The passwords do not match.")
                    else:
                        try:
                            if update_password(state["email"], new_password):
                                st.session_state["forgot_otp_state"] = None
                                st.success("Password updated. Select Login to sign in with your new password.")
                            else:
                                st.error("The account could not be updated. Restart the reset process.")
                        except PyMongoError:
                            st.error("Database could not update your password. Please try again.")
            if st.session_state.get("forgot_otp_state") is not None and st.button("Cancel password reset", key="cancel_reset", use_container_width=True):
                st.session_state["forgot_otp_state"] = None
                st.rerun()

# -----------------------------------------------------------------------------
# Sidebar and home
# -----------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        user_name = st.session_state.get("user_name", "Student")
        level_badge = st.session_state.get("student_level", LEVELS[0])
        st.markdown(
            f"""
            <div class="falcon-sidebar-profile">
              <div class="falcon-sidebar-user">
                <span><span class="falcon-status-dot"></span><strong>{user_name}</strong></span>
                <span style="background:rgba(255,255,255,0.16);padding:2px 8px;border-radius:99px;font-size:0.72rem;font-weight:700;">{level_badge}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("**🧭 NAVIGATION**")
        navigation = [
            ("🏠  Learn Hub", "Home"),
            ("⚡  Quiz Arena", "Quiz"),
            ("📊  Performance", "Performance"),
            ("🧠  Revision Center", "Revision"),
            ("🎤  Mock Interview", "MockInterview"),
            ("🗺️  Study Plan", "StudyPlan"),
        ]
        current_pg = st.session_state.get("current_page", "Home")
        for label, page in navigation:
            is_active = (current_pg == page) or (page == "Home" and str(current_pg).startswith("Classroom_"))
            if st.button(label, key=f"nav_{page}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state["current_page"] = page
                st.rerun()
        st.divider()
        st.markdown("**⚙️ LEARNING PREFERENCES**")
        st.selectbox("Preferred style", LEARNING_STYLES, key="learning_style", on_change=save_preferences)
        st.select_slider("Current level", LEVELS, key="student_level", on_change=save_preferences)
        st.caption("Preferences sync automatically to your profile.")
        st.divider()
        if st.button("🚪 Logout", type="primary", use_container_width=True):
            logout()
            st.rerun()


def render_home() -> None:
    email = st.session_state["user_email"]
    first_name = st.session_state["user_name"].split()[0]
    style_now = st.session_state.get("learning_style", LEARNING_STYLES[0])
    level_now = st.session_state.get("student_level", LEVELS[0])

    st.markdown(
        f"""
        <div class="falcon-hero-banner">
          <span class="falcon-badge-pill">✨ YOUR AI LEARNING COMMAND CENTER</span>
          <h1>Welcome back, {first_name}! 👋</h1>
          <p>Master core Computer Science concepts, test yourself with adaptive quizzes, and track your topic-by-topic accuracy in real time.</p>
          <div class="falcon-hero-chips">
            <span class="falcon-chip">🎯 Level: {level_now}</span>
            <span class="falcon-chip">💡 Style: {style_now}</span>
            <span class="falcon-chip">🔥 Subjects: DSA · OS · DBMS</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        user = find_user_by_email(email) or {}
        topic_rows = get_topic_analytics(user)
        history = get_user_history(email)
        score = int(user.get("score", 0) or 0)
        attempts = sum(int(row["Attempts"]) for row in topic_rows)
        correct = sum(int(row["Correct"]) for row in topic_rows)
        accuracy = (correct / attempts * 100) if attempts else 0.0
        c1, c2, c3 = st.columns(3)
        c1.metric("🏆 Total XP Earned", f"{score} XP")
        c2.metric("📝 Quiz Attempts", attempts)
        c3.metric("🎯 Overall Accuracy", f"{accuracy:.1f}%" if attempts else "—")
        with st.container(border=True):
            st.markdown("#### 💡 Recommended Next Step")
            st.markdown(compute_recommendation(topic_rows))
            weak_topics = [row for row in topic_rows if row["Attempts"] >= 2 and row["Accuracy (%)"] < 60]
            if weak_topics:
                best = sorted(weak_topics, key=lambda row: row["Accuracy (%)"])[0]
                if st.button(f"🚀 Open {best['Subject']} tutor for {best['Topic']}", type="primary"):
                    st.session_state["current_page"] = f"Classroom_{best['Subject']}"
                    st.session_state["prefill_tutor_topic"] = best["Topic"]
                    st.rerun()
    except PyMongoError:
        st.warning("Learning summary could not be loaded right now.")

    st.subheader("🚀 Your Interactive Classrooms")
    columns = st.columns(3, gap="medium")
    descriptions = {
        "DSA": (
            "Data Structures & Algorithms",
            "Arrays, sorting, searching, trees, graphs, and algorithmic problem-solving.",
            "dsa",
            "🧩",
            ["Arrays", "Trees", "Graphs", "DP"],
        ),
        "OS": (
            "Operating Systems",
            "Process management, CPU scheduling, deadlocks, virtual memory, and concurrency.",
            "os",
            "⚙️",
            ["Processes", "Scheduling", "Deadlocks", "Memory"],
        ),
        "DBMS": (
            "Database Management Systems",
            "Relational models, complex SQL queries, joins, normalization, and ACID transactions.",
            "dbms",
            "🗄️",
            ["SQL Queries", "Joins", "Normalization", "Indexing"],
        ),
    }
    for col, subject in zip(columns, SUBJECTS):
        title, description, css_class, icon, sample_topics = descriptions[subject]
        pills_html = "".join(f'<span class="falcon-topic-pill">{t}</span>' for t in sample_topics)
        with col:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="falcon-class-banner {css_class}">
                      <div class="falcon-class-tag">{icon} {subject} CLASSROOM</div>
                      <div class="falcon-class-title">{title}</div>
                    </div>
                    <p style="color:#475569;font-size:0.92rem;min-height:2.8rem;margin-bottom:0.4rem;">{description}</p>
                    <div class="falcon-topic-pills">{pills_html}</div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Enter {subject} classroom →", key=f"open_{subject}", use_container_width=True, type="primary"):
                    st.session_state["current_page"] = f"Classroom_{subject}"
                    st.rerun()

# -----------------------------------------------------------------------------
# Quiz screen: question generation, secure scoring and persisted topic analytics
# -----------------------------------------------------------------------------
def render_quiz() -> None:
    st.markdown(
        """
        <div class="falcon-hero-banner" style="background:linear-gradient(135deg,#311042 0%,#4f46e5 55%,#06b6d4 100%);">
          <span class="falcon-badge-pill">⚡ ADAPTIVE ASSESSMENT</span>
          <h1>Quiz Practice Arena</h1>
          <p>Generate subject-specific MCQs tailored to your level. Every correct answer earns 10 XP and updates your live mastery profile.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_s, col_t = st.columns(2)
    with col_s:
        subject = st.selectbox("Subject", SUBJECTS, key="quiz_subject")
    topic_choices = ["Mixed practice"] + SUBJECT_TOPICS[subject]
    with col_t:
        topic = st.selectbox("Topic", topic_choices, key="quiz_topic")
    level = st.session_state.get("student_level", LEVELS[0])
    if st.button("✨ Generate question", type="primary"):
        topic_instruction = f"The topic must be exactly: {topic}." if topic != "Mixed practice" else f"Choose topic from this list only: {', '.join(SUBJECT_TOPICS[subject])}."
        prompt = f"""Create one high-quality multiple-choice question for a Computer Science student.
Subject: {subject}
Difficulty: {level}
{topic_instruction}
Return only valid JSON with keys exactly:
{{"topic":"topic name","question":"question text","options":{{"A":"option","B":"option","C":"option","D":"option"}},"correct_answer":"A","explanation":"brief explanation"}}
Rules: four non-empty options A-D, exactly one best answer, correct_answer must be A/B/C/D, no answer reveal in the question text. For Mixed practice, topic must be one of the supplied topic names."""
        with st.spinner("Generating a question..."):
            try:
                parsed = parse_quiz_json(generate_ai_text(prompt, json_mode=True), subject, topic)
                parsed.update({
                    "attempt_id": uuid.uuid4().hex,
                    "subject": subject,
                    "difficulty": level,
                    "answered": False,
                    "was_correct": False,
                    "saved": False,
                })
                st.session_state["current_quiz"] = parsed
                st.session_state.pop("quiz_feedback", None)
            except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
                st.error(str(exc) or "Could not generate a valid quiz. Please try again.")

    quiz = st.session_state.get("current_quiz")
    if isinstance(quiz, dict):
        st.divider()
        with st.container(border=True):
            st.caption(f"📌 {quiz['subject']} · {quiz['topic']} · {quiz['difficulty']}")
            st.markdown(f"### {quiz['question']}")
            for letter in "ABCD":
                st.markdown(f"**{letter}.** {quiz['options'][letter]}")
            selected = st.selectbox(
                "Your answer",
                ["Select an answer", "A", "B", "C", "D"],
                key=f"quiz_answer_key_{quiz['attempt_id']}",
                format_func=lambda item: item if item == "Select an answer" else f"{item}. {quiz['options'][item]}",
                disabled=bool(quiz.get("answered")),
            )
            if not quiz.get("answered"):
                if st.button("Submit answer →", type="primary", key="submit_quiz_answer"):
                    if selected == "Select an answer":
                        st.warning("Select an answer before submitting.")
                    else:
                        attempt = dict(quiz)
                        attempt.update({"selected_answer": selected, "is_correct": selected == quiz["correct_answer"]})
                        try:
                            saved = save_quiz_attempt(st.session_state["user_email"], attempt)
                            if not saved:
                                st.error("This quiz attempt has already been saved or the account was not found. Generate another question.")
                            else:
                                quiz["answered"] = True
                                quiz["was_correct"] = attempt["is_correct"]
                                quiz["selected_answer"] = selected
                                quiz["saved"] = True
                                st.session_state["current_quiz"] = quiz
                                st.rerun()
                        except PyMongoError:
                            st.error("Your attempt could not be saved. Check the database connection before retrying.")
            else:
                if quiz.get("was_correct"):
                    st.success("🎉 Correct answer! 10 XP awarded and the attempt saved.")
                else:
                    st.error(f"Incorrect. The correct answer was {quiz['correct_answer']}.")
                st.markdown("**Explanation**")
                st.write(quiz.get("explanation") or "Review the topic and try another question.")
                if st.button("Generate another question →", type="primary", key="another_question"):
                    st.session_state["current_quiz"] = None
                    st.rerun()

# -----------------------------------------------------------------------------
# Performance: persisted topic and subject analytics
# -----------------------------------------------------------------------------
def render_performance() -> None:
    st.markdown(
        """
        <div class="falcon-hero-banner" style="background:linear-gradient(135deg,#064e3b 0%,#0f766e 52%,#2563eb 100%);">
          <span class="falcon-badge-pill">📊 REAL-TIME TELEMETRY</span>
          <h1>Learning Performance & Analytics</h1>
          <p>Review your all-time subject mastery, topic-wise accuracy breakdown, and recent quiz history.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        user = find_user_by_email(st.session_state["user_email"]) or {}
        score = int(user.get("score", 0) or 0)
        topic_rows = get_topic_analytics(user)
        history = get_user_history(st.session_state["user_email"])
    except PyMongoError:
        st.error("Performance data is temporarily unavailable. Check the database connection.")
        return
    total_attempts = sum(row["Attempts"] for row in topic_rows)
    total_correct = sum(row["Correct"] for row in topic_rows)
    accuracy = total_correct / total_attempts * 100 if total_attempts else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 Total Quiz XP", f"{score} XP")
    c2.metric("📝 Quiz Attempts", total_attempts)
    c3.metric("✅ Correct Answers", total_correct)
    c4.metric("🎯 Overall Accuracy", f"{accuracy:.1f}%" if total_attempts else "—")
    st.progress(min(score / 100, 1.0), text=f"Progress to 100 XP Milestone · {min(score, 100)} / 100")
    st.caption("XP is accumulated over time. Topic analytics are all-time counters; the attempt table below keeps the latest 500 attempts.")

    st.subheader("Subject-wise performance")
    subject_rows = []
    for subject in SUBJECTS:
        rows = [row for row in topic_rows if row["Subject"] == subject]
        attempts = sum(row["Attempts"] for row in rows)
        correct = sum(row["Correct"] for row in rows)
        subject_rows.append({"Subject": subject, "Attempts": attempts, "Correct": correct, "Accuracy (%)": round(correct / attempts * 100, 1) if attempts else 0.0})
    st.dataframe(pd.DataFrame(subject_rows), use_container_width=True, hide_index=True)

    st.subheader("Topic-wise performance")
    if topic_rows:
        st.dataframe(pd.DataFrame(topic_rows), use_container_width=True, hide_index=True)
        chart_rows = [row for row in topic_rows if row["Attempts"] > 0]
        if chart_rows:
            plot_df = pd.DataFrame(chart_rows).set_index("Topic")[["Accuracy (%)"]]
            st.bar_chart(plot_df)
        st.markdown("**Suggested revision**")
        st.markdown(compute_recommendation(topic_rows))
    else:
        st.info("No quiz attempts yet. Generate and submit your first quiz to see topic analytics.")

    st.subheader("Recent quiz history")
    if history:
        display = []
        for item in reversed(history[-50:]):
            created = item.get("created_at")
            if isinstance(created, datetime):
                date_text = created.astimezone().strftime("%d %b %Y, %I:%M %p") if created.tzinfo else created.strftime("%d %b %Y, %I:%M %p")
            else:
                date_text = "—"
            display.append({
                "Date": date_text,
                "Subject": item.get("subject", "—"),
                "Topic": item.get("topic", "—"),
                "Difficulty": item.get("difficulty", "—"),
                "Result": "Correct" if item.get("is_correct") else "Incorrect",
                "XP": int(item.get("xp_awarded", 0) or 0),
            })
        st.dataframe(pd.DataFrame(display), use_container_width=True, hide_index=True)
        st.markdown("**Review a previous attempt**")
        review_items = list(reversed(history[-20:]))
        review_map = {
            str(index): f"{item.get('subject', '—')} · {item.get('topic', '—')} · {'Correct' if item.get('is_correct') else 'Incorrect'} · {item.get('created_at').strftime('%d %b %Y %H:%M') if isinstance(item.get('created_at'), datetime) else 'date unavailable'}"
            for index, item in enumerate(review_items)
        }
        selected_review = st.selectbox(
            "Select an attempt to review",
            options=list(review_map.keys()),
            format_func=lambda key: review_map[key],
            key="review_quiz_attempt",
        )
        chosen = review_items[int(selected_review)]
        with st.container(border=True):
            st.markdown(f"**{chosen.get('question', 'Question text unavailable')}**")
            options = chosen.get("options", {})
            for letter in "ABCD":
                if letter in options:
                    st.write(f"{letter}. {options[letter]}")
            st.write(f"Your answer: {chosen.get('selected_answer', '—')}")
            st.write(f"Correct answer: {chosen.get('correct_answer', '—')}")
            st.caption("Correct" if chosen.get("is_correct") else "Incorrect")
            st.write(chosen.get("explanation") or "No explanation was saved for this attempt.")
    else:
        st.info("Your completed quiz attempts will appear here.")

# -----------------------------------------------------------------------------
# Phase 3 UI: Revision Center
# -----------------------------------------------------------------------------
def render_revision_center() -> None:
    st.markdown(
        """
        <div class="falcon-hero-banner" style="background:linear-gradient(135deg,#7c2d12 0%,#be185d 52%,#6d28d9 100%);">
          <span class="falcon-badge-pill">🧠 SMART RETENTION LAB</span>
          <h1>Smart Revision Center</h1>
          <p>Focus on weaker topics automatically, generate exam-ready revision notes, unlock achievements, and export your history.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    email = st.session_state["user_email"]
    try:
        user = find_user_by_email(email) or {}
        history = get_user_history(email)
        topic_rows = get_topic_analytics(user)
    except PyMongoError:
        st.error("Revision data could not be loaded. Check the database connection and try again.")
        return

    score = int(user.get("score", 0) or 0)
    total_attempts = len(history)
    correct = sum(1 for item in history if isinstance(item, dict) and item.get("is_correct"))
    accuracy = correct / total_attempts * 100 if total_attempts else 0.0
    streak, active_days = get_learning_streak(history)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🔥 Current Streak", f"{streak} day(s)")
    m2.metric("📅 Days Practised", active_days)
    m3.metric("❓ Questions Answered", total_attempts)
    m4.metric("🎯 Overall Accuracy", f"{accuracy:.1f}%" if total_attempts else "—")

    st.subheader("Your next best revision")
    recommended = choose_revision_topic(topic_rows)
    if recommended:
        st.markdown(
            f"Based on your quiz history, start with **{recommended['Topic']}** in **{recommended['Subject']}** "
            f"({recommended['Accuracy (%)']}% accuracy across {recommended['Attempts']} attempt(s))."
        )
    else:
        st.info("There is not enough quiz history for a personalised recommendation yet. Start with a topic below, then answer quizzes to build your revision profile.")

    st.divider()
    st.subheader("Generate focused revision notes")
    subjects = list(SUBJECTS)
    default_subject = recommended["Subject"] if recommended and recommended["Subject"] in subjects else "DSA"
    subject_index = subjects.index(default_subject)
    col_rs, col_rt = st.columns(2)
    with col_rs:
        subject = st.selectbox("Subject", subjects, index=subject_index, key="revision_subject")
    topics = SUBJECT_TOPICS[subject]
    default_topic = recommended["Topic"] if recommended and recommended["Subject"] == subject and recommended["Topic"] in topics else topics[0]
    topic_index = topics.index(default_topic)
    with col_rt:
        topic = st.selectbox("Topic", topics, index=topic_index, key="revision_topic")
    style = st.session_state.get("learning_style", LEARNING_STYLES[0])
    level = st.session_state.get("student_level", LEVELS[0])

    if st.button("📝 Generate revision notes", type="primary", key="generate_revision_notes"):
        prompt = f"""Create exam-ready revision notes for a Computer Science student.
Subject: {subject}
Topic: {topic}
Student level: {level}
Preferred style: {style}
Use accurate, concise explanations. Include:
1. Core idea and why it matters
2. Key definitions or formulas, where relevant
3. A simple example or analogy
4. Common mistakes to avoid
5. Three quick self-check questions with answers at the end
Use clear headings and Markdown. Do not claim that code was executed."""
        with st.spinner("Preparing revision notes..."):
            try:
                notes = generate_ai_text(prompt)
                st.session_state["revision_notes_result"] = {"subject": subject, "topic": topic, "notes": notes}
            except RuntimeError as exc:
                st.error(f"Revision notes could not be generated. {exc}")

    saved_notes = st.session_state.get("revision_notes_result")
    if isinstance(saved_notes, dict) and saved_notes.get("notes"):
        with st.container(border=True):
            st.markdown(f"### {saved_notes.get('subject', '')} · {saved_notes.get('topic', '')}")
            st.markdown(saved_notes["notes"])
            safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", f"{saved_notes.get('subject', 'revision')}_{saved_notes.get('topic', 'notes')}").strip("_").lower()
            st.download_button(
                "⬇️ Download revision notes (.md)",
                data=str(saved_notes["notes"]),
                file_name=f"{safe_name or 'revision_notes'}.md",
                mime="text/markdown",
            )

    if st.button("⚡ Practise this topic in Quiz", key="practice_revision_topic"):
        st.session_state["quiz_subject"] = subject
        st.session_state["quiz_topic"] = topic
        st.session_state["current_page"] = "Quiz"
        st.rerun()

    st.divider()
    st.subheader("🏅 Achievements")
    achievement_rows = get_achievements(history, score, streak)
    unlocked = sum(1 for item in achievement_rows if item["unlocked"])
    st.caption(f"Unlocked: {unlocked} / {len(achievement_rows)}")
    achievement_cols = st.columns(3)
    for index, item in enumerate(achievement_rows):
        with achievement_cols[index % 3]:
            with st.container(border=True):
                badge_icon = "🏆 Unlocked" if item["unlocked"] else "🔒 Locked"
                st.markdown(f"**{badge_icon} · {item['name']}**")
                st.caption(item["description"])

    st.divider()
    st.subheader("📥 Export your quiz history")
    if history:
        export_df = build_quiz_export(history)
        st.download_button(
            "⬇️ Download quiz history (.csv)",
            data=export_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="falcon_students_quiz_history.csv",
            mime="text/csv",
        )
        st.caption("The export includes saved quiz questions, your answers, answer keys, results, XP, and timestamps for your own revision.")
    else:
        st.info("Your quiz history export will be available after your first submitted attempt.")


# -----------------------------------------------------------------------------
# Study plans: structured plan stored in MongoDB with task completion
# -----------------------------------------------------------------------------
def render_study_plan() -> None:
    st.markdown(
        """
        <div class="falcon-hero-banner" style="background:linear-gradient(135deg,#0f172a 0%,#0369a1 52%,#10b981 100%);">
          <span class="falcon-badge-pill">🗺️ GOAL ROADMAP BUILDER</span>
          <h1>Personalized Study Plans</h1>
          <p>Generate a day-by-day CS roadmap, save it to your MongoDB account, and check off tasks as you complete them.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.form("study_plan_form"):
        goal = st.text_input("What is your goal?", placeholder="For example: Prepare for technical interviews", max_chars=250)
        timeline = st.selectbox("How much time do you have?", ("1 Week", "2 Weeks", "1 Month"))
        submitted = st.form_submit_button("✨ Generate and save my roadmap", type="primary")
    if submitted:
        clean_goal = goal.strip()
        if len(clean_goal) < 3:
            st.warning("Enter a specific learning goal first.")
        else:
            days_count = {"1 Week": 7, "2 Weeks": 14, "1 Month": 30}[timeline]
            prompt = f"""Build a realistic Computer Science study plan as JSON only.
Goal: {clean_goal}
Duration: {timeline} ({days_count} days)
Student level: {st.session_state.get('student_level', 'Beginner')}
Preferred style: {st.session_state.get('learning_style', LEARNING_STYLES[0])}
Return JSON with this schema:
{{"title":"short title","days":[{{"day":1,"focus":"one focus","tasks":["specific task 1","specific task 2"]}}]}}
Include exactly {days_count} day objects, in order. Each day has one short focus and 2 practical tasks. Mix learning, practice, revision and checkpoints. Keep each task concise. Do not include Markdown fences."""
            with st.spinner("Generating and saving your study plan..."):
                try:
                    plan_data = parse_study_plan(generate_ai_text(prompt, json_mode=True), clean_goal, timeline)
                    plan_id = create_study_plan(st.session_state["user_email"], clean_goal, timeline, plan_data)
                    st.session_state["active_plan_id"] = plan_id
                    st.session_state["selected_saved_plan"] = plan_id
                    st.success("Your study plan has been saved to your account.")
                except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
                    st.error(str(exc) or "Could not generate the study plan. Please retry.")
                except PyMongoError:
                    st.error("The plan could not be saved to the database. Check your connection and try again.")

    st.divider()
    st.subheader("Saved plans")
    try:
        plans = get_study_plans(st.session_state["user_email"])
    except PyMongoError:
        st.error("Saved plans could not be loaded.")
        return
    if not plans:
        st.info("You have not saved a study plan yet. Generate one above.")
        return

    plan_labels = {
        plan["plan_id"]: f"{plan.get('title', plan.get('goal', 'Study Plan'))} · {plan.get('timeline', '')} · {plan.get('created_at').strftime('%d %b %Y') if isinstance(plan.get('created_at'), datetime) else 'Saved plan'}"
        for plan in plans
    }
    ids = [plan["plan_id"] for plan in plans]
    default_index = ids.index(st.session_state["active_plan_id"]) if st.session_state.get("active_plan_id") in ids else 0
    selected_id = st.selectbox("Choose a saved plan", ids, index=default_index, format_func=lambda item: plan_labels.get(item, "Study plan"), key="selected_saved_plan")
    plan = next(item for item in plans if item["plan_id"] == selected_id)
    done, total, percent = get_plan_progress(plan)
    st.markdown(f"### {plan.get('title', 'Study Plan')}")
    st.caption(f"Goal: {plan.get('goal', '')} · Duration: {plan.get('timeline', '')}")
    st.progress(percent / 100 if total else 0.0, text=f"Tasks completed: {done} / {total} ({percent:.0f}%)")
    st.caption("Task changes save automatically to MongoDB.")
    for day in plan.get("days", []):
        with st.expander(f"📅 Day {day.get('day', '?')} · {day.get('focus', 'Study tasks')}", expanded=False):
            for task in day.get("tasks", []):
                widget_key = f"task_{plan['plan_id']}_{task['task_id']}"
                initial = bool(task.get("completed"))
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = initial
                current_value = st.checkbox(task.get("text", "Study task"), key=widget_key)
                if current_value != initial:
                    try:
                        if toggle_study_task(st.session_state["user_email"], plan["plan_id"], task["task_id"], current_value):
                            st.rerun()
                        else:
                            st.warning("The task status could not be updated. Refresh and try again.")
                    except PyMongoError:
                        st.error("The task status could not be saved. Check your database connection.")

# -----------------------------------------------------------------------------
# Phase 4 UI: Mock Interview
# -----------------------------------------------------------------------------
def render_mock_interview() -> None:
    email = st.session_state["user_email"]
    st.markdown(
        """
        <div class="falcon-hero-banner" style="background:linear-gradient(135deg,#1e1b4b 0%,#7e22ce 52%,#e11d48 100%);">
          <span class="falcon-badge-pill">🎤 AI INTERVIEW SIMULATOR</span>
          <h1>Mock Interview Practice</h1>
          <p>Practise 5-question technical interview rounds, answer in your own words, and get structured feedback with model solutions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "AI feedback is a learning aid, not a prediction of a real hiring decision."
    )

    with st.expander("➕ Start a new mock interview", expanded=not bool(st.session_state.get("active_mock_interview_id"))):
        with st.form("mock_interview_start_form"):
            subject = st.selectbox("Subject", INTERVIEW_SUBJECTS)
            difficulty = st.selectbox(
                "Difficulty",
                LEVELS,
                index=LEVELS.index(st.session_state.get("student_level", LEVELS[0]))
                if st.session_state.get("student_level", LEVELS[0]) in LEVELS else 0,
            )
            focus = st.selectbox("Interview focus", INTERVIEW_FOCI)
            start = st.form_submit_button("🚀 Generate 5 interview questions", type="primary")

        if start:
            topic_list = (
                {key: value for key, value in SUBJECT_TOPICS.items()}
                if subject == "Mixed CS"
                else {subject: SUBJECT_TOPICS[subject]}
            )
            prompt = f"""Create exactly five varied technical interview questions.
Subject: {subject}
Difficulty: {difficulty}
Focus: {focus}
Available topics: {json.dumps(topic_list, ensure_ascii=False)}
Return only valid JSON with this structure:
{{"questions":[{{"topic":"exact topic from the available list","question":"clear interview question"}}]}}
Rules: produce exactly 5 distinct questions; avoid duplicate concepts; make questions answerable in a mock interview; do not include answers or hints; keep question text concise. For Mixed CS, use topics from DSA, OS, and DBMS across the five questions."""
            with st.spinner("Creating your interview session..."):
                try:
                    raw = generate_ai_text(prompt, json_mode=True)
                    questions = parse_interview_questions(raw, subject)
                    session_id = save_mock_interview(
                        email, subject, difficulty, focus, questions
                    )
                    st.session_state["active_mock_interview_id"] = session_id
                    st.success("Interview session saved. Answer the questions below.")
                    st.rerun()
                except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
                    st.error(str(exc) or "Could not create the interview. Please retry.")
                except PyMongoError:
                    st.error("The interview could not be saved. Check the database connection.")

    try:
        sessions = get_mock_interviews(email)
    except PyMongoError:
        st.error("Saved mock interviews could not be loaded.")
        return

    if not sessions:
        st.info("No mock interviews yet. Start one above to build your practice history.")
        return

    active_id = st.session_state.get("active_mock_interview_id")
    valid_ids = [item.get("session_id") for item in sessions]
    if active_id not in valid_ids:
        active_id = valid_ids[0]
        st.session_state["active_mock_interview_id"] = active_id

    st.subheader("Saved interview sessions")
    for session in sessions:
        sid = str(session.get("session_id", ""))
        created = session.get("created_at")
        date_label = (
            created.astimezone().strftime("%d %b %Y, %I:%M %p")
            if isinstance(created, datetime) and created.tzinfo
            else created.strftime("%d %b %Y, %I:%M %p")
            if isinstance(created, datetime)
            else "Saved session"
        )
        qs = session.get("questions", [])
        evaluated = sum(
            1 for q in qs
            if isinstance(q, dict) and isinstance(q.get("evaluation"), dict)
        )
        label = (
            f"**{session.get('subject', 'CS')}** · {session.get('difficulty', '')} · "
            f"{evaluated}/{len(qs)} evaluated · {date_label}"
        )
        left, right = st.columns([8, 1])
        with left:
            st.markdown(label)
        with right:
            if sid == active_id:
                st.button("Active", key=f"open_mock_{sid}", disabled=True)
            elif st.button("Open", key=f"open_mock_{sid}"):
                st.session_state["active_mock_interview_id"] = sid
                st.rerun()

    try:
        interview = get_mock_interview(email, str(active_id))
    except PyMongoError:
        st.error("This interview session could not be loaded.")
        return

    if not interview:
        st.warning("The selected interview session was not found. Open another session.")
        return

    questions = interview.get("questions", [])
    if not isinstance(questions, list) or not questions:
        st.warning("This session has no questions. Start a new mock interview.")
        return

    evaluated_questions = [
        item for item in questions
        if isinstance(item, dict) and isinstance(item.get("evaluation"), dict)
    ]
    scores = [
        float(item["evaluation"].get("score", 0))
        for item in evaluated_questions
    ]
    average = sum(scores) / len(scores) if scores else 0.0
    complete = len(evaluated_questions) == len(questions)

    st.divider()
    st.subheader(f"{interview.get('subject', 'CS')} · {interview.get('focus', 'Technical')}")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Questions evaluated", f"{len(evaluated_questions)}/{len(questions)}")
    metric2.metric("Average score", f"{average:.1f}/10" if scores else "—")
    metric3.metric("Session status", "Completed" if complete else "In progress")
    st.progress(len(evaluated_questions) / len(questions))

    question_labels = {
        str(index): f"Question {index + 1}: {item.get('question', '')[:90]}"
        for index, item in enumerate(questions)
    }
    selected_index_text = st.selectbox(
        "Choose a question",
        list(question_labels.keys()),
        format_func=lambda index: question_labels[index],
        key=f"mock_question_select_{active_id}",
    )
    question_index = int(selected_index_text)
    question = questions[question_index]

    with st.container(border=True):
        st.caption(f"📌 Topic: {question.get('topic', 'General')}")
        st.markdown(f"### {question.get('question', 'Question unavailable')}")
        evaluation = question.get("evaluation")

        if isinstance(evaluation, dict) and str(question.get("answer", "")).strip():
            st.markdown("**Your answer**")
            st.write(question.get("answer", ""))
            st.metric("Answer score", f"{float(evaluation.get('score', 0)):.1f}/10")
            st.markdown("**Feedback**")
            st.write(evaluation.get("feedback", "No feedback available."))

            strengths = evaluation.get("strengths", [])
            improvements = evaluation.get("improvements", [])
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ What you did well**")
                if strengths:
                    for item in strengths:
                        st.markdown(f"- {item}")
                else:
                    st.write("Keep practising clear, structured explanations.")
            with col2:
                st.markdown("**🔧 What to improve**")
                if improvements:
                    for item in improvements:
                        st.markdown(f"- {item}")
                else:
                    st.write("Add a precise example or explain the reasoning step by step.")

            if evaluation.get("model_answer"):
                with st.expander("📖 Review a sample answer"):
                    st.markdown(evaluation["model_answer"])
        else:
            answer_key = f"mock_answer_{active_id}_{question_index}"
            previous_answer = st.session_state.get(answer_key, "")
            with st.form(f"evaluate_mock_answer_{active_id}_{question_index}"):
                answer = st.text_area(
                    "Your answer",
                    value=previous_answer,
                    placeholder="Explain your reasoning as you would in an interview...",
                    max_chars=3_000,
                    height=180,
                    key=f"mock_answer_input_{active_id}_{question_index}",
                )
                evaluate = st.form_submit_button("✨ Evaluate my answer", type="primary")

            if evaluate:
                if len(answer.strip()) < 2:
                    st.warning("Write an answer before requesting feedback.")
                else:
                    with st.spinner("Reviewing your answer..."):
                        try:
                            evaluate_mock_interview_answer(
                                email=email,
                                session_id=str(active_id),
                                question_index=question_index,
                                answer=answer,
                                question=question,
                                subject=str(interview.get("subject", "Mixed CS")),
                                difficulty=str(interview.get("difficulty", "Beginner")),
                            )
                            st.session_state[answer_key] = answer
                            st.success("Feedback saved to this interview session.")
                            st.rerun()
                        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
                            st.error(str(exc) or "Could not evaluate the answer. Please try again.")
                        except PyMongoError:
                            st.error("Your feedback could not be saved. Check the database connection.")

    if complete:
        st.success(
            f"Interview completed. Your average score was {average:.1f}/10. "
            "Review the improvement points before attempting another session."
        )


# -----------------------------------------------------------------------------
# AI classroom with conversation context passed to the model
# -----------------------------------------------------------------------------
def render_classroom(subject: str) -> None:
    header, back = st.columns([8, 1.2], vertical_alignment="center")
    with header:
        st.markdown(
            f"""
            <div class="falcon-hero-banner" style="padding:1.35rem 1.8rem;margin-bottom:0.6rem;">
              <span class="falcon-badge-pill">🤖 INTERACTIVE AI CLASSROOM</span>
              <h1 style="margin:0.25rem 0 0.2rem 0 !important;">{subject} AI Tutor</h1>
              <p>The tutor uses your recent conversation history in {subject} to answer follow-up questions with clarity.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with back:
        if st.button("← Back", use_container_width=True, type="secondary"):
            st.session_state["current_page"] = "Home"
            st.rerun()
    style = st.session_state.get("learning_style", LEARNING_STYLES[0])
    level = st.session_state.get("student_level", LEVELS[0])
    key = f"messages_{subject}"
    prefill = st.session_state.pop("prefill_tutor_topic", None)
    if prefill:
        st.info(f"Recommended revision topic: {prefill}. Ask the tutor to explain it or generate a practice example.")
    for message in st.session_state[key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(f"Ask a question about {subject}...", max_chars=MAX_PROMPT_LENGTH)
    if prompt:
        prompt = prompt.strip()
        if not prompt:
            st.warning("Enter a question first.")
            return
        # Build context from previous turns before appending the new question.
        previous = st.session_state[key][-MAX_CONTEXT_MESSAGES:]
        context_text = "\n".join(
            f"{item['role'].upper()}: {item['content']}" for item in previous
            if item.get("role") in ("user", "assistant") and isinstance(item.get("content"), str)
        )
        user_message = {"role": "user", "content": prompt}
        try:
            save_chat_message(st.session_state["user_email"], subject, user_message)
        except PyMongoError:
            st.error("Your message could not be saved, so it was not sent to the AI. Check the database connection.")
            return
        st.session_state[key].append(user_message)
        with st.chat_message("user"):
            st.markdown(prompt)
        engineered_prompt = f"""You are a careful Computer Science tutor for {subject}.
Student level: {level}
Preferred explanation style: {style}
Use recent conversation context when relevant. If the student says 'this', 'that', or asks a follow-up, interpret it using the history below. Correct misunderstandings politely. Use examples and code when appropriate. Do not claim code was executed.
Recent conversation (oldest first):
{context_text if context_text else '(No earlier messages in this subject.)'}
CURRENT QUESTION:
{prompt}"""
        with st.chat_message("assistant"):
            with st.spinner("Preparing an explanation..."):
                try:
                    response_text = generate_ai_text(engineered_prompt)
                    st.markdown(response_text)
                except RuntimeError as exc:
                    response_text = f"I could not get a response from the AI service. {exc}"
                    st.warning(response_text)
        st.session_state[key].append({"role": "assistant", "content": response_text})
        try:
            save_chat_message(st.session_state["user_email"], subject, {"role": "assistant", "content": response_text})
        except PyMongoError:
            st.warning("The answer is visible, but it could not be saved to chat history.")

# -----------------------------------------------------------------------------
# Application entry point
# -----------------------------------------------------------------------------
if not st.session_state.get("logged_in"):
    render_auth_screen()
else:
    render_sidebar()
    page = st.session_state.get("current_page", "Home")
    try:
        if page == "Home":
            render_home()
        elif page == "Quiz":
            render_quiz()
        elif page == "Performance":
            render_performance()
        elif page == "Revision":
            render_revision_center()
        elif page == "MockInterview":
            render_mock_interview()
        elif page == "StudyPlan":
            render_study_plan()
        elif page.startswith("Classroom_"):
            subject_name = page.split("_", 1)[1]
            if subject_name in SUBJECTS:
                render_classroom(subject_name)
            else:
                st.session_state["current_page"] = "Home"
                st.warning("That classroom does not exist.")
        else:
            st.session_state["current_page"] = "Home"
            st.rerun()
    except PyMongoError:
        st.error("A database operation failed. Check your MongoDB connection and try again.")