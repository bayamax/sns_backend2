import re
import logging
import random
import threading

import feedparser
import openai
from typing import List
from accounts.models import User, UserSNS
from .models import Post
from urllib.parse import quote_plus
from urllib.request import urlopen

logger = logging.getLogger(__name__)

# =============================
# 設定（このファイルだけ編集すればOK）
# =============================
# 全体ON/OFF（nanoで切り替え）
ENABLED = True

# OpenAI APIキー（ハードコード版）
# 注意: セキュリティ上は推奨されませんが、要件に合わせてここに記載
OPENAI_API_KEY = "REPLACE_WITH_YOUR_OPENAI_KEY"

# 複数テーマボット（必要に応じて増減）
BOTS = [
    {
        "enabled": True,
        "theme": "AI",
        "username": "AI_news",
        "rss_list": [
            "https://news.google.com/rss/search?q=AI&hl=ja&gl=JP&ceid=JP:ja",
        ],
        "model": "gpt-4o",
        "prob_timeline": 0.02,   # タイムライン閲覧1回あたり 2%
        "max_len_ja": 250,       # URL含めて最大250字に収める
    },
    {
        "enabled": True,
        "theme": "経済",
        "username": "経済_news",
        "rss_list": [
            "https://news.google.com/rss/search?q=%E7%B5%8C%E6%B8%88&hl=ja&gl=JP&ceid=JP:ja",
        ],
        "model": "gpt-4o",
        "prob_timeline": 0.02,   # 1%
        "max_len_ja": 250,
    },
    {
        "enabled": True,
        "theme": "World Politics News",
        "username": "world_politics_news",
        "rss_list": [
            "https://news.google.com/rss/search?q=World%20Politics%20News&hl=ja&gl=JP&ceid=JP:ja",
        ],
        "model": "gpt-4o",
        "prob_timeline": 0.02,   # 2%
        "max_len_ja": 250,
    },
    {
        "enabled": True,
        "theme": "Job Career News",
        "username": "job_career_news",
        "rss_list": [
            "https://news.google.com/rss/search?q=Job%20Career%20News&hl=ja&gl=JP&ceid=JP:ja",
        ],
        "model": "gpt-4o",
        "prob_timeline": 0.02,
        "max_len_ja": 250,
    },
    {
        "enabled": True,
        "theme": "Tech & Dev News",
        "username": "tech_dev_news",
        "rss_list": [
            "https://news.google.com/rss/search?q=Tech%20%26%20Dev%20News&hl=ja&gl=JP&ceid=JP:ja",
        ],
        "model": "gpt-4o",
        "prob_timeline": 0.02,
        "max_len_ja": 250,
    },
    {
        "enabled": True,
        "theme": "Art News",
        "username": "art_news",
        "rss_list": [
            "https://news.google.com/rss/search?q=Art%20News&hl=ja&gl=JP&ceid=JP:ja",
        ],
        "model": "gpt-4o",
        "prob_timeline": 0.02,
        "max_len_ja": 250,
    },
    {
        "enabled": True,
        "theme": "Anime News",
        "username": "anime_news",
        "rss_list": [
            "https://news.google.com/rss/search?q=Anime%20News&hl=ja&gl=JP&ceid=JP:ja",
        ],
        "model": "gpt-4o",
        "prob_timeline": 0.02,
        "max_len_ja": 250,
    },
    {
        "enabled": True,
        "theme": "Indie Games News",
        "username": "indie_games_news",
        "rss_list": [
            "https://news.google.com/rss/search?q=Indie%20Games%20News&hl=ja&gl=JP&ceid=JP:ja",
        ],
        "model": "gpt-4o",
        "prob_timeline": 0.02,
        "max_len_ja": 250,
    },
    {
        "enabled": True,
        "theme": "Food News",
        "username": "food_news",
        "rss_list": [
            "https://news.google.com/rss/search?q=Food%20News&hl=ja&gl=JP&ceid=JP:ja",
        ],
        "model": "gpt-4o",
        "prob_timeline": 0.02,
        "max_len_ja": 250,
    },
    {
        "enabled": True,
        "theme": "Company & Events News",
        "username": "company_events_news",
        "rss_list": [
            "https://news.google.com/rss/search?q=Company%20%26%20Events%20News&hl=ja&gl=JP&ceid=JP:ja",
        ],
        "model": "gpt-4o",
        "prob_timeline": 0.02,
        "max_len_ja": 250,
    },
]


def maybe_trigger_async() -> None:
    """タイムライン閲覧時に各ボットごとに確率で非同期実行。呼び出し元はすぐ返る。"""
    if not ENABLED:
        return
    for cfg in BOTS:
        if not cfg.get("enabled", False):
            continue
        prob = float(cfg.get("prob_timeline", 0.0))
        if prob > 0.0 and random.random() < prob:
            threading.Thread(target=_worker, args=(cfg,), daemon=True).start()


def _worker(cfg: dict) -> None:
    try:
        if not OPENAI_API_KEY or OPENAI_API_KEY == "REPLACE_WITH_YOUR_OPENAI_KEY":
            logger.warning("[news_bot] OPENAI_API_KEY 未設定のためスキップ")
            return
        openai.api_key = OPENAI_API_KEY

        theme = cfg["theme"]
        rss_list = cfg.get("rss_list") or [
            f"https://news.google.com/rss/search?q={theme}&hl=ja&gl=JP&ceid=JP:ja"
        ]

        # RSS候補収集（各フィードから上位数件）
        bullets, links = [], []
        for url in rss_list:
            feed = feedparser.parse(url)
            for e in getattr(feed, "entries", [])[:5]:
                title = e.get("title", "")
                link = e.get("link", "")
                if title:
                    bullets.append(f"- {title} — {link}")
                    if link:
                        links.append(link)
        candidates = "\n".join(bullets[:8]) or "- 候補なし"

        max_len = int(cfg.get("max_len_ja", 250))

        # プロンプト: URLを必ず1つ含め、可能なら文末。和文max_len字以内。
        user_prompt = (
            f"テーマ: {theme} に関連する最新ニュース候補から重要な1–2件を選び、"
            f"日本語で最大{max_len}字で自然な1文に要約してください。"
            "必ず下記候補のリンクの中から1つだけを含め、可能なら文末に配置してください。"
            "ハッシュタグや装飾は不要です。\n\n候補:\n" + candidates
        )

        resp = openai.ChatCompletion.create(
            model=cfg.get("model", "gpt-4o"),
            messages=[
                {"role": "system", "content": "You are a concise Japanese news summarizer."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=320,
        )
        content = (resp["choices"][0]["message"]["content"] or "").strip()
        if not content:
            logger.info("[news_bot] 生成結果が空のためスキップ")
            return

        # URLを1つに正規化し、長さをmax_len以内に収める（URLは保持）
        content = _normalize_with_single_url_and_limit(content, links, max_len)

        # ボットユーザー自動作成（未存在なら作成）
        username = cfg.get("username") or f"{theme}_news"
        bot_user, created = User.objects.get_or_create(username=username, defaults={"email": None})
        if created:
            bot_user.set_unusable_password()
            bot_user.save(update_fields=["password"])

        # UserSNS に threadplanet を付与（未作成なら作成、異なる場合は更新）
        try:
            usersns, created_sns = UserSNS.objects.get_or_create(user=bot_user, defaults={"sns_type": "threadplanet"})
            if not created_sns and usersns.sns_type != "threadplanet":
                usersns.sns_type = "threadplanet"
                usersns.save(update_fields=["sns_type"])
        except Exception as _ex:
            logger.warning(f"[news_bot] UserSNS attach failed for {bot_user.username}: {_ex}")

        Post.objects.create(user=bot_user, content=content)
        logger.info(f"[news_bot] posted by {bot_user.username} (theme={theme})")
    except Exception as ex:
        logger.warning(f"[news_bot] failed: {ex}", exc_info=True)


_SHORT_URL_CACHE: dict[str, str] = {}

def _shorten_url(url: str, timeout_sec: int = 5) -> str:
    """TinyURLで短縮し、失敗時はis.gdを使用。失敗したら元URLを返す。"""
    if not url:
        return url
    if url in _SHORT_URL_CACHE:
        return _SHORT_URL_CACHE[url]
    try:
        tiny_api = f"https://tinyurl.com/api-create.php?url={quote_plus(url)}"
        with urlopen(tiny_api, timeout=timeout_sec) as resp:
            short = resp.read().decode("utf-8").strip()
        if short.startswith("http") and len(short) < len(url):
            _SHORT_URL_CACHE[url] = short
            return short
    except Exception:
        pass
    try:
        isgd_api = f"https://is.gd/create.php?format=simple&url={quote_plus(url)}"
        with urlopen(isgd_api, timeout=timeout_sec) as resp:
            short = resp.read().decode("utf-8").strip()
        if short.startswith("http") and len(short) < len(url):
            _SHORT_URL_CACHE[url] = short
            return short
    except Exception:
        pass
    _SHORT_URL_CACHE[url] = url
    return url


def _normalize_with_single_url_and_limit(text: str, candidate_links: List[str], max_len: int) -> str:
    # 既存URL抽出
    urls = re.findall(r'https?://\S+', text)
    chosen = None
    for u in urls:
        if u in candidate_links:
            chosen = u
            break
    if not chosen and candidate_links:
        chosen = candidate_links[0]

    # 既存URLは全て除去 → chosenのみ残す
    for u in urls:
        text = text.replace(u, "")
    text = " ".join(text.split())  # 余分な空白圧縮

    # URL付与（可能なら文末）。長いURLは短縮。
    if chosen:
        short = _shorten_url(chosen)
        if not text.endswith(("。", "、", ".", "!", "！", "?", "？")):
            text = text.rstrip()
        text = f"{text} {short}".strip()

    # 長さ調整（URLは保持）
    if len(text) <= max_len:
        return text

    if chosen:
        short = _shorten_url(chosen)
        if short in text:
            base = text.replace(short, "").strip()
            used_url = short
        elif chosen in text:
            base = text.replace(chosen, "").strip()
            used_url = chosen
        else:
            base = text
            used_url = short
        room = max_len - len(used_url) - 1  # スペース1
        room = max(room, 0)
        base = base[:room].rstrip("、。.!！?？")
        return f"{base} {used_url}".strip()
    else:
        return text[:max_len]


