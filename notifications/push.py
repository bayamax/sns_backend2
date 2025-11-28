from __future__ import annotations

"""notifications/push.py
APNs (HTTP/2) を使って iOS デバイスへプッシュ通知を送信するユーティリティ。
環境変数に以下を設定してください。

APNS_AUTH_KEY_PATH: AuthKey_XXXX.p8 への絶対パス
APNS_KEY_ID: キーID (例: 1A2BC3D4E5)
APNS_TEAM_ID: Apple Developer Team ID (10桁)
APNS_TOPIC: 通常はアプリの Bundle ID (例: com.example.SNS)
DJANGO_DEBUG: True のときはサンドボックス(APNs development)へ送信
"""

import json
import logging
from typing import Dict, List

# -----------------------------
# Python 3.11 compatibility fix
# -----------------------------
# apns2 < 0.10 は `from collections import Iterable` など
# 廃止パスを参照しているため Python 3.10+ で ImportError になる。
# 先に collections.abc から alias を注入して回避する。

import collections, collections.abc

for _name in ("Iterable", "Mapping", "MutableMapping", "MutableSet", "Set"):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

# ---- 通常 import ----
import os
import logging

from apns2.client import APNsClient
from apns2.payload import Payload
from apns2.errors import BadDeviceToken, DeviceTokenNotForTopic, Unregistered

logger = logging.getLogger(__name__)


def _get_client() -> APNsClient | None:
    key_path = os.getenv("APNS_AUTH_KEY_PATH")
    key_id = os.getenv("APNS_KEY_ID")
    team_id = os.getenv("APNS_TEAM_ID")
    topic = os.getenv("APNS_TOPIC")

    if not all([key_path, key_id, team_id, topic]):
        logger.warning("APNs 環境変数が不足しているためプッシュ通知は送信されません")
        return None

    use_sandbox = os.getenv("DJANGO_DEBUG", "False") == "True"

    try:
        # apns2 0.7.x は TokenCredentials を使用
        from apns2.credentials import TokenCredentials

        credentials = TokenCredentials(auth_key_path=key_path, auth_key_id=key_id, team_id=team_id)

        client = APNsClient(credentials=credentials, use_sandbox=use_sandbox)
        client.topic = topic
        return client
    except Exception as e:
        logger.error(f"APNsClient 初期化失敗: {e}")
        return None


_client_cache: APNsClient | None = None


def _client() -> APNsClient | None:
    global _client_cache
    if _client_cache is None:
        _client_cache = _get_client()
    return _client_cache


def send_ios_notification(
    device_tokens: List[str],
    title: str,
    body: str,
    custom_data: Dict | None = None,
    badge: int | None = None,
):
    """指定したデバイストークンへプッシュ通知を送信する。
    device_tokens   : 送信先トークンのリスト (最大1000件程度推奨)
    title, body     : 表示するタイトル/本文
    custom_data     : userInfo に含める追加ペイロード(dict)
    badge           : アプリアイコンに表示するバッジ数（Noneの場合は1）
    """
    if not device_tokens:
        logger.debug("送信対象デバイスがありません")
        return

    client = _client()
    if client is None:
        return

    payload = Payload(
        alert={
            "title": title,
            "body": body,
        },
        badge=badge if badge is not None else 1,
        sound="default",
        custom=custom_data or {}
    )

    try:
        print(f"APNs 送信開始: devices={len(device_tokens)} title={title}")

        for token in device_tokens:
            try:
                response = client.send_notification(token, payload, client.topic)
                status = getattr(response, "status", None) or response.get("status")
                reason = getattr(response, "description", None) or response.get("reason")

                # raw debug
                print(f"APNs RAW response token={token[:12]} obj={response!r}")

                if str(status) == "200":
                    print(f"APNs 送信成功 token={token[:10]}... status={status}")
                else:
                    print(f"APNs 送信失敗 token={token[:10]}... status={status} reason={reason}")
                    if reason in [BadDeviceToken.reason, Unregistered.reason, DeviceTokenNotForTopic.reason]:
                        from notifications.models import Device
                        Device.objects.filter(token=token).delete()
                        print(f"無効なデバイストークンを削除しました: {token[:10]}...")
            except Exception as ex:
                # apns2.errors.exception.APNSServerException 系では status / reason 属性を持つ
                status = getattr(ex, "status", None)
                reason = getattr(ex, "reason", None) or getattr(ex, "description", None)

                print(f"APNs 個別送信エラー token={token[:10]}... status={status} reason={reason} err={ex}")

    except Exception as e:
        logger.error(
            f"APNs 送信中に予期せぬエラーが発生しました: {e}", exc_info=True
        )
