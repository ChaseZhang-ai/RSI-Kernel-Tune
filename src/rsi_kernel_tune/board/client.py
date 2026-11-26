"""The SOL-ExecBench HTTP client (stdlib only, injectable transport).

Everything here mirrors the leaderboard's own web UI; ``docs/board.md`` records the
endpoint table and the exact JavaScript each row was read from (bundle
``assets/index-3d17caa7.js``, fetched 2026-11-24). No endpoint outside that table is
used, and the client never retries around a rate limit.

Contract points that matter:

* token: only ``SOL_EXECBENCH_TOKEN`` (or an explicit constructor argument used by
  tests); never printed, never written. :func:`redact` is applied to every message
  this module raises, including server-supplied ``detail`` strings.
* uploads: multipart/form-data with the same field names the UI sends --
  ``file``, ``kernel_id`` (single-kernel only), ``gpu_type``, ``submission_mode``.
* responses: the API wraps payloads in ``{"data": ...}``; ``_request`` unwraps it.
* HTTP errors become typed exceptions with actionable text; ``--dry-run`` and the
  audit log never see the token either.

The production transport accepts HTTPS only, an allowlisted host, no redirects, and
refuses to connect when any resolved address is loopback/private/link-local. That is
deliberate: the API base is a constant, so the only way a request can reach an
unexpected host is a redirect or a DNS answer -- both are refused. Tests inject a
fake transport and never touch the network.
"""

from __future__ import annotations

import ipaddress
import json
import mimetypes
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

API_BASE = "https://research.nvidia.com/benchmarks/sol-execbench/api"
TOKEN_ENV = "SOL_EXECBENCH_TOKEN"
# Explicit opt-in for a mirror: comma-separated extra hostnames (HTTPS and public IPs
# are still enforced by the transport). Empty by default.
ALLOWED_HOSTS_ENV = "SOL_EXECBENCH_ALLOWED_HOSTS"
DEFAULT_ALLOWED_HOSTS = frozenset({"research.nvidia.com"})

# The bundle's GPU selector (`DM=["A100","H100","B200"]`); B200 is the evaluation GPU.
GPU_TYPES = ("A100", "H100", "B200")
DEFAULT_GPU = "B200"

# `submission_mode` values the UI exposes (`{value:"private"}` / `{value:"release"}`).
SUBMISSION_MODES = ("private", "release")

MAX_RESPONSE_BYTES = 64 * 1024 * 1024

TOKEN_HINT = (
    "未设置环境变量 SOL_EXECBENCH_TOKEN（榜单接口的全部命令都只从这个变量读令牌）。\n"
    "  1) 浏览器登录 https://research.nvidia.com/benchmarks/sol-execbench\n"
    "  2) DevTools → Application → Local Storage → 该站点 → 复制 solbench_token 的值\n"
    "  3) export SOL_EXECBENCH_TOKEN='<粘贴>'\n"
    "令牌等同登录态，不要写进任何文件、命令行参数或报告；没有令牌时也可以按 "
    "docs/board.md 的“网页手动上传的等价步骤”操作。"
)

_BEARER_RE = re.compile(r"(Bearer\s+)[^\s,;'\"]+", re.IGNORECASE)


def redact(text: Any, *secrets: str | None) -> str:
    """Replace tokens (and any ``Bearer …``) in ``text`` with ``***``."""

    out = str(text)
    for secret in secrets:
        if secret:
            out = out.replace(secret, "***")
    return _BEARER_RE.sub(r"\1***", out)


def mask_token(token: str | None) -> str:
    """A safe display form (only ever used for diagnostics, never the full token)."""

    if not token:
        return "(未设置)"
    if len(token) <= 12:
        return "***"
    return f"{token[:4]}…{token[-4:]}"


def allowed_hosts() -> frozenset[str]:
    extra = os.environ.get(ALLOWED_HOSTS_ENV, "")
    return frozenset(DEFAULT_ALLOWED_HOSTS
                     | {h.strip().lower() for h in extra.split(",") if h.strip()})


# --------------------------------------------------------------------------- errors


class BoardError(RuntimeError):
    """Base class; ``status`` is the HTTP status when there was a response."""

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


class TokenMissing(BoardError):
    """``SOL_EXECBENCH_TOKEN`` is not set."""


class NetworkError(BoardError):
    """Connection failure / timeout / DNS before an HTTP status existed."""


class ProtocolError(BoardError):
    """The server answered but not with the JSON shape the UI reads."""


class AuthError(BoardError):
    """401 / 403."""


class NotFound(BoardError):
    """404."""


class RateLimited(BoardError):
    """429 (or a server-provided retry hint)."""

    def __init__(self, message: str, *, status: int | None = None,
                 retry_after_s: float | None = None):
        super().__init__(message, status=status)
        self.retry_after_s = retry_after_s


class ServerError(BoardError):
    """5xx."""


class APIError(BoardError):
    """Any other non-2xx."""


class BoardLookupError(BoardError):
    """A problem name / track did not resolve to exactly one board object."""


# --------------------------------------------------------------------------- transport


@dataclass
class Response:
    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""

    def header(self, name: str) -> str | None:
        for key, value in self.headers.items():
            if key.lower() == name.lower():
                return value
        return None

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"响应不是 JSON：{exc}", status=self.status) from exc


class Transport(Protocol):
    """One HTTP request. Tests inject a fake; production uses urllib."""

    def request(self, method: str, url: str, *, headers: Mapping[str, str],
                body: bytes | None, timeout: float) -> Response:
        ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects instead of following them to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        raise urllib.error.HTTPError(req.full_url, code,
                                     f"redirect refused ({newurl})", headers, fp)


def check_request_url(url: str, *, hosts: frozenset[str] = DEFAULT_ALLOWED_HOSTS) -> None:
    """Raise unless ``url`` is HTTPS on an allowed host resolving to public IPs.

    Kept separate so the rule is directly testable. DNS rebinding is narrowed (not
    eliminated) by checking every address the resolver returns immediately before
    the connection; redirects are refused outright.
    """

    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https":
        raise BoardError(f"拒绝请求：只允许 https，收到 {parts.scheme or '(无)'}://")
    host = (parts.hostname or "").lower()
    if not host or host not in hosts:
        raise BoardError(f"拒绝请求：主机 {host or '(未知)'} 不在允许列表 "
                         f"{sorted(hosts)}（镜像域名需设置 {ALLOWED_HOSTS_ENV}）")
    try:
        infos = socket.getaddrinfo(host, parts.port or 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise BoardError(f"拒绝请求：解析 {host} 失败：{exc}") from exc
    if not infos:
        raise BoardError(f"拒绝请求：{host} 没有解析结果")
    for info in infos:
        addr = str(info[4][0]).split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError as exc:
            raise BoardError(f"拒绝请求：{host} 解析出无法识别的地址 {addr}") from exc
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            raise BoardError(f"拒绝请求：{host} 解析到非公网地址 {addr}")


class UrllibTransport:
    """Stdlib transport. HTTP error statuses are returned, not raised."""

    def __init__(self, *, hosts: frozenset[str] | None = None,
                 max_response_bytes: int = MAX_RESPONSE_BYTES):
        self.hosts = hosts if hosts is not None else allowed_hosts()
        self.max_response_bytes = int(max_response_bytes)
        self._opener = urllib.request.build_opener(_NoRedirect)

    def request(self, method: str, url: str, *, headers: Mapping[str, str],
                body: bytes | None, timeout: float) -> Response:
        check_request_url(url, hosts=self.hosts)
        req = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
        try:
            with self._opener.open(req, timeout=timeout) as resp:
                length = resp.headers.get("Content-Length")
                if length is not None:
                    try:
                        if int(length) > self.max_response_bytes:
                            raise ProtocolError(
                                f"响应过大（{length} 字节 > {self.max_response_bytes}）")
                    except ValueError:
                        pass
                payload = resp.read(self.max_response_bytes + 1)
                if len(payload) > self.max_response_bytes:
                    raise ProtocolError(f"响应超过 {self.max_response_bytes} 字节上限")
                return Response(int(resp.status), dict(resp.headers.items()), payload)
        except urllib.error.HTTPError as exc:
            try:
                payload = exc.read(self.max_response_bytes)
            except Exception:  # noqa: BLE001 - body is best-effort
                payload = b""
            return Response(int(exc.code), dict((exc.headers or {}).items()), payload)


# --------------------------------------------------------------------------- multipart


def _quote_filename(name: str) -> str:
    return name.replace("\r", " ").replace("\n", " ").replace('"', "'")


def encode_multipart(fields: Mapping[str, str], *, file_field: str, filename: str,
                     content: bytes, content_type: str = "application/octet-stream",
                     boundary: str | None = None) -> tuple[bytes, str]:
    """Encode exactly what the UI's ``FormData`` sends.

    ``a.append("file", e)``, ``a.append("kernel_id", String(t))``,
    ``a.append("gpu_type", n)``, ``a.append("submission_mode", r)`` -- same order,
    same names.
    """

    boundary = boundary or f"----rsi_kernel_tune{uuid.uuid4().hex}"
    lines: list[bytes] = []
    for name, value in fields.items():
        lines += [
            f"--{boundary}".encode(),
            f'Content-Disposition: form-data; name="{name}"'.encode(),
            b"",
            str(value).encode("utf-8"),
        ]
    lines += [
        f"--{boundary}".encode(),
        (f'Content-Disposition: form-data; name="{file_field}"; '
         f'filename="{_quote_filename(filename)}"').encode(),
        f"Content-Type: {content_type}".encode(),
        b"",
        content,
        f"--{boundary}--".encode(),
        b"",
    ]
    return b"\r\n".join(lines), f"multipart/form-data; boundary={boundary}"


def _guess_content_type(filename: str) -> str:
    if filename.lower().endswith(".py"):
        return "text/x-python"
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


# --------------------------------------------------------------------------- client


_ALIASES = {
    "l1": {"l1", "l1singleoperations", "singleoperations"},
    "l2": {"l2", "l2fusedoperations", "fusedoperations"},
    "quant": {"quant", "quantization", "quantized", "quantizationtrack"},
    "flashinfer-bench": {"flashinferbench", "flashinfer"},
}


def _norm(text: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(text)).lower()


class BoardClient:
    """One account's view of the board. Inject ``transport`` in tests."""

    def __init__(self, token: str | None = None, *, api_base: str = API_BASE,
                 transport: Transport | None = None, timeout: float = 60.0,
                 clock=time.time):
        self._token = (token if token is not None else os.environ.get(TOKEN_ENV, "")).strip()
        self.api_base = api_base.rstrip("/")
        self.transport: Transport = transport or UrllibTransport()
        self.timeout = float(timeout)
        self._clock = clock
        self.user_agent = "rsi_kernel_tune-board/0.1"

    # -- token ------------------------------------------------------------

    @property
    def token(self) -> str:
        return self._token

    def require_token(self) -> "BoardClient":
        if not self._token:
            raise TokenMissing(TOKEN_HINT)
        return self

    def _safe(self, text: Any) -> str:
        return redact(text, self._token)

    # -- one request ------------------------------------------------------

    def _url(self, path: str, params: Mapping[str, Any] | None = None) -> str:
        url = f"{self.api_base}/{path.lstrip('/')}"
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            if clean:
                url += "?" + urllib.parse.urlencode(clean, doseq=True)
        return url

    def _headers(self, *, auth: bool, content_type: str | None = None) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": self.user_agent}
        if auth:
            self.require_token()
            headers["Authorization"] = f"Bearer {self._token}"
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def _request(self, method: str, path: str, *, params: Mapping[str, Any] | None = None,
                 fields: Mapping[str, str] | None = None,
                 file_path: str | Path | None = None,
                 auth: bool = True, action: str | None = None,
                 raw: bool = False) -> Any:
        action = action or f"{method} {path}"
        url = self._url(path, params)
        body: bytes | None = None
        content_type: str | None = None

        if file_path is not None:
            fp = Path(file_path)
            try:
                content = fp.read_bytes()
            except OSError as exc:
                raise BoardError(self._safe(f"{action}：读不到上传文件 {fp}：{exc}")) from exc
            body, content_type = encode_multipart(
                fields or {}, file_field="file", filename=fp.name,
                content=content, content_type=_guess_content_type(fp.name))
        elif fields is not None:
            # No current endpoint takes a JSON body; this keeps a future one honest.
            body = json.dumps(dict(fields)).encode("utf-8")
            content_type = "application/json"

        headers = self._headers(auth=auth, content_type=content_type)
        try:
            resp = self.transport.request(method, url, headers=headers, body=body,
                                          timeout=self.timeout)
        except (socket.timeout, TimeoutError) as exc:
            raise NetworkError(self._safe(
                f"{action}：请求超时（{self.timeout:.0f} 秒）。网络或榜单服务繁忙，稍后重试。")) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, (socket.timeout, TimeoutError)):
                raise NetworkError(self._safe(
                    f"{action}：请求超时（{self.timeout:.0f} 秒）。稍后重试。")) from exc
            raise NetworkError(self._safe(
                f"{action}：网络错误（{reason}）。检查网络/代理后重试；客户端不重试。")) from exc
        except OSError as exc:
            raise NetworkError(self._safe(f"{action}：连接失败（{exc}）。稍后重试。")) from exc

        if not 200 <= resp.status < 300:
            self._raise_for_status(resp, action)

        if raw:
            return resp
        payload = resp.json()
        if isinstance(payload, dict) and "data" in payload:
            return payload["data"]
        return payload

    def _error_detail(self, resp: Response) -> str:
        try:
            payload = resp.json()
        except ProtocolError:
            text = resp.body[:300].decode("utf-8", "replace")
            return text.strip()
        for key in ("detail", "message", "error"):
            value = payload.get(key) if isinstance(payload, dict) else None
            if value:
                if isinstance(value, (dict, list)):
                    return json.dumps(value, ensure_ascii=False)[:500]
                return str(value)
        return ""

    def _raise_for_status(self, resp: Response, action: str) -> None:
        detail = self._error_detail(resp)
        suffix = f"服务端说明：{detail}" if detail else "服务端没有给出说明。"
        status = resp.status
        if status == 401:
            raise AuthError(self._safe(
                f"{action}：认证失败（HTTP 401）。令牌无效或已过期，"
                "重新从网页登录态获取 SOL_EXECBENCH_TOKEN 后重试。"), status=status)
        if status == 403:
            raise AuthError(self._safe(
                f"{action}：没有权限（HTTP 403）。账号组可能不允许此操作。{suffix}"),
                status=status)
        if status == 404:
            raise NotFound(self._safe(f"{action}：找不到（HTTP 404）。{suffix}"), status=status)
        if status == 429:
            retry = resp.header("Retry-After")
            retry_s = None
            if retry:
                try:
                    retry_s = float(retry)
                except ValueError:
                    retry_s = None
            wait = f"服务端要求 {retry_s:.0f} 秒后重试。" if retry_s else "稍后再试。"
            raise RateLimited(self._safe(
                f"{action}：被限速（HTTP 429）。{wait}"
                "客户端不会绕过榜单限速。"), status=status, retry_after_s=retry_s)
        if 500 <= status < 600:
            raise ServerError(self._safe(
                f"{action}：榜单服务错误（HTTP {status}）。这是服务端问题，稍后重试。{suffix}"),
                status=status)
        raise APIError(self._safe(f"{action}：请求失败（HTTP {status}）。{suffix}"), status=status)

    # -- reads ------------------------------------------------------------

    def me(self) -> dict:
        data = self._request("GET", "/me", action="读取账号信息")
        if not isinstance(data, dict):
            raise ProtocolError("读取账号信息：响应缺少 user 对象", status=200)
        return data

    def queue_status(self) -> dict:
        data = self._request("GET", "/queue/status", auth=False, action="读取队列状态")
        return data if isinstance(data, dict) else {}

    def list_kernels(self) -> list[dict]:
        data = self._request("GET", "/kernels", auth=False, action="读取题目列表")
        kernels = data.get("kernels") if isinstance(data, dict) else data
        if not isinstance(kernels, list):
            raise ProtocolError("读取题目列表：响应里没有 kernels 列表")
        return [k for k in kernels if isinstance(k, dict)]

    def get_kernel(self, kernel_id: int | str) -> dict:
        data = self._request("GET", f"/kernels/{int(kernel_id)}", auth=False,
                             action=f"读取题目 {kernel_id}")
        if not isinstance(data, dict):
            raise ProtocolError(f"读取题目 {kernel_id}：响应不是对象")
        return data

    def list_collections(self) -> list[dict]:
        data = self._request("GET", "/collections", auth=False, action="读取赛道列表")
        collections = data.get("collections") if isinstance(data, dict) else data
        if not isinstance(collections, list):
            raise ProtocolError("读取赛道列表：响应里没有 collections 列表")
        return [c for c in collections if isinstance(c, dict)]

    def get_collection(self, collection_id: int | str) -> dict:
        data = self._request("GET", f"/collections/{int(collection_id)}", auth=False,
                             action=f"读取赛道 {collection_id}")
        if not isinstance(data, dict):
            raise ProtocolError(f"读取赛道 {collection_id}：响应不是对象")
        return data

    def list_submissions(self, *, kernel_id: int | None = None, user_id: int | None = None,
                         status: str | None = None, collection_id: int | None = None,
                         collection_submission_id: int | None = None,
                         gpu_type: str | None = None, username: str | None = None,
                         submitted_from_utc: str | None = None,
                         submitted_before_utc: str | None = None,
                         offset: int = 0, limit: int = 50) -> tuple[list[dict], int]:
        data = self._request(
            "GET", "/submissions",
            params={"kernel_id": kernel_id, "user_id": user_id, "status": status,
                    "collection_id": collection_id,
                    "collection_submission_id": collection_submission_id,
                    "gpu_type": gpu_type, "username": username,
                    "submitted_from_utc": submitted_from_utc,
                    "submitted_before_utc": submitted_before_utc,
                    "offset": offset, "limit": limit},
            action="读取提交列表")
        if not isinstance(data, dict):
            raise ProtocolError("读取提交列表：响应不是对象")
        subs = data.get("submissions") or []
        total = data.get("total")
        if not isinstance(subs, list):
            raise ProtocolError("读取提交列表：submissions 不是列表")
        try:
            total_n = int(total) if total is not None else len(subs)
        except (TypeError, ValueError):
            total_n = len(subs)
        return [s for s in subs if isinstance(s, dict)], total_n

    def iter_submissions(self, *, page: int = 100, max_items: int | None = None,
                         **filters: Any) -> Iterator[dict]:
        """All submissions matching the filters, paging until the server runs out."""

        offset = 0
        seen = 0
        while True:
            batch, total = self.list_submissions(offset=offset, limit=page, **filters)
            if not batch:
                return
            for item in batch:
                yield item
                seen += 1
                if max_items is not None and seen >= max_items:
                    return
            offset += len(batch)
            if offset >= total or len(batch) < page:
                return

    def get_submission(self, submission_id: int | str) -> dict:
        data = self._request("GET", f"/submissions/{int(submission_id)}",
                             action=f"读取提交 #{submission_id}")
        if not isinstance(data, dict):
            raise ProtocolError(f"读取提交 #{submission_id}：响应不是对象")
        return data

    def download_submission(self, submission_id: int | str) -> tuple[bytes, str]:
        resp = self._request("GET", f"/submissions/{int(submission_id)}/download",
                             action=f"下载提交 #{submission_id}", raw=True)
        disposition = resp.header("Content-Disposition") or ""
        filename = f"submission_{int(submission_id)}.py"
        match = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)\"?", disposition)
        if match:
            filename = match.group(1).strip()
        return resp.body, filename

    # -- resolution -------------------------------------------------------

    def resolve_kernel(self, problem: str, subset: str | None = None) -> dict:
        """Exact kernel ``name`` (then ``title``, then a unique prefix) -> record."""

        kernels = self.list_kernels()
        if subset:
            kernels = [k for k in kernels if subset in (k.get("tags") or [])]

        def pick(candidates: list[dict], how: str) -> dict:
            if len(candidates) == 1:
                return candidates[0]
            if not candidates:
                raise BoardLookupError(self._safe(
                    f"题目 {problem!r}{f'（赛道标签 {subset}）' if subset else ''} 在榜单上找不到。"
                    "核对 GET /api/kernels 的 name 字段（如 069_rms_norm）。"))
            ids = ", ".join(f"{c.get('name')}(id={c.get('id')})" for c in candidates[:8])
            raise BoardLookupError(self._safe(
                f"题目 {problem!r} 按{how}匹配到 {len(candidates)} 个：{ids}。"
                "请写完整题目名。"))

        exact = [k for k in kernels if k.get("name") == problem]
        if exact:
            return pick(exact, "名称")
        title = [k for k in kernels if k.get("title") == problem]
        if title:
            return pick(title, "标题")
        lower = problem.lower()
        ci = [k for k in kernels if str(k.get("name", "")).lower() == lower]
        if ci:
            return pick(ci, "名称（忽略大小写）")
        ci_title = [k for k in kernels if str(k.get("title", "")).lower() == lower]
        if ci_title:
            return pick(ci_title, "标题（忽略大小写）")
        prefix = [k for k in kernels if str(k.get("name", "")).lower().startswith(lower)]
        if len(prefix) == 1:
            return prefix[0]
        import difflib

        close = difflib.get_close_matches(problem, [str(k.get("name")) for k in kernels], n=5,
                                          cutoff=0.5)
        hint = f" 相近的题目名：{', '.join(close)}。" if close else ""
        raise BoardLookupError(self._safe(
            f"题目 {problem!r} 在榜单上找不到（共 {len(kernels)} 道题）。{hint}"))

    def resolve_collection(self, track: str) -> dict:
        """Track name / title / id -> collection record.

        Accepted: ``l1``, ``L1``, ``L1 - Single Operations``, ``1``; ``l2``; ``quant``,
        ``Quantization``, ``3``; ``flashinfer-bench``, ``FlashInfer``, ``4``.
        """

        collections = self.list_collections()
        text = str(track).strip()
        if text.isdigit():
            wanted = int(text)
            for coll in collections:
                if int(coll.get("id", -1)) == wanted:
                    return self._with_kernels(coll)
            raise BoardLookupError(self._safe(f"赛道 id={wanted} 不存在。"))

        norm = _norm(text)
        hits = []
        for coll in collections:
            names = {_norm(coll.get("name", "")), _norm(coll.get("title", ""))}
            name = str(coll.get("name", ""))
            names |= _ALIASES.get(name.lower(), set())
            if norm in names or norm == _norm(name):
                hits.append(coll)
        unique = {int(c.get("id", -1)): c for c in hits}
        if len(unique) == 1:
            return self._with_kernels(next(iter(unique.values())))
        if len(unique) > 1:
            listing = "; ".join(f"{c.get('name')}(id={c.get('id')})" for c in unique.values())
            raise BoardLookupError(self._safe(f"赛道 {track!r} 匹配到多个：{listing}。"))
        available = "; ".join(f"{c.get('name')}={c.get('title')}(id={c.get('id')})"
                              for c in collections)
        raise BoardLookupError(self._safe(
            f"赛道 {track!r} 不存在。可用：{available}"))

    def _with_kernels(self, collection: dict) -> dict:
        if collection.get("kernels"):
            return collection
        cid = collection.get("id")
        if cid is None:
            return collection
        detail = self.get_collection(cid)
        return {**collection, **detail}

    # -- writes -----------------------------------------------------------

    def submit_kernel(self, kernel_id: int | str, file_path: str | Path, *,
                      gpu_type: str = DEFAULT_GPU,
                      mode: str = "private") -> dict:
        if mode not in SUBMISSION_MODES:
            raise BoardError(f"submission_mode 只能是 {SUBMISSION_MODES}，收到 {mode!r}")
        data = self._request(
            "POST", "/submissions/upload",
            fields={"kernel_id": str(int(kernel_id)), "gpu_type": gpu_type,
                    "submission_mode": mode},
            file_path=file_path,
            action=f"上传单题提交（kernel {kernel_id}）")
        return data if isinstance(data, dict) else {"raw": data}

    def submit_collection(self, collection_id: int | str, file_path: str | Path, *,
                          gpu_type: str = DEFAULT_GPU,
                          mode: str = "private") -> dict:
        if mode not in SUBMISSION_MODES:
            raise BoardError(f"submission_mode 只能是 {SUBMISSION_MODES}，收到 {mode!r}")
        data = self._request(
            "POST", f"/collections/{int(collection_id)}/submit",
            fields={"gpu_type": gpu_type, "submission_mode": mode},
            file_path=file_path,
            action=f"上传赛道提交（collection {collection_id}）")
        return data if isinstance(data, dict) else {"raw": data}

    def make_public(self, submission_id: int | str) -> dict:
        data = self._request("POST", f"/submissions/{int(submission_id)}/make-public",
                             action=f"把提交 #{submission_id} 设为公开")
        return data if isinstance(data, dict) else {"raw": data}
