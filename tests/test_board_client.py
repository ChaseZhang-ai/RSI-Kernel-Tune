"""HTTP-layer tests for the board client and the CLI's command wiring.

A fake transport records exactly what would go on the wire; no test touches the
network. The emphasis is on (a) the request matching the web UI's own calls, (b)
typed, actionable failures for 401/429/5xx/network/timeout, and (c) the token never
appearing in an error or in the audit log.
"""

from __future__ import annotations

import io
import json
import socket
import urllib.error
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from rsi_kernel_tune.board import client as bc
from rsi_kernel_tune.ext import board as cli

TOKEN = "sekret-token-0123456789"


# --------------------------------------------------------------------------- fixtures


class Call:
    def __init__(self, method, url, headers, body, timeout):
        self.method = method
        self.url = url
        self.headers = headers
        self.body = body
        self.timeout = timeout


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[Call] = []

    def request(self, method, url, *, headers, body, timeout):
        self.calls.append(Call(method, url, dict(headers), body, timeout))
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def resp(data=None, status=200, headers=None, body=None) -> bc.Response:
    if body is None:
        body = json.dumps({"data": data}).encode()
    return bc.Response(status, headers or {"Content-Type": "application/json"}, body)


def make_client(*responses, token=TOKEN) -> tuple[bc.BoardClient, FakeTransport]:
    transport = FakeTransport(*responses)
    return bc.BoardClient(token=token, transport=transport, api_base="https://board.test/api"), transport


# --------------------------------------------------------------------------- requests


def test_whoami_reads_me_with_bearer():
    me = {"authenticated": True, "user": {"id": 7, "username": "u", "email": "u@x"}}
    client, transport = make_client(resp(me))
    assert client.me() == me
    call = transport.calls[0]
    assert call.method == "GET"
    assert call.url == "https://board.test/api/me"
    assert call.headers["Authorization"] == f"Bearer {TOKEN}"


def test_submit_kernel_sends_the_ui_fields(tmp_path):
    f = tmp_path / "submission.py"
    f.write_text("def run(x):\n    return x\n", encoding="utf-8")
    client, transport = make_client(resp({"submission_id": 11}))
    out = client.submit_kernel(45, f, gpu_type="B200", mode="private")
    assert out == {"submission_id": 11}
    call = transport.calls[0]
    assert call.method == "POST"
    assert call.url == "https://board.test/api/submissions/upload"
    assert call.headers["Content-Type"].startswith("multipart/form-data; boundary=")
    body = call.body
    assert b'name="file"; filename="submission.py"' in body
    assert b'name="kernel_id"' in body and b"45" in body
    assert b'name="gpu_type"' in body and b"B200" in body
    assert b'name="submission_mode"' in body and b"private" in body
    assert call.headers["Authorization"] == f"Bearer {TOKEN}"


def test_submit_collection_has_no_kernel_id(tmp_path):
    f = tmp_path / "l1.zip"
    f.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    client, transport = make_client(resp({"submission_id": 12}))
    client.submit_collection(3, f, gpu_type="B200", mode="release")
    call = transport.calls[0]
    assert call.url == "https://board.test/api/collections/3/submit"
    assert b'name="kernel_id"' not in call.body
    assert b'name="submission_mode"' in call.body and b"release" in call.body


def test_publish_is_a_bodyless_post():
    client, transport = make_client(resp({"ok": True}))
    client.make_public(9)
    call = transport.calls[0]
    assert call.method == "POST"
    assert call.url == "https://board.test/api/submissions/9/make-public"
    assert call.body is None


def test_list_submissions_builds_query_params_only_for_given_filters():
    page = {"submissions": [{"id": 1}], "total": 1}
    client, transport = make_client(resp(page))
    subs, total = client.list_submissions(user_id=7, gpu_type="B200", status="COMPLETED",
                                          offset=20, limit=10)
    assert subs == [{"id": 1}] and total == 1
    url = transport.calls[0].url
    for part in ("user_id=7", "gpu_type=B200", "status=COMPLETED", "offset=20", "limit=10"):
        assert part in url
    assert "kernel_id" not in url
    assert transport.calls[0].headers["Authorization"] == f"Bearer {TOKEN}"


def test_iter_submissions_pages_until_total():
    first = {"submissions": [{"id": 1}, {"id": 2}], "total": 4}
    second = {"submissions": [{"id": 3}, {"id": 4}], "total": 4}
    client, transport = make_client(resp(first), resp(second))
    got = list(client.iter_submissions(page=2, user_id=7))
    assert [s["id"] for s in got] == [1, 2, 3, 4]
    assert "offset=0" in transport.calls[0].url
    assert "offset=2" in transport.calls[1].url


def test_download_reads_filename_from_content_disposition():
    response = bc.Response(200, {"Content-Disposition": 'attachment; filename="my.py"'},
                           b"print(1)")
    client, _ = make_client(response)
    body, name = client.download_submission(4)
    assert body == b"print(1)" and name == "my.py"


# --------------------------------------------------------------------------- failures


def test_401_is_actionable_and_does_not_leak_token():
    client, _ = make_client(bc.Response(401, {}, b'{"detail":"token expired"}'))
    with pytest.raises(bc.AuthError) as info:
        client.me()
    msg = str(info.value)
    assert "HTTP 401" in msg and "SOL_EXECBENCH_TOKEN" in msg
    assert TOKEN not in msg


def test_429_reports_retry_after():
    client, _ = make_client(bc.Response(429, {"Retry-After": "120"}, b'{"detail":"slow"}'))
    with pytest.raises(bc.RateLimited) as info:
        client.list_submissions(user_id=7)
    assert info.value.retry_after_s == 120
    assert "429" in str(info.value) and "限速" in str(info.value)
    assert TOKEN not in str(info.value)


def test_500_is_server_error():
    client, _ = make_client(bc.Response(500, {}, b"oops"))
    with pytest.raises(bc.ServerError) as info:
        client.me()
    assert "500" in str(info.value) and TOKEN not in str(info.value)


def test_network_error_and_timeout_are_wrapped():
    client, _ = make_client(urllib.error.URLError("dns down"))
    with pytest.raises(bc.NetworkError) as info:
        client.me()
    assert "网络错误" in str(info.value) and TOKEN not in str(info.value)

    client, _ = make_client(socket.timeout("slow"))
    with pytest.raises(bc.NetworkError) as info:
        client.me()
    assert "超时" in str(info.value) and TOKEN not in str(info.value)


def test_non_json_response_is_protocol_error():
    client, _ = make_client(bc.Response(200, {}, b"<html>"))
    with pytest.raises(bc.ProtocolError):
        client.list_kernels()


def test_server_detail_containing_token_is_redacted():
    client, _ = make_client(
        bc.Response(400, {}, json.dumps({"detail": f"bad token {TOKEN}"}).encode()))
    with pytest.raises(bc.BoardError) as info:
        client.me()
    assert TOKEN not in str(info.value)
    assert "***" in str(info.value)


def test_missing_token_raises_the_shared_hint(monkeypatch):
    monkeypatch.delenv(bc.TOKEN_ENV, raising=False)
    client = bc.BoardClient(transport=FakeTransport())
    with pytest.raises(bc.TokenMissing) as info:
        client.require_token()
    assert "SOL_EXECBENCH_TOKEN" in str(info.value)
    assert "docs/board.md" in str(info.value)


# --------------------------------------------------------------------------- resolution


def _kernels():
    return [
        {"id": 45, "name": "069_rms_norm", "title": "RMS Norm", "tags": ["L1"],
         "gpu_types": ["B200"]},
        {"id": 46, "name": "070_rms_norm_backward", "title": "RMS Norm Backward",
         "tags": ["L1"], "gpu_types": ["B200"]},
        {"id": 47, "name": "001_fp8_attention_output_projection", "title": "FP8 AOP",
         "tags": ["Quant"], "gpu_types": ["B200"]},
    ]


def test_resolve_kernel_exact_title_and_prefix():
    client, _ = make_client(resp({"kernels": _kernels()}))
    assert client.resolve_kernel("069_rms_norm")["id"] == 45
    client, _ = make_client(resp({"kernels": _kernels()}))
    assert client.resolve_kernel("RMS Norm")["id"] == 45
    client, _ = make_client(resp({"kernels": _kernels()}))
    assert client.resolve_kernel("001_fp8")["id"] == 47


def test_resolve_kernel_unknown_lists_close_matches():
    client, _ = make_client(resp({"kernels": _kernels()}))
    with pytest.raises(bc.BoardLookupError) as info:
        client.resolve_kernel("069_rmsnorm")
    assert "069_rms_norm" in str(info.value) or "相近" in str(info.value)


def test_resolve_collection_aliases_and_ids():
    collections = [
        {"id": 1, "name": "l1", "title": "L1 - Single Operations", "gpu_types": ["B200"],
         "kernels": [{"name": "069_rms_norm"}]},
        {"id": 2, "name": "l2", "title": "L2 - Fused Operations", "gpu_types": ["B200"],
         "kernels": [{"name": "030_attention_output_projection_with_residual"}]},
        {"id": 3, "name": "quant", "title": "Quantization", "gpu_types": ["B200"],
         "kernels": [{"name": "001_fp8_attention_output_projection"}]},
        {"id": 4, "name": "flashinfer-bench", "title": "FlashInfer-Bench",
         "gpu_types": ["B200"], "kernels": [{"name": "flashinfer_attention"}]},
    ]
    for text, want in (("l1", 1), ("L1 - Single Operations", 1), ("2", 2),
                       ("quant", 3), ("Quantization", 3), ("flashinfer", 4), ("4", 4)):
        client, _ = make_client(resp({"collections": collections}))
        assert client.resolve_collection(text)["id"] == want
    client, _ = make_client(resp({"collections": collections}))
    with pytest.raises(bc.BoardLookupError):
        client.resolve_collection("nope")


def test_unknown_collection_fetches_detail_for_kernels():
    listing = {"collections": [{"id": 1, "name": "l1", "title": "L1 - Single Operations"}]}
    detail = {"id": 1, "name": "l1", "title": "L1 - Single Operations",
              "kernels": [{"name": "069_rms_norm"}]}
    client, transport = make_client(resp(listing), resp(detail))
    coll = client.resolve_collection("l1")
    assert coll["kernels"][0]["name"] == "069_rms_norm"
    assert transport.calls[1].url == "https://board.test/api/collections/1"


# --------------------------------------------------------------------------- URL guard


def test_url_guard_requires_https_and_allowlisted_host():
    with pytest.raises(bc.BoardError) as info:
        bc.check_request_url("http://research.nvidia.com/x")
    assert "https" in str(info.value)
    with pytest.raises(bc.BoardError) as info:
        bc.check_request_url("https://evil.example/x")
    assert "允许列表" in str(info.value)


def test_url_guard_refuses_private_addresses():
    with pytest.raises(bc.BoardError) as info:
        bc.check_request_url("https://127.0.0.1/x", hosts=frozenset({"127.0.0.1"}))
    assert "非公网" in str(info.value)


# --------------------------------------------------------------------------- CLI


class FakeClient:
    """Just enough of BoardClient for the CLI's non-network paths."""

    api_base = "https://board.test/api"
    token = TOKEN

    def __init__(self):
        self.submitted = []
        self.published = []
        self.returned = {"submission_id": 77}
        self.extra_submissions = []

    def require_token(self):
        return self

    def me(self):
        return {"authenticated": True,
                "user": {"id": 7, "username": "u", "email": "u@x", "display_name": "U",
                         "account_group": "regular",
                         "submission_access": {
                             "limits": {"max_concurrent_submissions": 5},
                             "usage": {"active_submissions": 0,
                                       "kernel_submissions_today": 0},
                             "result_delay": {"max_delay_minutes": 60,
                                              "collection_delay_hours": 6}}}}

    def resolve_kernel(self, problem, subset=None):
        return {"id": 45, "name": "069_rms_norm", "gpu_types": ["B200"]}

    def submit_kernel(self, kernel_id, path, *, gpu_type="B200", mode="private"):
        self.submitted.append((kernel_id, Path(path).name, gpu_type, mode))
        return dict(self.returned)

    def submit_collection(self, collection_id, path, *, gpu_type="B200", mode="private"):
        self.submitted.append((collection_id, Path(path).name, gpu_type, mode))
        return dict(self.returned)

    def make_public(self, submission_id):
        self.published.append(submission_id)
        return {"ok": True}

    def get_submission(self, submission_id):
        return {"id": submission_id, "status": "COMPLETED", "is_correct": True,
                "is_disqualified": False, "sol_score": 0.75, "latency_ms": 0.01,
                "kernel_name": "069_rms_norm", "gpu_type": "B200",
                "submission_mode": "private", "benchmark_results": []}

    def resolve_collection(self, track):
        return {"id": 1, "name": "l1", "title": "L1 - Single Operations",
                "gpu_types": ["B200"], "kernels": [{"name": "069_rms_norm"}]}

    def list_submissions(self, **kwargs):
        return list(self.extra_submissions), len(self.extra_submissions)

    def iter_submissions(self, **kwargs):
        return iter(self.extra_submissions)


def run_cli(argv, client=None):
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(argv, client=client, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def _kernel_file(tmp_path):
    f = tmp_path / "submission.py"
    f.write_text("def run(x):\n    return x\n", encoding="utf-8")
    return f


def test_cli_help_exits_zero():
    with pytest.raises(SystemExit) as info:
        cli.main(["--help"])
    assert info.value.code == 0


@pytest.mark.parametrize("command", ["whoami", "submit", "status", "result", "publish",
                                     "submit-collection", "import"])
def test_cli_every_subcommand_has_help(command, capsys):
    with pytest.raises(SystemExit) as info:
        cli.main([command, "--help"])
    assert info.value.code == 0
    assert "usage: forge board" in capsys.readouterr().out


def test_cli_without_token_prints_one_hint(monkeypatch):
    monkeypatch.delenv(bc.TOKEN_ENV, raising=False)
    code, out, err = run_cli(["whoami"])
    assert code == cli.EXIT_ERROR
    assert "SOL_EXECBENCH_TOKEN" in err and "docs/board.md" in err
    assert out == ""


def test_cli_whoami_renders_policy():
    code, out, _ = run_cli(["whoami"], client=FakeClient())
    assert code == 0
    assert "账号：u" in out and "并发上限：5" in out and "今日单题提交：0" in out


def test_cli_submit_dry_run_sends_nothing(tmp_path):
    fake = FakeClient()
    code, out, err = run_cli(["submit", "069_rms_norm", str(_kernel_file(tmp_path)),
                              "--dry-run"], client=fake)
    assert code == 0
    assert "[DRY-RUN]" in out and "kernel_id: 45" in out
    assert fake.submitted == []


def test_cli_submit_uploads_and_writes_audit_log_without_token(tmp_path):
    fake = FakeClient()
    log = tmp_path / "log.jsonl"
    code, out, err = run_cli(["submit", "069_rms_norm", str(_kernel_file(tmp_path)),
                              "--no-wait", "--log", str(log)], client=fake)
    assert code == 0, err
    assert fake.submitted == [(45, "submission.py", "B200", "private")]
    assert "已提交 #77" in out
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["command"] == "submit" and record["submission_id"] == 77
    assert record["mode"] == "private" and record["gpu_type"] == "B200"
    assert TOKEN not in log.read_text(encoding="utf-8")


def test_cli_submit_refuses_over_concurrency_limit(tmp_path):
    fake = FakeClient()
    base_me = fake.me

    def over():
        data = base_me()
        data["user"]["submission_access"]["usage"]["active_submissions"] = 5
        return data

    fake.me = over  # type: ignore[method-assign]
    code, out, err = run_cli(["submit", "069_rms_norm", str(_kernel_file(tmp_path)),
                              "--no-wait"], client=fake)
    assert code == cli.EXIT_BLOCKED
    assert "在途提交 5/5" in err
    assert fake.submitted == []


def test_cli_submit_attempts_even_when_the_self_check_is_bad(tmp_path):
    # a bad self-check is advisory now: it warns but does not block the submission.
    f = tmp_path / "submission.py"
    f.write_text("def runner(x):\n    return x\n", encoding="utf-8")
    fake = FakeClient()
    code, _, err = run_cli(["submit", "069_rms_norm", str(f), "--no-wait",
                            "--log", str(tmp_path / "log.jsonl")], client=fake)
    assert code in (0, cli.EXIT_OK)
    assert fake.submitted and "提示" in err


def test_cli_publish_needs_no_confirmation_by_default(tmp_path):
    fake = FakeClient()
    code, out, _ = run_cli(["publish", "9", "--log", str(tmp_path / "log.jsonl")],
                           client=fake)
    assert code == 0 and fake.published == [9]
    assert "已设为公开" in out


def test_cli_publish_yes_calls_api(tmp_path):
    fake = FakeClient()
    code, out, _ = run_cli(["publish", "9", "--yes",
                            "--log", str(tmp_path / "log.jsonl")], client=fake)
    assert code == 0 and fake.published == [9]
    assert "已设为公开" in out


def test_cli_result_json_is_machine_readable():
    code, out, _ = run_cli(["result", "9", "--json"], client=FakeClient())
    assert code == 0
    payload = json.loads(out)
    assert payload["sol_score"] == 0.75
    assert TOKEN not in out


def test_cli_status_without_id_lists_submissions():
    fake = FakeClient()
    fake.list_submissions = lambda **kw: ([{"id": 1, "status": "COMPLETED",
                                            "is_correct": True, "sol_score": 0.5,
                                            "kernel_name": "k", "submitted_at": "t"}], 1)
    code, out, _ = run_cli(["status"], client=fake)
    assert code == 0
    assert "#" in out and "完成(COMPLETED)" in out and "0.500000" in out


def _collection_zip(tmp_path, name="069_rms_norm"):
    archive = tmp_path / "l1.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"{name}/submission.py", "def run(x):\n    return x\n")
    return archive


def test_cli_submit_collection_dry_run_after_self_check(tmp_path):
    fake = FakeClient()
    archive = _collection_zip(tmp_path)
    code, out, err = run_cli(["submit-collection", "l1", str(archive),
                              "--yes", "--dry-run"], client=fake)
    assert code == 0, err
    assert "[DRY-RUN]" in out and "1/1 道题" in out
    assert fake.submitted == []


def test_cli_submit_collection_uploads_even_when_the_self_check_is_bad(tmp_path):
    fake = FakeClient()
    archive = tmp_path / "l1.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("not_a_kernel/readme.txt", "hi")
    code, _, err = run_cli(["submit-collection", "l1", str(archive),
                            "--yes", "--dry-run"], client=fake)
    # a bad self-check is advisory now: the upload still happens.
    assert code == cli.EXIT_OK
    assert "提示" in err and "自检未通过" in err


def test_cli_submit_collection_blocks_third_upload_of_the_day(tmp_path):
    fake = FakeClient()
    fake.extra_submissions = [
        {"id": 1, "type": "collection", "collection_id": 1,
         "submitted_at": "2026-11-24T01:00:00Z", "status": "COMPLETED"},
        {"id": 2, "type": "collection", "collection_id": 1,
         "submitted_at": "2026-11-24T02:00:00Z", "status": "COMPLETED"},
    ]
    # Pin the clock so "today" is the fixture's day.
    import rsi_kernel_tune.board.policy as pl
    old = pl.now_utc
    pl.now_utc = lambda: datetime(2026, 11, 24, 12, tzinfo=timezone.utc)
    try:
        code, _, err = run_cli(["submit-collection", "l1", str(_collection_zip(tmp_path)),
                                "--yes", "--dry-run"], client=fake)
    finally:
        pl.now_utc = old
    assert code == cli.EXIT_BLOCKED
    assert "2/2" in err


def test_cli_submit_collection_accepts_yes_and_uploads(tmp_path):
    fake = FakeClient()
    fake.returned = {"submission_id": 88}
    code, out, err = run_cli(["submit-collection", "l1", str(_collection_zip(tmp_path)),
                              "--yes", "--log", str(tmp_path / "log.jsonl")], client=fake)
    assert code == 0, err
    assert fake.submitted and fake.submitted[0][0] == 1
    assert "已上传赛道提交 #88" in out


def test_cli_submit_warns_about_daily_delay_without_waiting(tmp_path):
    fake = FakeClient()
    base_me = fake.me

    def delayed():
        data = base_me()
        data["user"]["submission_access"]["usage"]["kernel_submissions_today"] = 5
        return data

    fake.me = delayed  # type: ignore[method-assign]
    code, out, err = run_cli(["submit", "069_rms_norm", str(_kernel_file(tmp_path)),
                              "--no-wait", "--log", str(tmp_path / "log.jsonl")],
                             client=fake)
    assert code == 0
    assert "第 6 次" in err and "10 分钟" in err
    assert fake.submitted


def test_cli_submit_explicit_wait_beyond_timeout_submits_nothing(tmp_path):
    fake = FakeClient()
    base_me = fake.me

    def very_delayed():
        data = base_me()
        data["user"]["submission_access"]["usage"]["kernel_submissions_today"] = 20
        return data

    fake.me = very_delayed  # type: ignore[method-assign]
    code, _, err = run_cli(["submit", "069_rms_norm", str(_kernel_file(tmp_path)),
                            "--wait", "--timeout", "5"], client=fake)
    assert code == cli.EXIT_BLOCKED
    assert "60 分钟" in err
    assert fake.submitted == []


def test_ext_module_contract_and_discovery():
    """The step-0 registry discovers `rsi_kernel_tune.ext` modules by name and calls main()."""

    import pkgutil

    import rsi_kernel_tune.ext as ext_pkg
    from rsi_kernel_tune.ext import board as ext_board

    assert ext_board.COMMAND == "board"
    assert callable(ext_board.main)
    names = {module.name for module in pkgutil.iter_modules(ext_pkg.__path__)}
    assert "board" in names

