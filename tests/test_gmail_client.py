"""Tests for app.gmail.client: GmailClient (auth, list, get) and
fetch_recent_messages. No real Gmail API calls anywhere — the Gmail
service object and OAuth Credentials are always mocked/faked.
"""

from unittest.mock import MagicMock

import httplib2
import pytest
from googleapiclient.errors import HttpError

from app.core.models import EmailMessage
from app.gmail.client import (
    READONLY_SCOPE,
    FetchResult,
    GmailAPIError,
    GmailAuthError,
    GmailClient,
    fetch_recent_messages,
)
from app.gmail.parser import GmailParsingError


def make_http_error(status: int, message: str) -> HttpError:
    resp = httplib2.Response({"status": status})
    content = ('{"error": {"message": "%s"}}' % message).encode("utf-8")
    return HttpError(resp, content)


def make_fake_service(list_result=None, list_error=None, get_results=None, get_errors=None):
    """A MagicMock standing in for the Gmail `service` object, matching
    `service.users().messages().list(...).execute()` /
    `.get(...).execute()` chains.
    """
    get_results = get_results or {}
    get_errors = get_errors or {}
    service = MagicMock()

    def list_execute():
        if list_error:
            raise list_error
        return list_result or {"messages": []}

    service.users.return_value.messages.return_value.list.return_value.execute.side_effect = list_execute

    def get_call(userId, id, format):
        execute_mock = MagicMock()
        if id in get_errors:
            execute_mock.execute.side_effect = get_errors[id]
        else:
            execute_mock.execute.return_value = get_results[id]
        return execute_mock

    service.users.return_value.messages.return_value.get.side_effect = get_call
    return service


def make_valid_credentials(expired=False, refresh_token="rt", valid=True):
    creds = MagicMock()
    creds.expired = expired
    creds.refresh_token = refresh_token
    creds.valid = valid
    creds.to_json.return_value = '{"fake": "token"}'
    return creds


class TestReadOnlyScope:
    def test_scope_is_the_narrow_readonly_scope(self):
        assert READONLY_SCOPE == "https://www.googleapis.com/auth/gmail.readonly"

    def test_scope_does_not_request_modify_or_send(self):
        assert "modify" not in READONLY_SCOPE
        assert "send" not in READONLY_SCOPE
        assert "compose" not in READONLY_SCOPE


class TestGmailClientHasNoMutationMethods:
    def test_no_send_delete_modify_label_methods_exist(self):
        forbidden = ["send", "delete", "modify", "trash", "untrash", "insert", "import_", "batchModify", "batchDelete"]
        client_methods = {name for name in dir(GmailClient) if not name.startswith("_")}
        assert not (client_methods & set(forbidden))
        assert client_methods == {"list_message_ids", "get_message", "from_token_file"}


class TestFromTokenFile:
    def test_missing_token_file_raises_gmail_auth_error(self, tmp_path):
        with pytest.raises(GmailAuthError):
            GmailClient.from_token_file(tmp_path / "does_not_exist.json", tmp_path / "secret.json")

    def test_invalid_token_file_raises_gmail_auth_error(self, tmp_path):
        token_path = tmp_path / "token.json"
        token_path.write_text("not valid json{{{", encoding="utf-8")

        with pytest.raises(GmailAuthError):
            GmailClient.from_token_file(token_path, tmp_path / "secret.json")

    def test_valid_non_expired_token_creates_a_client(self, tmp_path, monkeypatch):
        token_path = tmp_path / "token.json"
        token_path.write_text('{"fake": "token"}', encoding="utf-8")

        creds = make_valid_credentials(expired=False, valid=True)
        monkeypatch.setattr("app.gmail.client.Credentials.from_authorized_user_file", lambda *a, **kw: creds)
        monkeypatch.setattr("app.gmail.client.build", lambda *a, **kw: MagicMock())

        client = GmailClient.from_token_file(token_path, tmp_path / "secret.json")

        assert isinstance(client, GmailClient)

    def test_expired_token_is_refreshed_and_saved(self, tmp_path, monkeypatch):
        token_path = tmp_path / "token.json"
        token_path.write_text('{"fake": "old-token"}', encoding="utf-8")

        creds = make_valid_credentials(expired=True, refresh_token="rt", valid=True)
        monkeypatch.setattr("app.gmail.client.Credentials.from_authorized_user_file", lambda *a, **kw: creds)
        monkeypatch.setattr("app.gmail.client.build", lambda *a, **kw: MagicMock())
        monkeypatch.setattr("app.gmail.client.Request", lambda: MagicMock())

        GmailClient.from_token_file(token_path, tmp_path / "secret.json")

        creds.refresh.assert_called_once()
        assert token_path.read_text(encoding="utf-8") == '{"fake": "token"}'

    def test_refresh_failure_raises_gmail_auth_error(self, tmp_path, monkeypatch):
        from google.auth.exceptions import RefreshError

        token_path = tmp_path / "token.json"
        token_path.write_text('{"fake": "old-token"}', encoding="utf-8")

        creds = make_valid_credentials(expired=True, refresh_token="rt")
        creds.refresh.side_effect = RefreshError("refresh failed")
        monkeypatch.setattr("app.gmail.client.Credentials.from_authorized_user_file", lambda *a, **kw: creds)
        monkeypatch.setattr("app.gmail.client.Request", lambda: MagicMock())

        with pytest.raises(GmailAuthError):
            GmailClient.from_token_file(token_path, tmp_path / "secret.json")

    def test_invalid_credentials_without_expiry_raises_gmail_auth_error(self, tmp_path, monkeypatch):
        token_path = tmp_path / "token.json"
        token_path.write_text('{"fake": "token"}', encoding="utf-8")

        creds = make_valid_credentials(expired=False, valid=False)
        monkeypatch.setattr("app.gmail.client.Credentials.from_authorized_user_file", lambda *a, **kw: creds)

        with pytest.raises(GmailAuthError):
            GmailClient.from_token_file(token_path, tmp_path / "secret.json")

    def test_auth_error_message_never_contains_token_file_contents(self, tmp_path):
        """Uses the REAL (unmocked) Credentials.from_authorized_user_file:
        a token file missing required fields raises ValueError, which we
        wrap — this confirms the wrapped GmailAuthError message never
        echoes the secret value back, even from a real error path.
        """
        token_path = tmp_path / "token.json"
        secret_value = "super-secret-refresh-token-value-12345"
        token_path.write_text(f'{{"refresh_token": "{secret_value}"}}', encoding="utf-8")

        with pytest.raises(GmailAuthError) as excinfo:
            GmailClient.from_token_file(token_path, tmp_path / "secret.json")

        assert secret_value not in str(excinfo.value)


class TestListMessageIds:
    def test_returns_ids_from_the_service_response(self):
        service = make_fake_service(list_result={"messages": [{"id": "a"}, {"id": "b"}]})
        client = GmailClient.__new__(GmailClient)
        client._service = service

        ids = client.list_message_ids(max_results=10)

        assert ids == ["a", "b"]

    def test_empty_response_returns_empty_list(self):
        service = make_fake_service(list_result={})
        client = GmailClient.__new__(GmailClient)
        client._service = service

        assert client.list_message_ids(max_results=10) == []

    def test_http_error_is_wrapped_in_gmail_api_error(self):
        service = make_fake_service(list_error=make_http_error(429, "rate limited"))
        client = GmailClient.__new__(GmailClient)
        client._service = service

        with pytest.raises(GmailAPIError):
            client.list_message_ids(max_results=10)


class TestGetMessage:
    def test_returns_the_raw_message_dict(self):
        service = make_fake_service(get_results={"msg-1": {"id": "msg-1", "payload": {}}})
        client = GmailClient.__new__(GmailClient)
        client._service = service

        result = client.get_message("msg-1")

        assert result == {"id": "msg-1", "payload": {}}

    def test_http_error_is_wrapped_in_gmail_api_error(self):
        service = make_fake_service(get_errors={"msg-1": make_http_error(403, "permission denied")})
        client = GmailClient.__new__(GmailClient)
        client._service = service

        with pytest.raises(GmailAPIError):
            client.get_message("msg-1")


class FakeGmailClient:
    """A minimal test double satisfying fetch_recent_messages()'s duck-typed
    contract, avoiding the need to mock the full googleapiclient chain for
    tests that are really about fetch_recent_messages' own aggregation
    logic, not GmailClient's internals (covered separately above).
    """

    def __init__(self, ids, messages=None, get_errors=None):
        self._ids = ids
        self._messages = messages or {}
        self._get_errors = get_errors or {}

    def list_message_ids(self, max_results):
        return self._ids[:max_results]

    def get_message(self, message_id):
        if message_id in self._get_errors:
            raise self._get_errors[message_id]
        return self._messages[message_id]


def valid_message(message_id, body="Hello"):
    import base64

    return {
        "id": message_id,
        "internalDate": "1767607200000",
        "payload": {
            "mimeType": "text/plain",
            "headers": [{"name": "From", "value": "a@example.com"}, {"name": "Subject", "value": "Hi"}],
            "body": {"data": base64.urlsafe_b64encode(body.encode()).decode().rstrip("=")},
        },
    }


class TestFetchRecentMessages:
    def test_fetches_and_parses_all_messages(self):
        client = FakeGmailClient(
            ids=["1", "2"],
            messages={"1": valid_message("1", "First"), "2": valid_message("2", "Second")},
        )

        result = fetch_recent_messages(client, max_results=10)

        assert isinstance(result, FetchResult)
        assert len(result.emails) == 2
        assert all(isinstance(e, EmailMessage) for e in result.emails)
        assert result.failed_message_ids == []

    def test_respects_max_results(self):
        client = FakeGmailClient(
            ids=["1", "2", "3"],
            messages={i: valid_message(i) for i in ("1", "2", "3")},
        )

        result = fetch_recent_messages(client, max_results=2)

        assert len(result.emails) == 2

    def test_per_message_api_error_is_recorded_not_raised(self):
        client = FakeGmailClient(
            ids=["1", "2"],
            messages={"2": valid_message("2")},
            get_errors={"1": GmailAPIError("boom")},
        )

        result = fetch_recent_messages(client, max_results=10)

        assert len(result.emails) == 1
        assert result.emails[0].message_id == "2"
        assert len(result.failed_message_ids) == 1
        assert result.failed_message_ids[0][0] == "1"

    def test_per_message_parsing_failure_is_recorded_not_raised(self):
        malformed = {"id": "bad-1", "payload": {"mimeType": "text/plain", "headers": [], "body": {}}}  # no timestamp
        client = FakeGmailClient(ids=["bad-1", "2"], messages={"bad-1": malformed, "2": valid_message("2")})

        result = fetch_recent_messages(client, max_results=10)

        assert len(result.emails) == 1
        assert result.emails[0].message_id == "2"
        assert result.failed_message_ids[0][0] == "bad-1"

    def test_listing_failure_propagates_rather_than_being_swallowed(self):
        class FailingListClient(FakeGmailClient):
            def list_message_ids(self, max_results):
                raise GmailAPIError("listing failed")

        client = FailingListClient(ids=[])

        with pytest.raises(GmailAPIError):
            fetch_recent_messages(client, max_results=10)

    def test_no_messages_gives_empty_fetch_result(self):
        client = FakeGmailClient(ids=[])

        result = fetch_recent_messages(client, max_results=10)

        assert result.emails == []
        assert result.failed_message_ids == []
