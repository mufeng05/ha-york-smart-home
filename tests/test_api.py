"""HTTP level tests for the York cloud client."""

from __future__ import annotations

import asyncio

import aiohttp
import pytest
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
    AiohttpClientMockResponse,
)
from yarl import URL

from custom_components.york_smart_home.api import (
    Session,
    YorkApi,
    YorkApiError,
    YorkAuthError,
    YorkClientOutdatedError,
    YorkCommandError,
    YorkConnectionError,
    YorkLoginError,
    YorkRateLimitError,
    YorkSessionConflictError,
    YorkSessionTakenError,
    is_on,
)
from custom_components.york_smart_home.const import API_BASE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .conftest import CODE, FIXTURE, PHONE

OK = {"code": "200", "message": "处理成功！"}
UNAUTHORIZED = {"code": "401", "message": "未授权用户！"}
DEVICE_LIST = f"{API_BASE}/api/apphome/homes/homeList/deviceList"
EMPTY_LIST = {**OK, "data": {"allHomeDeviceList": []}}


def _login_body(token: str = "tok2", user_id: int = 200002) -> dict:
    return {
        **OK,
        "data": {
            "user": {"userId": user_id, "token": token, "refreshToken": "rt2", "version": 1},
            "homeList": [],
        },
    }


def _api(hass, **kwargs) -> YorkApi:
    kwargs.setdefault("client_id", "cid")
    return YorkApi(async_get_clientsession(hass), PHONE, **kwargs)


async def _async_resp(body: dict, status: int = 200, headers: dict | None = None):
    return AiohttpClientMockResponse(
        "GET", URL(API_BASE), json=body, status=status, headers=headers
    )


def _resp(body: dict, status: int = 200, headers: dict | None = None):
    """Awaitable response for AiohttpClientMocker side effects."""
    return _async_resp(body, status, headers)


def _sequence(*bodies: dict):
    it = iter(bodies)
    return lambda *args, **kwargs: _resp(next(it))


def test_is_on() -> None:
    assert all(is_on(v) for v in ("1", 1, True))
    assert not any(is_on(v) for v in ("0", 0, False, None, "true", "", 2))


# ---------------------------------------------------------------------- login


async def test_password_login(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> None:
    aioclient_mock.get(f"{API_BASE}/login", json=_login_body())
    sessions = []
    api = _api(hass, password="pw", on_session=sessions.append)
    auth = await api.login_password()
    assert auth == Session("tok2", 200002, "rt2", 1)
    assert sessions == [auth]
    url = aioclient_mock.mock_calls[0][1]
    assert url.query["jgRegId"] == "cid"  # never empty: the server rejects that
    assert url.query["password"] == "pw"


async def test_login_rejected_keeps_server_message(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.get(f"{API_BASE}/login", json={"code": "500", "message": "密码错误"})
    with pytest.raises(YorkLoginError) as info:
        await _api(hass, password="bad").login_password()
    assert info.value.message == "密码错误"


async def test_password_not_leaked(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.get(f"{API_BASE}/login", exc=aiohttp.ClientConnectionError("boom"))
    with pytest.raises(YorkConnectionError) as info:
        await _api(hass, password="secret-pw").login_password()
    assert "secret-pw" not in str(info.value)
    assert info.value.__cause__ is None and info.value.__suppress_context__


async def test_relogin_on_401(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> None:
    aioclient_mock.get(DEVICE_LIST, side_effect=_sequence(UNAUTHORIZED, EMPTY_LIST))
    aioclient_mock.get(f"{API_BASE}/login", json=_login_body("fresh"))
    api = _api(hass, password="pw", auth=Session("old", 200002), reclaim_session=True)
    assert await api.get_wire_controllers() == []
    assert api.auth.token == "fresh"
    # the retried request carried the new token
    assert aioclient_mock.mock_calls[-1][3]["Authorization"] == "Bearer fresh"


async def test_non_json_401_relogins(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    calls = iter([401, 200])

    def respond(*args, **kwargs):
        status = next(calls)
        return _resp(EMPTY_LIST if status == 200 else {}, status)

    aioclient_mock.get(DEVICE_LIST, side_effect=respond)
    aioclient_mock.get(f"{API_BASE}/login", json=_login_body("fresh"))
    api = _api(hass, password="pw", auth=Session("old", 200002), reclaim_session=True)
    assert await api.get_wire_controllers() == []


async def test_concurrent_401_single_login(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    def respond(method, url, data):
        headers = aioclient_mock.mock_calls[-1][3]
        return _resp(UNAUTHORIZED if headers["Authorization"] == "Bearer old" else EMPTY_LIST)

    aioclient_mock.get(DEVICE_LIST, side_effect=respond)
    aioclient_mock.get(f"{API_BASE}/login", json=_login_body("fresh"))
    api = _api(hass, password="pw", auth=Session("old", 200002), reclaim_session=True)
    await asyncio.gather(api.get_wire_controllers(), api.get_wire_controllers())
    logins = [c for c in aioclient_mock.mock_calls if c[1].path == "/login"]
    assert len(logins) == 1


async def test_sms_session_renews_with_refresh_token(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.get(DEVICE_LIST, side_effect=_sequence(UNAUTHORIZED, EMPTY_LIST))
    aioclient_mock.get(f"{API_BASE}/smsLogin", json=_login_body("fresh"))
    api = _api(
        hass,
        auth=Session("old", 200002, refresh_token="rt", login_version=3),
        reclaim_session=True,
    )
    await api.get_wire_controllers()
    query = next(c for c in aioclient_mock.mock_calls if c[1].path == "/smsLogin")[1].query
    assert query["refreshToken"] == "rt"
    assert query["version"] == "3"
    assert "verifyCode" not in query
    assert api.auth.refresh_token == "rt2"


async def test_session_conflict_stops_relogin(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """A renewed session revoked again soon after means another client logged in."""
    aioclient_mock.get(DEVICE_LIST, side_effect=_sequence(UNAUTHORIZED, EMPTY_LIST, UNAUTHORIZED))
    aioclient_mock.get(f"{API_BASE}/login", json=_login_body("fresh"))
    api = _api(hass, password="pw", auth=Session("old", 200002), reclaim_session=True)
    await api.get_wire_controllers()
    with pytest.raises(YorkSessionConflictError):
        await api.get_wire_controllers()
    logins = [c for c in aioclient_mock.mock_calls if c[1].path == "/login"]
    assert len(logins) == 1


async def test_session_taken_yields_by_default(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """Like the app (LOGIN_ELSEWHERE), a 401 does not trigger a new login."""
    aioclient_mock.get(DEVICE_LIST, json=UNAUTHORIZED)
    aioclient_mock.get(f"{API_BASE}/login", json=_login_body("fresh"))
    api = _api(hass, password="pw", auth=Session("old", 200002))
    with pytest.raises(YorkSessionTakenError):
        await api.get_wire_controllers()
    assert not [c for c in aioclient_mock.mock_calls if c[1].path == "/login"]
    assert api.auth.token == "old"


async def test_http_401_is_session_taken(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.get(DEVICE_LIST, side_effect=lambda *a, **k: _resp({}, 401))
    with pytest.raises(YorkSessionTakenError):
        await _api(hass, password="pw", auth=Session("old", 200002)).get_wire_controllers()


async def test_control_reclaims_session(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """A command is explicit intent: log in again right away, even after a renewal."""
    aioclient_mock.post(CTRL_URL, side_effect=_sequence(UNAUTHORIZED, OK, UNAUTHORIZED, OK))
    aioclient_mock.get(f"{API_BASE}/login", json=_login_body("fresh"))
    api = _api(hass, password="pw", auth=Session("old", 200002))
    await api.control(100001, CODE, "C", "9", {"if_run": "1"})
    assert api.auth.token == "fresh"
    # A second takeover right after is not treated as ping-pong for commands.
    await api.control(100001, CODE, "C", "9", {"if_run": "0"})
    logins = [c for c in aioclient_mock.mock_calls if c[1].path == "/login"]
    assert len(logins) == 2


async def test_relogin(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> None:
    aioclient_mock.get(f"{API_BASE}/smsLogin", json=_login_body("fresh"))
    api = _api(hass, auth=Session("old", 200002, refresh_token="rt"))
    assert (await api.relogin()).token == "fresh"
    assert aioclient_mock.mock_calls[0][1].query["refreshToken"] == "rt"


async def test_concurrent_relogin_single_login(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    async def slow_login(*args, **kwargs):
        await asyncio.sleep(0)  # let the second caller start meanwhile
        return await _async_resp(_login_body("fresh"))

    aioclient_mock.get(f"{API_BASE}/login", side_effect=slow_login)
    api = _api(hass, password="pw", auth=Session("old", 200002))
    await asyncio.gather(api.relogin(), api.relogin())
    assert len(aioclient_mock.mock_calls) == 1


async def test_login_endpoint_401_is_auth_failure(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.get(f"{API_BASE}/login", json=UNAUTHORIZED)
    with pytest.raises(YorkAuthError) as info:
        await _api(hass, password="pw").login_password()
    assert not isinstance(info.value, YorkSessionTakenError)


async def test_password_login_keeps_refresh_token(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    body = _login_body("fresh")
    del body["data"]["user"]["refreshToken"]
    aioclient_mock.get(f"{API_BASE}/login", json=body)
    api = _api(hass, password="pw", auth=Session("old", 200002, refresh_token="rt"))
    assert (await api.login_password()).refresh_token == "rt"


async def test_force_update_code(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> None:
    aioclient_mock.get(
        DEVICE_LIST,
        json={
            "code": 300,
            "message": "请更新",
            "data": {"version": "1.9.0", "url": "https://example.invalid/app.apk"},
        },
    )
    with pytest.raises(YorkClientOutdatedError) as info:
        await _api(hass, auth=Session("tok", 200002)).get_wire_controllers()
    assert info.value.version == "1.9.0"
    assert info.value.url == "https://example.invalid/app.apk"


async def test_wire_controller_detail(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    url = f"{API_BASE}/api/appdevice/homes/100001/xkqs/{CODE}"
    aioclient_mock.get(url, json={**OK, "data": {"xkqCode": CODE, "constantJson": {"B9": "1"}}})
    api = _api(hass, auth=Session("tok", 200002))
    assert await api.get_wire_controller_detail(100001, CODE) == {"B9": "1"}

    aioclient_mock.clear_requests()
    aioclient_mock.get(url, json={**OK, "data": {"constantJson": '{"B9": "0"}'}})
    assert await api.get_wire_controller_detail(100001, CODE) == {"B9": "0"}

    aioclient_mock.clear_requests()
    aioclient_mock.get(url, json={**OK, "data": None})
    with pytest.raises(YorkApiError):
        await api.get_wire_controller_detail(100001, CODE)


async def test_403_still_renews(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> None:
    aioclient_mock.get(
        DEVICE_LIST, side_effect=_sequence({"code": "403", "message": "x"}, EMPTY_LIST)
    )
    aioclient_mock.get(f"{API_BASE}/login", json=_login_body("fresh"))
    api = _api(hass, password="pw", auth=Session("old", 200002))
    assert await api.get_wire_controllers() == []
    assert api.auth.token == "fresh"


async def test_renewal_for_other_account_rejected(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.get(f"{API_BASE}/login", json=_login_body(user_id=1))
    api = _api(hass, password="pw", auth=Session("old", 200002))
    with pytest.raises(YorkAuthError):
        await api.login_password()


async def test_rate_limit(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> None:
    aioclient_mock.get(
        DEVICE_LIST, side_effect=lambda *a, **k: _resp({}, 429, {"Retry-After": "120"})
    )
    with pytest.raises(YorkRateLimitError) as info:
        await _api(hass, auth=Session("tok", 200002)).get_wire_controllers()
    assert info.value.retry_after == 120


# -------------------------------------------------------------------- polling


async def test_discovery_and_status(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    other_brand = {**FIXTURE["device"], "xkqCode": "other", "brand": "1"}
    other_type = {**FIXTURE["device"], "xkqCode": "other2", "xkqType": "5"}
    aioclient_mock.get(
        DEVICE_LIST,
        json={
            **OK,
            "data": {
                "allHomeDeviceList": [
                    {
                        "homeId": 100001,
                        "homeName": "家",
                        "deviceList": [{"iezCode": "gateway"}],
                        "xkqList": [FIXTURE["device"], other_brand, other_type],
                    }
                ]
            },
        },
    )
    aioclient_mock.post(
        f"{API_BASE}/api/appstatus/homes/100001/status",
        json={**OK, "data": {"xkqStatusListGjy": [FIXTURE["status"]]}},
    )
    aioclient_mock.post(
        f"{API_BASE}/api/appcontrol/cmds/xkqOnlineStatus",
        json={**OK, "data": [{"xkqCode": CODE, "online": 1}]},
    )
    api = _api(hass, auth=Session("tok", 200002))
    devices = await api.get_wire_controllers()
    assert [(d["xkqCode"], d["homeId"]) for d in devices] == [(CODE, 100001)]
    assert api.unsupported_count == 3
    status = await api.get_status(100001, [(CODE, 0)])
    assert status[CODE]["if_run"] == "1"
    assert await api.get_online([(CODE, "C")]) == {CODE: True}


ONLINE_STATUS = f"{API_BASE}/api/appcontrol/cmds/xkqOnlineStatus"
OFFLINE = "0000000000000002"


async def test_online_some_offline(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """Code 209 names the offline controllers but still lists every flag."""
    aioclient_mock.post(
        ONLINE_STATUS,
        json={
            "code": "209",
            "message": "此线控器设备【0002】连接网络失败，请保证设备在线。",
            "data": [
                {"xkqCode": OFFLINE, "online": False, "gjyCode": 200},
                {"xkqCode": CODE, "online": True, "gjyCode": 200},
            ],
        },
    )
    api = _api(hass, auth=Session("tok", 200002))
    assert await api.get_online([(OFFLINE, "C"), (CODE, "C")]) == {OFFLINE: False, CODE: True}


@pytest.mark.parametrize(
    "body",
    [
        {"code": "209", "message": "m"},
        {"code": "209", "message": "m", "data": None},
        {"code": "209", "message": "m", "data": []},
        {"code": "209", "message": "m", "data": {"xkqCode": CODE, "online": True}},
        {"code": "209", "message": "m", "data": [{"xkqCode": [], "online": True}]},
        {"code": "209", "message": "m", "data": [{"xkqCode": CODE, "online": True}, "x"]},
        {"code": "400", "message": "参数有误。", "data": {"description": "xkqCtrlList error!"}},
    ],
)
async def test_online_unreadable_answer(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, body: dict
) -> None:
    """Any other code, or a 209 without a readable list, still fails."""
    aioclient_mock.post(ONLINE_STATUS, json=body)
    api = _api(hass, auth=Session("tok", 200002))
    with pytest.raises(YorkApiError, match=f"code {body['code']}"):
        await api.get_online([(CODE, "C")])


# -------------------------------------------------------------------- control

CTRL_URL = f"{API_BASE}/api/appcontrol/cmds/multiIuOuCtrl"


async def _control(hass):
    api = _api(hass, auth=Session("tok", 200002))
    await api.control(100001, CODE, "C", "9", {"set_temp": 28, "set_temperature_add05": "0"})


async def test_control_like_the_app(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.post(
        CTRL_URL,
        json={**OK, "data": [{"iezCode": CODE, "hiiotToken": "h1", "hiiotErrorCode": 200}]},
    )
    await _control(hass)
    assert aioclient_mock.mock_calls[0][2] == {
        "homeId": 100001,
        "ctrlType": "YO420",
        "ctrlList": [],
        "xkqCtrlList": [
            {
                "xkqCode": CODE,
                "deviceType": "C",
                "xkqType": "9",
                "ctrlJson": {"set_temp": 28, "set_temperature_add05": "0"},
            }
        ],
    }
    # The control page does not poll multiCtrlResult.
    assert len(aioclient_mock.mock_calls) == 1


async def test_control_failure_message(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.post(CTRL_URL, json={"code": "500", "message": "空调不在线"})
    with pytest.raises(YorkCommandError, match="空调不在线"):
        await _control(hass)


async def test_control_403_still_renews(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """A command keeps renewing the session on a 403 (only scene actions take it as a refusal)."""
    aioclient_mock.post(CTRL_URL, side_effect=_sequence({"code": "403", "message": "x"}, OK))
    aioclient_mock.get(f"{API_BASE}/login", json=_login_body("fresh"))
    api = _api(hass, password="pw", auth=Session("old", 200002))
    await api.control(100001, CODE, "C", "9", {"if_run": "1"})
    assert api.auth.token == "fresh"
    assert [call[1].path for call in aioclient_mock.mock_calls] == [
        "/api/appcontrol/cmds/multiIuOuCtrl",
        "/login",
        "/api/appcontrol/cmds/multiIuOuCtrl",
    ]


async def test_status_request_is_incremental(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.post(
        f"{API_BASE}/api/appstatus/homes/100001/status",
        json={**OK, "data": {"xkqStatusListGjy": []}},
    )
    api = _api(hass, auth=Session("tok", 200002))
    assert await api.get_status(100001, [(CODE, 1790763654000)]) == {}
    assert aioclient_mock.mock_calls[0][2]["xkqListGjy"] == [
        {"xkqCode": CODE, "timestamp": 1790763654000}
    ]


async def test_password_renewal_sends_last_login_version(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.get(f"{API_BASE}/login", json=_login_body())
    api = _api(hass, password="pw", auth=Session("old", 200002, login_version=4))
    await api.login_password()
    assert aioclient_mock.mock_calls[0][1].query["version"] == "4"
