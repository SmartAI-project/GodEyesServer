from __future__ import annotations

import json
import uuid
from pathlib import Path
from urllib import error, parse, request


class ServerClientError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class ServerClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 12.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.access_token = ""
        self.profile = None

    @property
    def connected(self) -> bool:
        return bool(self.access_token)

    def set_base_url(self, base_url: str):
        value = (base_url or "").strip().rstrip("/")
        if not value:
            raise ServerClientError("Server URL is empty.")
        self.base_url = value

    def save_config(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"server_url": self.base_url}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_config(cls, path: Path):
        client = cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("server_url"):
                client.base_url = str(data["server_url"]).strip().rstrip("/")
        except Exception:
            pass
        return client

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"

    @staticmethod
    def _json_bytes(data) -> bytes:
        return json.dumps(data, ensure_ascii=False).encode("utf-8")

    def _request(self, method: str, path: str, body: bytes | None = None, content_type: str = "application/json"):
        headers = {"Accept": "application/json", "User-Agent": "GodEyesClient/0.13"}
        if body is not None:
            headers["Content-Type"] = content_type
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        req = request.Request(self._url(path), data=body, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read()
                content_type_header = response.headers.get("Content-Type", "")
                if "application/json" in content_type_header or raw[:1] in (b"{", b"["):
                    try:
                        return json.loads(raw.decode("utf-8"))
                    except Exception:
                        return {}
                return raw
        except error.HTTPError as exc:
            raw = exc.read()
            message = f"Server returned HTTP {exc.code}."
            try:
                data = json.loads(raw.decode("utf-8"))
                detail = data.get("detail") if isinstance(data, dict) else None
                if detail:
                    message = str(detail)
            except Exception:
                pass
            if exc.code in (401, 403):
                self.access_token = ""
                self.profile = None
            raise ServerClientError(message, exc.code) from exc
        except error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise ServerClientError(f"Không kết nối được God Eyes Server: {reason}") from exc
        except TimeoutError as exc:
            raise ServerClientError("Kết nối God Eyes Server quá thời gian cho phép.") from exc

    def login(self, username: str, password: str) -> dict:
        payload = {"username": str(username).strip(), "password": password}
        data = self._request("POST", "/auth/login", self._json_bytes(payload))
        if not isinstance(data, dict) or not data.get("access_token"):
            raise ServerClientError("Server không trả về access token hợp lệ.")
        if str(data.get("role", "")).upper() != "TEACHER":
            raise ServerClientError("God Eyes Client chỉ cho phép tài khoản giáo viên.")
        self.access_token = str(data["access_token"])
        self.profile = self.get_me()
        return self.profile

    def logout(self):
        try:
            self._request("POST", "/auth/logout")
        except Exception:
            pass
        self.access_token = ""
        self.profile = None

    def get_me(self) -> dict:
        data = self._request("GET", "/api/v1/me")
        if not isinstance(data, dict):
            raise ServerClientError("Phản hồi tài khoản không hợp lệ.")
        self.profile = data
        return data

    def get_classes(self) -> list[dict]:
        data = self._request("GET", "/api/v1/classes")
        items = data.get("items") if isinstance(data, dict) else None
        return list(items or [])

    def get_roster(self, class_id: int) -> dict:
        data = self._request("GET", f"/api/v1/classes/{int(class_id)}/face-roster")
        if not isinstance(data, dict):
            raise ServerClientError("Danh sách học sinh không hợp lệ.")
        return data

    def create_session(self, class_id: int, client_version: str, camera_type: str = "WEBCAM") -> dict:
        payload = {
            "class_id": int(class_id),
            "client_version": str(client_version)[:40],
            "camera_type": str(camera_type).upper()[:40],
        }
        data = self._request("POST", "/api/v1/sessions", self._json_bytes(payload))
        if not isinstance(data, dict) or not data.get("session_id"):
            raise ServerClientError("Server không tạo được session.")
        return data

    def post_events(self, session_id: int, events: list[dict]) -> dict:
        if not events:
            return {"inserted": 0, "ids": []}
        data = self._request(
            "POST",
            f"/api/v1/sessions/{int(session_id)}/events",
            self._json_bytes({"events": events[:200]}),
        )
        if not isinstance(data, dict):
            raise ServerClientError("Phản hồi event không hợp lệ.")
        return data

    def upload_evidence(self, session_id: int, student_id: int, event_type: str, confidence: float, captured_at: str, image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
        boundary = f"----GodEyes{uuid.uuid4().hex}"
        parts = []

        def field(name: str, value):
            parts.append(f"--{boundary}\r\n".encode())
            parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            parts.append(str(value).encode())
            parts.append(b"\r\n")

        field("student_id", int(student_id))
        field("event_type", str(event_type))
        field("confidence", float(confidence))
        field("captured_at", str(captured_at))

        filename = "evidence.jpg" if mime_type == "image/jpeg" else "evidence.png"
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="evidence"; filename="{filename}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n".encode()
        )
        parts.append(bytes(image_bytes))
        parts.append(b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())

        body = b"".join(parts)
        data = self._request(
            "POST",
            f"/api/v1/sessions/{int(session_id)}/evidence",
            body,
            f"multipart/form-data; boundary={boundary}",
        )
        if not isinstance(data, dict) or not data.get("id"):
            raise ServerClientError("Server không lưu được evidence.")
        return data

    def heartbeat(self, session_id: int) -> dict:
        data = self._request("POST", f"/api/v1/sessions/{int(session_id)}/heartbeat", b"{}")
        if not isinstance(data, dict):
            raise ServerClientError("Phản hồi heartbeat không hợp lệ.")
        return data

    def finish_session(self, session_id: int, duration_seconds: int) -> dict:
        payload = {"duration_seconds": max(0, int(duration_seconds))}
        data = self._request("POST", f"/api/v1/sessions/{int(session_id)}/finish", self._json_bytes(payload))
        if not isinstance(data, dict):
            raise ServerClientError("Phản hồi kết thúc session không hợp lệ.")
        return data

    def get_history(self, limit: int = 50) -> list[dict]:
        query = parse.urlencode({"limit": max(1, min(100, int(limit)))})
        data = self._request("GET", f"/api/v1/history?{query}")
        return list(data.get("items") or []) if isinstance(data, dict) else []
