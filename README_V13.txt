GOD EYES CLIENT v0.13.0
=======================

Muc tieu cua ban nay:
- Ket noi God Eyes.exe voi God Eyes Server v12.
- Dang nhap tai khoan TEACHER.
- Tai lop hoc tu server.
- Tao server session khi bat dau buoi hoc.
- Tai roster + Face ID READY cua lop.
- Dung YuNet + SFace tren may local de nhan dien hoc sinh.
- Hien thi Student Code + Ho ten thay vi HS 01, HS 02.
- Dong bo observation + evidence len server.
- Heartbeat dinh ky va finish session tren server.
- Van giu local history/evidence de tranh mat du lieu khi mang gap loi.

CAU TRUC FILE
-------------
main.py
engine.py
server_client.py
copy_face_models.ps1
README_V13.txt

MODEL FACE ID
-------------
Server da co 2 model da kiem tra:
- face_detection_yunet_2023mar.onnx
- face_recognition_sface_2021dec_int8.onnx

Tren may Windows cua ban, mo PowerShell tai thu muc client va chay:

powershell -ExecutionPolicy Bypass -File .\copy_face_models.ps1

Script nay copy model tu:
D:\GodEyesServer\data\face_models

sang:
client\assets\models

LUU Y
-----
V13 KHONG luu mat khau server vao file.
server_config.json chi luu Server URL sau khi ket noi thanh cong.
Access token chi nam trong RAM cua app.

CHAY TEST SYNTAX
----------------
python -m py_compile main.py engine.py server_client.py

CHAY APP
--------
Giu cach chay package hien tai cua project God Eyes.
main.py dang dung relative import:
    from .db import Database, EVIDENCE_DIR
    from .engine import CameraWorker, AIWorker
    from .server_client import ServerClient, ServerClientError

Vi vay khong nen chay main.py bang python main.py neu project hien tai chua ho tro package.
Hay thay main.py + engine.py va them server_client.py vao dung thu muc package hien tai.

LUONG V13
---------
START SESSION
 -> Server login
 -> Load classes
 -> Select class
 -> Connect webcam
 -> Create remote session
 -> Load Face ID roster
 -> Local YuNet + SFace recognition
 -> Lock student identities
 -> Live Monitor
 -> Observation + evidence local
 -> Background sync observation + evidence to server
 -> Heartbeat every 20 seconds
 -> Finish remote + local session

CAMERA
------
V13 van dung Webcam USB.
IP Camera / RTSP / ONVIF se them sau vao Camera Manager, khong can thay doi API session hien tai.

KIEM TRA
--------
Da compile-check thanh cong main.py, engine.py va server_client.py.
Chua test end-to-end voi Windows webcam/server cua ban trong moi truong nay.
