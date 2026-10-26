#!/usr/bin/env python3
"""Самопроверка образа без окна: собирает все модели, гоняет физику и рисует по кадру каждого робота.

    sim test                 # итог в консоль, кадры в ./selftest/*.png (на хосте — папка out/selftest)
    sim test --no-render     # только физика (если offscreen-рендер не работает)

Рендер offscreen через EGL (GPU или Mesa); при LIBGL_ALWAYS_SOFTWARE=1 — через OSMesa (чисто CPU).
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa" if os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1" else "egl"
if os.environ["MUJOCO_GL"] == "osmesa":
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import g1_sliders  # noqa: E402
import h2_sliders  # noqa: E402
import robot_sliders  # noqa: E402
import zbot_sliders  # noqa: E402

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, info: str) -> None:
    results.append((name, ok, info))
    print(f"  [{'OK ' if ok else 'ERR'}] {name:<28} {info}", flush=True)


def settle(spec: mujoco.MjSpec, seconds: float) -> tuple[mujoco.MjModel, mujoco.MjData]:
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    lo, hi = m.actuator_ctrlrange.T
    d.ctrl[:] = np.clip(0.3 * (lo + hi) / 2, lo, hi)  # небольшой сдвиг от нуля, чтобы суставам было что догонять
    while d.time < seconds:
        mujoco.mj_step(m, d)
    return m, d


def tracking_deg(m: mujoco.MjModel, d: mujoco.MjData) -> float:
    q = d.qpos[m.jnt_qposadr[m.actuator_trnid[:, 0]]]
    return float(np.degrees(np.abs(q - d.ctrl)).max())


def udp_roundtrip(port: int) -> str:
    """examples/udp_bridge.py без окна + клиент: info -> команда локтю -> сустав пришёл к цели."""
    bridge = subprocess.Popen([sys.executable, str(APP / "examples" / "udp_bridge.py"), "h2", "--headless",
                               "--seconds", "8", "--port", str(port)], stdout=subprocess.DEVNULL)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.5)
        info = None
        for _ in range(30):  # мост поднимается ~1 с
            sock.sendto(b'{"info": true}', ("127.0.0.1", port))
            try:
                while info is None:
                    msg = json.loads(sock.recv(65536))
                    info = msg if "actuators" in msg else None
                break
            except (socket.timeout, ConnectionRefusedError):
                time.sleep(0.3)
        if info is None:
            raise RuntimeError("мост не ответил")
        i = info["actuators"].index("left_elbow")
        sock.sendto(json.dumps({"ctrl": {"left_elbow": 0.8}}).encode(), ("127.0.0.1", port))
        t_end, q = time.time() + 1.5, None
        while time.time() < t_end:
            msg = json.loads(sock.recv(65536))
            q = msg["q"][i] if "q" in msg else q
        if q is None or abs(q - 0.8) > 0.03:
            raise RuntimeError(f"left_elbow = {q}, ждали 0.8")
        return f"{len(info['actuators'])} приводов, left_elbow → {np.degrees(q):.1f}° (цель 45.8°)"
    finally:
        bridge.kill()


def render(m: mujoco.MjModel, d: mujoco.MjData, path: Path) -> str:
    with mujoco.Renderer(m, 540, 720) as r:
        r.update_scene(d)
        img = r.render()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(path, img)
    return f"кадр {path}"


def write_png(path: Path, rgb: np.ndarray) -> None:
    """PNG без сторонних библиотек (zlib + struct)."""
    import struct
    import zlib

    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("selftest"), help="куда класть кадры")
    ap.add_argument("--no-render", action="store_true", help="не рисовать кадры")
    args = ap.parse_args()
    t0 = time.time()
    print(f"MuJoCo {mujoco.__version__}, Python {sys.version.split()[0]}, MUJOCO_GL={os.environ['MUJOCO_GL']}")

    robots = {
        "g1": lambda mode: g1_sliders.build_spec(g1_sliders.DEFAULT_XML, mode == "free", mode == "band", False),
        "h2": lambda mode: h2_sliders.build_spec(h2_sliders.DEFAULT_URDF, mode == "free", mode == "band", False),
        "zbot": lambda mode: zbot_sliders.build_spec(zbot_sliders.DEFAULT_ASSETS, mode == "free", None),
    }
    frames = {}
    for robot, build in robots.items():
        print(f"{robot}:")
        for mode in (["hang", "band", "free"] if robot != "zbot" else ["hang", "free"]):
            try:
                m, d = settle(build(mode), 1.5)
                finite = bool(np.isfinite(d.qpos).all())
                err = tracking_deg(m, d)
                info = f"{m.nu} приводов, {m.body_subtreemass[0]:.1f} кг, ошибка слежения {err:.1f}°"
                ok = finite and (err < 2.0 if mode == "hang" and robot != "zbot" else True)
                check(f"{robot} {mode}", ok, info if finite else "NaN в состоянии")
                if mode == "hang":
                    frames[robot] = (m, d)
            except Exception as e:  # noqa: BLE001 — итог по всем проверкам важнее первой ошибки
                check(f"{robot} {mode}", False, f"{type(e).__name__}: {e}")

    print("robot (универсальный загрузчик):")
    universal = {"H2.urdf": h2_sliders.DEFAULT_URDF, "G1.xml": g1_sliders.DEFAULT_XML,
                 "zbot.mjcf": zbot_sliders.DEFAULT_ASSETS / "robot.mjcf", "h1": "h1", "r1": "r1", "dex3": "dex3"}
    for name, src in universal.items():
        for mode in ("hang", "free"):
            try:
                m, d = settle(robot_sliders.build_spec(robot_sliders.find_model(src), free=mode == "free"), 1.0)
                finite = bool(np.isfinite(d.qpos).all())
                err = tracking_deg(m, d)
                ok = finite and (err < 2.0 if mode == "hang" else True)
                check(f"robot {name} {mode}", ok, f"{m.nu} ползунков, {m.body_subtreemass[0]:.1f} кг"
                      + (f", ошибка слежения {err:.1f}°" if mode == "hang" else "") if finite else "NaN в состоянии")
            except BaseException as e:  # noqa: BLE001 — find_model выходит через SystemExit
                check(f"robot {name} {mode}", False, f"{type(e).__name__}: {e}")

    print("примеры:")
    try:
        r = subprocess.run([sys.executable, str(APP / "examples" / "script_control.py"), "g1", "--band", "--headless",
                            "--seconds", "1"], capture_output=True, text=True, timeout=120)
        check("example script_control", r.returncode == 0,
              f"G1 на подвеске 1 с, строк отчёта: {len(r.stdout.splitlines())}" if r.returncode == 0
              else r.stderr.strip()[-200:])
    except Exception as e:  # noqa: BLE001
        check("example script_control", False, f"{type(e).__name__}: {e}")
    try:
        check("example udp_bridge", True, udp_roundtrip(9871))
    except Exception as e:  # noqa: BLE001
        check("example udp_bridge", False, f"{type(e).__name__}: {e}")

    print("zbot walk:")
    try:
        import zbot_walk

        w = zbot_walk.Walker(zbot_walk.POLICIES / zbot_walk.DEFAULT_POLICY)
        while w.data.time < 4.0:
            w.step()
        check("zbot walk 4 с", w.base_height() > zbot_walk.FALL_HEIGHT,
              f"прошёл {w.data.qpos[0]:.2f} м, высота базы {w.base_height():.3f} м")
    except Exception as e:  # noqa: BLE001
        check("zbot walk 4 с", False, f"{type(e).__name__}: {e}")
    try:
        import zbot_demo

        check("zbot demos", True, ", ".join(p.stem for p in sorted(zbot_demo.DEMOS.glob("*.json"))))
    except Exception as e:  # noqa: BLE001
        check("zbot demos", False, f"{type(e).__name__}: {e}")

    if not args.no_render:
        print("рендер:")
        for robot, (m, d) in frames.items():
            try:
                check(f"{robot} кадр", True, render(m, d, args.out / f"{robot}.png"))
            except Exception as e:  # noqa: BLE001
                check(f"{robot} кадр", False, f"{type(e).__name__}: {e} (попробуйте MUJOCO_GL=osmesa)")

    bad = [n for n, ok, _ in results if not ok]
    print(f"\nИтог: {len(results) - len(bad)}/{len(results)} OK за {time.time() - t0:.0f} с"
          + (f"; ошибки: {', '.join(bad)}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
