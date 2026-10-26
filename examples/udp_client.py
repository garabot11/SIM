#!/usr/bin/env python3
"""Клиент к udp_bridge.py: качает один сустав синусом и печатает, что происходит с роботом.

Только стандартная библиотека Python 3 — запускается прямо на хосте, MuJoCo не нужен:
    python3 examples/udp_client.py                          # список приводов и качание первого
    python3 examples/udp_client.py --joint left_elbow --amp 0.6 --freq 0.5
    python3 examples/udp_client.py --host 192.168.1.20      # симулятор на другом компьютере
"""
import argparse
import json
import math
import socket
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="адрес симулятора")
    ap.add_argument("--port", type=int, default=9870)
    ap.add_argument("--joint", help="имя привода (по умолчанию первый)")
    ap.add_argument("--amp", type=float, default=0.5, help="амплитуда, рад")
    ap.add_argument("--freq", type=float, default=0.5, help="частота, Гц")
    ap.add_argument("--seconds", type=float, default=float("inf"))
    args = ap.parse_args()

    sim = (args.host, args.port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    sock.sendto(json.dumps({"info": True}).encode(), sim)
    while True:  # первым может прийти пакет состояния — ждём ответ на info
        try:
            info = json.loads(sock.recv(65536))
        except socket.timeout:
            raise SystemExit(f"нет ответа от {args.host}:{args.port}/udp — запущен ли udp_bridge (и с NET=host)?")
        if "actuators" in info:
            break
    names = info["actuators"]
    print(f"робот {info['robot']}, приводы: {', '.join(names)}")
    i = names.index(args.joint) if args.joint else 0
    lo, hi = info["ctrlrange"][i]
    center = min(max(0.0, lo), hi)

    sock.setblocking(False)
    start = time.time()
    last_print = last_state = 0.0
    while time.time() - start < args.seconds:
        t = time.time() - start
        target = min(max(center + args.amp * math.sin(2 * math.pi * args.freq * t), lo), hi)
        sock.sendto(json.dumps({"ctrl": {names[i]: target}}).encode(), sim)
        time.sleep(0.02)  # команды ~50 Гц
        state = None
        while True:  # забрать всё, что пришло, оставить последнее состояние
            try:
                packet = json.loads(sock.recv(65536))
            except BlockingIOError:
                break
            if "q" in packet:
                state = packet
        if state is None:
            if t - last_state > 2.0:
                print("нет состояния от симулятора")
                last_state = t
            continue
        last_state = t
        if t - last_print >= 0.5:
            last_print = t
            print(f"t_sim={state['t']:7.2f}  {names[i]}: цель {math.degrees(target):7.1f}°  "
                  f"факт {math.degrees(state['q'][i]):7.1f}°  момент {state['tau'][i]:7.2f} Н·м  "
                  f"база z={state['base_pos'][2]:.3f} м", flush=True)


if __name__ == "__main__":
    main()
