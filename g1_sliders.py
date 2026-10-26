#!/usr/bin/env python3
"""Unitree G1 (29 суставов тела + кисти Dex3) в MuJoCo с ползунками для 43 суставов.

Модель — g1_body29_hand14.xml из xr_teleoperate (assets/g1: MJCF + STL-меши, 34 кг; это g1_29dof_with_hand_rev_1_0
из unitree_ros). Её torque-моторы заменяются сервоприводами позиционного типа:
ползунок = целевой угол сустава (рад), ctrlrange = пределы сустава.
    τ = kp·(q_target − q) − kv·dq,  clip ±actuatorfrcrange
kp — штатные коэффициенты Unitree: тело — unitree_sdk2_python/example/g1/low_level/g1_low_level_example.py,
пальцы Dex3 — xr_teleoperate/teleop/robot_control/robot_hand_unitree.py (kp 1.5, kd 0.2).
Инерция роторов (armature) — как в mujoco_playground (locomotion/g1/xmls), для пальцев — оценка.
По умолчанию kv ≈ критическое демпфирование (--unitree-kd — взять Kd как есть).
В режиме подвеса включена компенсация силы тяжести через приводы, иначе с kp 40 поясница (waist_pitch) под весом
торса проседает на ~22° (--no-gravcomp — увидеть провисание).
Самостолкновения выключены (как в h2_sliders.py); с полом сталкиваются все звенья.

Ползунки: правая панель viewer'а -> раздел "Control". Сброс в исходную позу: Backspace. Пауза: пробел.

Запуск:
    .venv/bin/python g1_sliders.py            # робот подвешен (таз закреплён), ноги не касаются пола
    .venv/bin/python g1_sliders.py --free     # робот на полу; без контроллера баланса падает
    .venv/bin/python g1_sliders.py --band     # робот на полу на мягкой подвеске (выключается в панели Equality)
Папку assets репозитория xr_teleoperate можно задать переменной окружения XR_TELEOP_ASSETS.
"""
import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from h2_sliders import XR_ASSETS, add_band, add_gravcomp, add_scene

DEFAULT_XML = XR_ASSETS / "g1" / "g1_body29_hand14.xml"
HANG_HEIGHT = 0.12  # на сколько поднять таз в режиме подвеса, м
BAND_DROP = 0.0015  # с подвеской таз ниже на 1.5 мм: стопы стоят на полу и несут около 2/3 веса

# Штатные Kp/Kd Unitree G1 (порядок моторов SDK = порядок суставов тела ниже).
LEG_GAINS = {"hip_pitch": (60, 1), "hip_roll": (60, 1), "hip_yaw": (60, 1), "knee": (100, 2),
             "ankle_pitch": (40, 1), "ankle_roll": (40, 1)}
UNITREE_GAINS = {
    **{f"{side}_{name}_joint": g for side in ("left", "right") for name, g in LEG_GAINS.items()},
    "waist_yaw_joint": (60, 1), "waist_roll_joint": (40, 1), "waist_pitch_joint": (40, 1),
    **{f"{side}_{name}_joint": (40, 1) for side in ("left", "right")
       for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                    "wrist_roll", "wrist_pitch", "wrist_yaw")},
}
HAND_GAINS = (1.5, 0.2)  # все 7 моторов кисти Dex3

# Инерция ротора, приведённая к суставу, по моменту мотора (mujoco_playground g1_mjx_feetonly.xml).
ARMATURE = {139: 0.025101925, 88: 0.01017752, 50: 0.00721945, 25: 0.003609725, 5: 0.00425}
HAND_ARMATURE = 0.001  # Dex3: данных нет, оценка


def gains_for(joint: str) -> tuple[float, float]:
    return HAND_GAINS if "_hand_" in joint else UNITREE_GAINS[joint]


def build_spec(xml: Path, free: bool, band: bool, unitree_kd: bool, gravcomp: bool = True) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.meshdir = str(xml.parent / spec.meshdir)  # абсолютный путь, чтобы --save-xml открывался из любой папки
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # демпфирование сервопривода считается неявно

    # Штатные torque-моторы модели (ctrl = момент) заменяются сервоприводами ниже.
    for act in list(spec.actuators):
        spec.delete(act)
    # Свои пол, свет и чёрное небо модели заменить общей сценой (как у H2).
    spec.delete(spec.geom("floor"))
    for item in [*spec.lights, *spec.materials, *spec.textures]:
        spec.delete(item)
    add_scene(spec)
    spec.visual.headlight.specular = [0.3, 0.3, 0.3]  # в модели 0.9 — сильные блики

    pelvis = spec.body("pelvis")
    if not (free or band):
        spec.delete(spec.joint("floating_base_joint"))  # таз жёстко прикреплён к миру — робот «висит на стенде»
        pelvis.pos = [pelvis.pos[0], pelvis.pos[1], pelvis.pos[2] + HANG_HEIGHT]
        if gravcomp:
            add_gravcomp(spec)
    if band:
        pelvis.pos = [pelvis.pos[0], pelvis.pos[1], pelvis.pos[2] - BAND_DROP]
        add_band(spec, "pelvis")

    # Геомы робота: коллизии (group 0) скрыть в group 3 и сталкивать только с полом (conaffinity=0 => не друг с другом).
    for g in spec.geoms:
        if g.name == "floor":
            continue
        if g.contype:
            g.group = 3
            g.conaffinity = 0

    # Стартовая камера viewer'а: спереди-сбоку, робот на весь кадр.
    spec.visual.global_.azimuth = 150
    spec.visual.global_.elevation = -12
    spec.stat.center = [0, 0, pelvis.pos[2] - 0.15]
    spec.stat.extent = 1.1

    joints = [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]
    for j in joints:
        effort = j.actfrcrange[1]
        j.armature = HAND_ARMATURE if "_hand_" in j.name else ARMATURE[round(effort)]
        j.damping = [0.05 if "_hand_" in j.name else 0.5, 0, 0]  # в MuJoCo 3.15 damping — вектор
        j.frictionloss = 0.01 if "_hand_" in j.name else 0.1
        kp, kd = gains_for(j.name)
        act = spec.add_actuator(name=j.name.removesuffix("_joint"), target=j.name,
                                trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=kp, kv=kd)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = j.range
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-effort, effort]

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if not unitree_kd:
        # kv = 2·sqrt(kp·M_ii): M_ii — диагональ матрицы масс в исходной позе (с armature), т.е. ≈ критическое демпфирование.
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, mass)  # в 3.15 сигнатура (m, d, dst)
        for act in spec.actuators:
            dof = model.joint(act.target).dofadr[0]
            kp, kd = gains_for(act.target)
            act.set_to_position(kp=kp, kv=max(kd, 2.0 * np.sqrt(kp * mass[dof, dof])))

    ctrl0 = np.clip(np.zeros(model.nu), model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    spec.add_key(name="home", qpos=model.qpos0.copy(), ctrl=ctrl0)
    return spec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", type=Path, default=DEFAULT_XML, help="g1_body29_hand14.xml (рядом папка meshes/)")
    ap.add_argument("--free", action="store_true", help="свободная база: робот на полу (без контроллера баланса падает)")
    ap.add_argument("--band", action="store_true", help="на полу, но таз держит мягкая подвеска (панель Equality)")
    ap.add_argument("--unitree-kd", action="store_true", help="штатные Kd Unitree вместо критического демпфирования")
    ap.add_argument("--no-gravcomp", action="store_true", help="без компенсации силы тяжести в режиме подвеса")
    ap.add_argument("--save-xml", type=Path, default=None, help="сохранить собранную сцену в MJCF и выйти")
    args = ap.parse_args()

    spec = build_spec(args.xml.resolve(), args.free, args.band, args.unitree_kd, not args.no_gravcomp)
    model = spec.compile()
    if args.save_xml:
        args.save_xml.write_text(spec.to_xml())
        print("saved", args.save_xml)
        return

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    print(f"G1: {model.nu} суставов, {model.body_subtreemass[0]:.1f} кг. "
          "Ползунки: правая панель -> Control. Backspace — сброс позы, пробел — пауза.")
    for i in range(model.nu):
        lo, hi = np.degrees(model.actuator_ctrlrange[i])
        print(f"  {model.actuator(i).name:<22} {lo:7.1f}° .. {hi:6.1f}°   kp {model.actuator_gainprm[i, 0]:5.1f}"
              f"  kv {-model.actuator_biasprm[i, 2]:5.2f}   max {model.actuator_forcerange[i, 1]:6.1f} Н·м")

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
