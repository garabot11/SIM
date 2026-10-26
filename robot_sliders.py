#!/usr/bin/env python3
"""Любой робот (URDF или MJCF) в MuJoCo с ползунками для всех суставов.

Берёт файл модели, добавляет пол и свет (если их нет) и ставит на каждый сустав (hinge/slide) сервопривод
позиционного типа: ползунок = целевой угол (рад) или смещение (м), ctrlrange = пределы сустава.
    τ = kp·(q_target − q) − kv·dq,  clip ±effort
Коэффициенты подбираются сами, если не заданы: kp = max(M_ii·(2π·freq)², G_i / 0.05 рад), где M_ii — инерция,
которую «видит» сустав, G_i — наибольший момент силы тяжести на нём (провис не больше ~3°), kv — критическое
демпфирование 2·sqrt(kp·M_ii). Предельный момент — effort из URDF, actuatorfrcrange или ctrlrange моторов модели.
Если у модели уже есть сервоприводы позиционного типа (например, у манипуляторов из mujoco_menagerie), они
остаются как есть (--actuators keep/replace — явно).

Режимы базы:
    (по умолчанию)  подвес: корень робота закреплён, ноги не достают до пола; включена компенсация силы тяжести
    --free          свободная база на полу (если у модели её нет — добавляется freejoint)
    --band          на полу, корень держит мягкая подвеска (галочка band в панели Equality)
Закреплённые модели (манипуляторы без freejoint) в подвесе остаются там, где их поставил автор модели.

Ползунки: правая панель viewer'а -> раздел "Control". Сброс в исходную позу: Backspace. Пауза: пробел.

Запуск:
    python3 robot_sliders.py models/my_robot/robot.urdf
    python3 robot_sliders.py mujoco_menagerie/unitree_go2/scene.xml --free
    python3 robot_sliders.py robot.urdf --kp 50 --effort 20 --info     # только напечатать таблицу и выйти
    python3 robot_sliders.py robot.urdf --gains gains.json             # свои kp/kv/effort/armature по суставам
gains.json: {"регулярное выражение по имени сустава": {"kp": 100, "kv": 2, "effort": 50, "armature": 0.01}, ...};
первое совпавшее выражение побеждает, любые поля можно опустить.
"""
import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from h2_sliders import BAND_SOLREF, add_scene

HANG_CLEARANCE = 0.10  # в подвесе нижняя точка робота на столько выше пола, м
DROOP_RAD = 0.05       # допустимый провис от силы тяжести без компенсации: kp ≥ G / DROOP (для slide — 5 мм)
DROOP_M = 0.005
MESH_EXT = {".stl", ".obj", ".msh"}  # что умеет читать MuJoCo; .dae нужно перевести в .stl/.obj заранее

JOINT_1DOF = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)

# Где искать модель, если путь не найден как есть: папка models/ хоста (в контейнере /models) и модели Unitree.
MODELS_DIR = Path(os.environ.get("GUMANOID_MODELS", "/models"))
XR_ASSETS = Path(os.environ.get("XR_TELEOP_ASSETS", Path.home() / "xr_teleoperate" / "assets"))
# Короткие имена для моделей из xr_teleoperate, у которых нет своего скрипта (идут через этот загрузчик).
SHORT_NAMES = {
    "g1": "g1/g1_body29_hand14.xml", "h2": "h2/H2.urdf",  # у них свои скрипты; здесь — с подобранными коэффициентами
    "h1": "h1/h1_with_hand.urdf", "h1_2": "h1_2/h1_2.urdf", "g1_23": "g1/g1_body23.urdf",
    "r1": "r1/r1_a7.urdf", "r1_a5": "r1/r1_a5.urdf",
    "dex3": "unitree_hand/unitree_dex3_left.urdf", "inspire": "inspire_hand/inspire_hand_left.urdf",
    "brainco": "brainco_hand/brainco_left.urdf",
}


# ---------------------------------------------------------------- загрузка модели

def find_model(name: str | Path) -> Path:
    """Путь как есть -> models/<путь> -> короткое имя (h1, r1, dex3 …) -> xr_teleoperate/assets/<путь>."""
    name = str(name)
    cands = [Path(name), MODELS_DIR / name]
    if name in SHORT_NAMES:
        cands.append(XR_ASSETS / SHORT_NAMES[name])
    cands.append(XR_ASSETS / name)
    for c in cands:
        if c.is_file():
            return c.resolve()
    sys.exit(f"robot_sliders: модель «{name}» не найдена (искал: {', '.join(map(str, cands))}).\n"
             f"Короткие имена: {', '.join(SHORT_NAMES)}. Свои модели кладите в папку models/ рядом с run.sh.")


def load_robot(name: str, free: bool = False, band: bool = False) -> mujoco.MjSpec:
    """Сцена по имени: g1 / h2 / zbot — их собственными скриптами (настоящие коэффициенты моторов), всё остальное —
    универсальным загрузчиком. Удобно в своих скриптах: spec = load_robot("g1"); model = spec.compile()."""
    if name == "g1":
        import g1_sliders
        return g1_sliders.build_spec(g1_sliders.DEFAULT_XML, free, band, False)
    if name == "h2":
        import h2_sliders
        return h2_sliders.build_spec(h2_sliders.DEFAULT_URDF, free, band, False)
    if name in ("zbot", "zeroth"):
        import zbot_sliders  # подвески у Zeroth-01 нет: --band = на полу
        return zbot_sliders.build_spec(zbot_sliders.DEFAULT_ASSETS, free or band, None)
    return build_spec(find_model(name), free, band)


def resolve_mesh(filename: str, urdf_dir: Path) -> Path | None:
    """Путь меша из URDF -> файл на диске. Понимает package://pkg/..., относительные и абсолютные пути."""
    rel = filename.removeprefix("file://")
    cands: list[Path] = []
    if rel.startswith("package://"):
        pkg, _, rest = rel.removeprefix("package://").partition("/")
        for up in [urdf_dir, *urdf_dir.parents][:4]:  # пакет — это папка с URDF или одна из папок выше
            cands += [up / pkg / rest, up / rest] + ([up / rest] if up.name == pkg else [])
    else:
        cands += [Path(rel)] if Path(rel).is_absolute() else [urdf_dir / rel]
    for c in cands:
        if c.is_file():
            return c.resolve()
    # последний шанс: файл с тем же именем где-нибудь рядом (меши часто перекладывают в meshes/)
    name = Path(rel).name
    for base in (urdf_dir, urdf_dir.parent):
        hit = next(base.rglob(name), None)
        if hit:
            return hit.resolve()
    return None


def load_urdf(path: Path) -> mujoco.MjSpec:
    tree = ET.parse(path)
    root = tree.getroot()
    missing, wrong_ext = [], []
    for mesh in root.iter("mesh"):
        found = resolve_mesh(mesh.get("filename", ""), path.parent)
        if found is None:
            missing.append(mesh.get("filename"))
            continue
        if found.suffix.lower() not in MESH_EXT:
            wrong_ext.append(found.name)
        mesh.set("filename", str(found))  # абсолютный путь: модель потом откроется из любой папки
    if missing:
        sys.exit(f"robot_sliders: не найдены меши ({len(missing)}): {', '.join(sorted(set(missing))[:5])} …\n"
                 "Положите папку meshes/ рядом с URDF (или пакет ROS целиком) и повторите.")
    if wrong_ext:
        sys.exit(f"robot_sliders: MuJoCo не читает {', '.join(sorted(set(wrong_ext))[:5])}: переведите меши в .stl или .obj "
                 "(например, в Blender или `assimp export file.dae file.obj`) и поправьте пути в URDF.")
    # Настройки импорта URDF в MuJoCo: пути мешей не обрезать, визуальные меши оставить, звенья без суставов не сливать.
    mj = root.find("mujoco")
    if mj is None:
        mj = ET.SubElement(root, "mujoco")
    comp = mj.find("compiler")
    if comp is None:
        comp = ET.SubElement(mj, "compiler")
    comp.attrib.pop("meshdir", None)
    comp.set("strippath", "false")
    comp.set("discardvisual", "false")
    comp.set("fusestatic", "false")
    return mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))


def load_spec(path: Path) -> mujoco.MjSpec:
    if path.suffix.lower() == ".urdf" or ET.parse(path).getroot().tag == "robot":
        spec = load_urdf(path)
    elif path.suffix.lower() == ".xml":
        spec = mujoco.MjSpec.from_file(str(path))
    else:  # .mjcf и прочие: from_file их не принимает, читаем текст (тогда не работают <include>)
        spec = mujoco.MjSpec.from_string(path.read_text())
    # пути ассетов — абсолютные, чтобы --save-xml открывался из любой папки
    spec.meshdir = str(path.parent / spec.meshdir)
    spec.texturedir = str(path.parent / spec.texturedir)
    spec.modelname = path.parent.name if path.stem in ("scene", "robot", "model") else path.stem  # заголовок окна
    return spec


# ---------------------------------------------------------------- устройство модели

def robot_root(spec: mujoco.MjSpec) -> mujoco.MjsBody:
    """Корень робота: тело верхнего уровня, под которым больше всего суставов (остальные — стол, предметы…)."""
    tops = list(spec.worldbody.bodies)
    if not tops:
        sys.exit("robot_sliders: в модели нет ни одного тела")
    return max(tops, key=lambda b: len(b.find_all(mujoco.mjtObj.mjOBJ_JOINT)))


def subtree(body: mujoco.MjsBody) -> list[mujoco.MjsBody]:
    return [body, *body.find_all(mujoco.mjtObj.mjOBJ_BODY)]


def is_servo(act: mujoco.MjsActuator) -> bool:
    """Позиционный сервопривод (<position> или эквивалентный <general>): ctrl — это цель, а не момент."""
    return (act.gaintype == mujoco.mjtGain.mjGAIN_FIXED and act.biastype == mujoco.mjtBias.mjBIAS_AFFINE
            and act.gainprm[0] > 0 and act.biasprm[1] < 0)


def motor_efforts(spec: mujoco.MjSpec) -> dict[str, float]:
    """Сустав -> предельный момент по моторам модели (ctrlrange·gear или forcerange), пока их не удалили."""
    out = {}
    for a in spec.actuators:
        if a.trntype != mujoco.mjtTrn.mjTRN_JOINT or is_servo(a):
            continue
        lim = 0.0
        if a.ctrllimited != mujoco.mjtLimited.mjLIMITED_FALSE and a.ctrlrange[1] > a.ctrlrange[0]:
            lim = max(abs(a.ctrlrange[0]), abs(a.ctrlrange[1])) * abs(a.gear[0])
        if a.forcelimited != mujoco.mjtLimited.mjLIMITED_FALSE and a.forcerange[1] > a.forcerange[0]:
            f = max(abs(a.forcerange[0]), abs(a.forcerange[1]))
            lim = min(lim, f) if lim else f
        if lim:
            out[a.target] = lim
    return out


def keyframe_pose(spec: mujoco.MjSpec, key: str | None) -> tuple[dict[str, float], np.ndarray | None]:
    """Углы суставов и ctrl из ключевой позы модели (по умолчанию — "home", если она есть). Читается до перестройки
    модели: потом размер qpos меняется (удаляется или добавляется freejoint) и сами <key> приходится удалить."""
    if key == "none":
        return {}, None
    if key is not None and not spec.keys:
        sys.exit(f"robot_sliders: --key {key}: в модели нет ключевых поз")
    if key is None and not any(k.name == "home" for k in spec.keys):
        return {}, None
    try:
        m = spec.compile()
    except Exception:  # noqa: BLE001 — модель с ошибкой всё равно попробуем собрать после правок
        return {}, None
    name = key or "home"
    k = int(name) if name.isdigit() else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, name)
    if not 0 <= k < m.nkey:
        sys.exit(f"robot_sliders: --key {key}: нет такой позы (есть: {', '.join(m.key(i).name or str(i) for i in range(m.nkey))})")
    qpos = m.key_qpos[k]
    pose = {m.joint(j).name: float(qpos[m.jnt_qposadr[j]]) for j in range(m.njnt) if int(m.jnt_type[j]) in JOINT_1DOF}
    return pose, m.key_ctrl[k].copy()


def geom_bounds(m: mujoco.MjModel, d: mujoco.MjData, root_id: int, collision: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Габариты робота (мин/макс по x, y, z) в текущей позе; collision — только по геомам столкновений, если они есть."""
    mine = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root_id
            and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]
    coll = [g for g in mine if m.geom_contype[g] or m.geom_conaffinity[g]] if collision else []
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in coll or mine:
        c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        rot = d.geom_xmat[g].reshape(3, 3)
        mid, half = d.geom_xpos[g] + rot @ c, np.abs(rot) @ h
        lo, hi = np.minimum(lo, mid - half), np.maximum(hi, mid + half)
    return lo, hi


def set_pose(m: mujoco.MjModel, d: mujoco.MjData, pose: dict[str, float]) -> None:
    d.qpos[:] = m.qpos0
    for name, q in pose.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = q
    mujoco.mj_forward(m, d)


def load_gains(path: Path | None) -> list[tuple[re.Pattern, dict]]:
    if path is None:
        return []
    return [(re.compile(k), v) for k, v in json.loads(path.read_text()).items()]


# ---------------------------------------------------------------- сборка сцены

def build_spec(path: Path, free: bool = False, band: bool = False, *, gravcomp: bool = True,
               actuators: str = "auto", kp: float | None = None, kv: float | None = None, freq: float = 4.0,
               effort: float | None = None, gains: Path | None = None, key: str | None = None,
               exclude: str | None = None, self_collision: bool = False) -> mujoco.MjSpec:
    spec = load_spec(path)
    floating = free or band
    if spec.option.integrator == mujoco.mjtIntegrator.mjINT_EULER:
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # демпфирование привода — неявно, устойчивее

    # Пол, свет и небо — только если своих нет (у сцен из mujoco_menagerie они уже есть).
    has_floor = any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.worldbody.geoms)
    if not has_floor:
        had_lights = bool(spec.lights)
        names = {e.name for e in [*spec.geoms, *spec.textures, *spec.materials]}
        add_scene(spec, prefix="gs_" if names & {"floor", "grid", "sky", "floor_mat"} else "")
        if had_lights:  # свет у модели был свой — наш второй раз не нужен (add_scene ставит два в конец мира)
            for light in list(spec.worldbody.lights)[-2:]:
                spec.delete(light)

    pose, key_ctrl = keyframe_pose(spec, key)
    for k in list(spec.keys):
        spec.delete(k)

    root = robot_root(spec)
    if not root.name:
        root.name = "robot_root"
    bodies = subtree(root)
    freejoints = [j for j in root.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]
    was_floating = bool(freejoints)
    if floating and not freejoints:
        root.add_freejoint(name="floating_base")
    if not floating:
        for j in freejoints:  # корень жёстко прикреплён к миру — робот «висит на стенде»
            spec.delete(j)

    # Геомы робота: сталкиваются с полом и предметами, но не друг с другом (меши соседних звеньев часто
    # перекрываются уже в нулевой позе). Если у звена есть отдельные визуальные меши, коллизии прячутся в группу 3.
    for b in bodies:
        has_visual = any(not (g.contype or g.conaffinity) for g in b.geoms)
        for g in b.geoms:
            if g.contype or g.conaffinity:
                if not self_collision:
                    g.contype, g.conaffinity = 1, 0
                if has_visual:
                    g.group = 3

    # Какие суставы получат ползунки: все hinge/slide робота, кроме ведомых через <equality joint>.
    driven = {e.name1 for e in spec.equalities if e.type == mujoco.mjtEq.mjEQ_JOINT}
    skip = re.compile(exclude) if exclude else None
    joints = [j for b in bodies for j in b.joints
              if j.type in JOINT_1DOF and j.name not in driven and not (skip and skip.search(j.name))]
    unnamed = [j for j in joints if not j.name]
    for i, j in enumerate(unnamed):
        j.name = f"joint{i}"

    old_efforts = motor_efforts(spec)
    if actuators == "auto":
        actuators = "keep" if spec.actuators and all(is_servo(a) for a in spec.actuators) else "replace"
    if actuators == "replace":
        for s in list(spec.sensors):  # датчики на удаляемых моторах сломали бы компиляцию
            if s.objtype == mujoco.mjtObj.mjOBJ_ACTUATOR:
                spec.delete(s)
        for a in list(spec.actuators):
            spec.delete(a)

    if gravcomp and not floating:
        # Компенсация силы тяжести моментами приводов, как feed-forward у контроллера настоящего робота.
        for b in bodies:
            b.gravcomp = 1
        for j in joints:
            j.actgravcomp = True

    # Компиляция №1: поза, габариты, инерция суставов и плечи силы тяжести.
    m = spec.compile()
    d = mujoco.MjData(m)
    set_pose(m, d, pose)
    root_id = m.body(root.name).id
    low = geom_bounds(m, d, root_id, collision=True)[0][2]  # чем робот касается пола
    if floating:
        dz = -low + 0.002  # стопы на 2 мм над полом
    elif was_floating or low < -0.005:
        dz = HANG_CLEARANCE - low
    else:
        dz = 0.0  # манипулятор на своём месте
    root.pos = [root.pos[0], root.pos[1], root.pos[2] + dz]
    if band:
        spec.add_equality(name="band", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                          name1=root.name, solref=BAND_SOLREF, data=[0] * 10 + [1])  # см. h2_sliders.add_band

    lo, hi = geom_bounds(m, d, root_id)  # стартовая камера: робот целиком
    spec.stat.center = ((lo + hi) / 2 + [0, 0, dz]).tolist()
    spec.stat.extent = float(np.linalg.norm(hi - lo)) * 0.8
    spec.visual.global_.azimuth = 150
    spec.visual.global_.elevation = -15
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, 1920)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, 1080)

    if actuators == "replace":
        mass = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, mass)
        overrides = load_gains(gains)
        dt = m.opt.timestep
        for j in joints:
            jid = m.joint(j.name).id
            dof, body = m.jnt_dofadr[jid], m.jnt_bodyid[jid]
            hinge = m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
            o = next((v for rx, v in overrides if rx.search(j.name)), {})
            if "armature" in o:
                mass[dof, dof] += o["armature"] - j.armature
                j.armature = o["armature"]
            m_ii = mass[dof, dof]
            # наибольший момент силы тяжести: масса поддерева × g × расстояние от оси до его центра масс
            r = d.subtree_com[body] - d.xanchor[jid]
            axis = d.xaxis[jid]
            lever = np.linalg.norm(r - axis * (r @ axis)) if hinge else 1.0
            grav = m.body_subtreemass[body] * np.linalg.norm(m.opt.gravity) * lever
            k = o.get("kp", kp)
            if k is None:
                k = max(m_ii * (2 * np.pi * freq) ** 2, grav / (DROOP_RAD if hinge else DROOP_M))
            if "armature" not in o and k * dt * dt > 0.25 * m_ii:  # иначе явный шаг по kp неустойчив
                j.armature += k * dt * dt / 0.25 - m_ii
                m_ii = k * dt * dt / 0.25
            c = o.get("kv", kv)
            if c is None:
                c = 2.0 * np.sqrt(k * m_ii)
            lim = o.get("effort", effort)
            if lim is None:
                lim = j.actfrcrange[1] if j.actfrcrange[1] > 0 else old_efforts.get(j.name)
            if m.jnt_limited[jid]:
                rng = m.jnt_range[jid]
            else:
                rng = [-np.pi, np.pi] if hinge else [-1.0, 1.0]
            act = spec.add_actuator(name=j.name.removesuffix("_joint"), target=j.name,
                                    trntype=mujoco.mjtTrn.mjTRN_JOINT)
            act.set_to_position(kp=float(k), kv=float(c))
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.ctrlrange = rng
            if lim:
                act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
                act.forcerange = [-lim, lim]

    # Компиляция №2: исходная поза "home" (Backspace в окне возвращает к ней).
    m = spec.compile()
    d = mujoco.MjData(m)
    set_pose(m, d, pose)
    ctrl = key_ctrl if actuators == "keep" and key_ctrl is not None and len(key_ctrl) == m.nu else np.zeros(m.nu)
    for i in range(m.nu):
        if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT and is_servo(spec.actuators[i]):
            ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
    ctrl = np.where(m.actuator_ctrllimited, np.clip(ctrl, *m.actuator_ctrlrange.T), ctrl)
    spec.add_key(name="home", qpos=d.qpos.copy(), ctrl=ctrl)
    return spec


def print_table(m: mujoco.MjModel) -> None:
    print(f"{m.nu} ползунков, масса {m.body_subtreemass[0]:.2f} кг, шаг {m.opt.timestep * 1000:g} мс")
    print(f"  {'#':>3} {'ползунок':<28} {'диапазон':>18} {'kp':>9} {'kv':>8} {'макс. сила':>11}")
    for i in range(m.nu):
        a = m.actuator(i)
        hinge = (m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
                 and m.jnt_type[m.actuator_trnid[i, 0]] == mujoco.mjtJoint.mjJNT_HINGE)
        lo, hi = m.actuator_ctrlrange[i]
        rng = f"{np.degrees(lo):7.1f}° .. {np.degrees(hi):6.1f}°" if hinge else f"{lo:7.3f} .. {hi:7.3f}"
        lim = f"{m.actuator_forcerange[i, 1]:9.2f}" if m.actuator_forcelimited[i] else "      нет"
        print(f"  {i + 1:>3} {a.name:<28} {rng:>18} {m.actuator_gainprm[i, 0]:9.2f} {-m.actuator_biasprm[i, 2]:8.3f} {lim:>11}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="файл робота (.urdf, .xml, .mjcf) — путь, путь внутри models/ или короткое имя: "
                    + ", ".join(SHORT_NAMES))
    ap.add_argument("--free", action="store_true", help="свободная база на полу (без контроллера баланса шагающие упадут)")
    ap.add_argument("--band", action="store_true", help="на полу, корень держит мягкая подвеска (панель Equality)")
    ap.add_argument("--no-gravcomp", action="store_true", help="без компенсации силы тяжести в подвесе")
    ap.add_argument("--actuators", choices=["auto", "replace", "keep"], default="auto",
                    help="replace — свои сервоприводы на все суставы; keep — оставить приводы модели (auto: keep, "
                         "если они уже позиционные)")
    ap.add_argument("--kp", type=float, help="жёсткость всех приводов, Н·м/рад (по умолчанию — подбор)")
    ap.add_argument("--kv", type=float, help="демпфирование, Н·м·с/рад (по умолчанию — критическое)")
    ap.add_argument("--freq", type=float, default=4.0, help="собственная частота привода при подборе kp, Гц")
    ap.add_argument("--effort", type=float, help="предельный момент всех приводов, Н·м (по умолчанию — из модели)")
    ap.add_argument("--gains", type=Path, help="JSON с kp/kv/effort/armature по суставам (см. выше)")
    ap.add_argument("--key", help="стартовая поза: имя или номер <key> модели; none — нули (по умолчанию home, если есть)")
    ap.add_argument("--exclude", help="регулярное выражение: суставы без ползунков (например 'finger|hand')")
    ap.add_argument("--self-collision", action="store_true", help="не выключать столкновения звеньев друг с другом")
    ap.add_argument("--info", action="store_true", help="напечатать таблицу ползунков и выйти")
    ap.add_argument("--save-xml", type=Path, help="сохранить собранную сцену в MJCF и выйти")
    args = ap.parse_args()

    path = find_model(args.model)
    spec = build_spec(path, args.free, args.band, gravcomp=not args.no_gravcomp, actuators=args.actuators,
                      kp=args.kp, kv=args.kv, freq=args.freq, effort=args.effort, gains=args.gains, key=args.key,
                      exclude=args.exclude, self_collision=args.self_collision)
    model = spec.compile()
    if args.save_xml:
        args.save_xml.write_text(spec.to_xml())
        print("saved", args.save_xml)
        return
    print(f"{path.name}: ", end="")
    print_table(model)
    if args.info:
        return
    print("Ползунки: правая панель -> Control. Backspace — сброс позы, пробел — пауза.")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
