"""
reward_function.py

This is a Python port of the compute_reward() function from
example/Example1_Simple.gd (the GDScript reward shaping logic used to
train the DQN co-pilot).

WHY THIS FILE EXISTS:
GDScript can't be run or imported by pytest directly, since pytest is a
Python tool and Godot is a separate engine. So to unit test the reward
SHAPING LOGIC (the actual math/rules that decide "was this a good
action or a bad action"), we re-write the same logic in plain Python,
one-to-one, matching every branch and constant from the .gd file.

This does NOT replace testing inside Godot. It tests that the reward
function behaves the way it's supposed to (rewards good flight,
punishes stalling, etc) in isolation, fast, and repeatably -- which is
exactly what a unit test is for.

Source of truth: example/Example1_Simple.gd, lines ~543-590
(compute_reward, TARGET_ALT, ALT_SIGMA, ALT_FLOOR)
"""

import math

# --- constants, copied exactly from Example1_Simple.gd ---
TARGET_ALT = 200.0
ALT_SIGMA = 80.0
ALT_FLOOR = 50.0


def compute_reward(
    alt: float,
    spd: float,
    vs: float,
    fuel: float,
    engine_on: bool,
    gear_down: bool,
    has_landed_safely: bool,
    is_done: bool,
    is_stalled: bool,
    g_force: float,
    aircraft_valid: bool = True,
) -> float:
    """
    Python equivalent of compute_reward() in Example1_Simple.gd.

    Parameters map directly to the GDScript variables:
        alt                -> aircraft.local_altitude (clamped >= 0)
        spd                -> aircraft.forward_air_speed
        vs                 -> aircraft.linear_velocity.y (vertical speed)
        fuel               -> get_fuel_soc()
        engine_on          -> engine_module.is_engine_working
        gear_down          -> landing_gear_module.is_deployed
        has_landed_safely  -> has_landed_safely (episode flag)
        is_done            -> is_done (episode flag)
        is_stalled         -> aircraft.is_stalled
        g_force            -> aircraft.local_g_force
        aircraft_valid     -> is_instance_valid(aircraft) in Godot
    """
    if not aircraft_valid:
        return -1.0

    alt = max(alt, 0.0)
    rw = 0.0

    on_ground = alt < 10.0

    if has_landed_safely:
        rw += 15.0
    elif on_ground:
        rw -= 0.5
        if spd < 3.0 and not is_done:
            rw += 2.0
    else:
        rw += min(spd * 0.002, 0.3)
        if fuel > 0.1 and engine_on:
            alt_dev = abs(alt - TARGET_ALT)
            rw += math.exp(-(alt_dev * alt_dev) / (2.0 * ALT_SIGMA * ALT_SIGMA)) * 3.0
            if alt_dev < ALT_SIGMA * 2:
                rw -= abs(vs) * 0.05
            if alt < ALT_FLOOR:
                rw -= (ALT_FLOOR - alt) / ALT_FLOOR * 2.0
        else:
            rw += 0.5 if gear_down else -0.5
            rw += 0.3 if vs < -1.0 else 0.0
            if alt < 50.0 and spd < 20.0:
                rw += 2.0
            if not engine_on and gear_down:
                rw += 0.5

    if alt > 50.0 and spd > 30.0 and not is_stalled and engine_on:
        rw += 0.2
    if is_stalled:
        rw -= 1.0
    if g_force > 5.0:
        rw -= 0.5

    return rw
