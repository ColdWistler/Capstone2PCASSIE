"""
test_reward_function.py

Unit tests for reward_function.compute_reward().

Test IDs follow the UT-MO-xx naming convention (MO = Model Optimization),
matching the UT-DA-xx style already used in the group's ST_STP document
for the Data Architect tests.

Run with:
    pytest test_reward_function.py -v
"""

import pytest
from reward_function import compute_reward, TARGET_ALT, ALT_SIGMA


# A default "normal cruising flight" baseline we can tweak per test,
# so each test only changes the ONE thing it's actually checking.
def base_kwargs(**overrides):
    kwargs = dict(
        alt=200.0,          # at target altitude
        spd=40.0,           # decent cruise speed
        vs=0.0,              # no climb/descent
        fuel=0.8,            # plenty of fuel
        engine_on=True,
        gear_down=False,
        has_landed_safely=False,
        is_done=False,
        is_stalled=False,
        g_force=1.0,
        aircraft_valid=True,
    )
    kwargs.update(overrides)
    return kwargs


# UT-MO-01: invalid aircraft instance always returns -1.0
def test_invalid_aircraft_returns_minus_one():
    r = compute_reward(**base_kwargs(aircraft_valid=False))
    assert r == -1.0


# UT-MO-02: landing safely gives a flat +15 bonus regardless of other state
def test_safe_landing_gives_flat_bonus():
    r = compute_reward(**base_kwargs(has_landed_safely=True, alt=0.0, spd=0.0))
    assert r == pytest.approx(15.0)


# UT-MO-03: reward is highest exactly at TARGET_ALT, and falls off as you move away
def test_reward_peaks_at_target_altitude():
    r_at_target = compute_reward(**base_kwargs(alt=TARGET_ALT))
    r_near = compute_reward(**base_kwargs(alt=TARGET_ALT + ALT_SIGMA))
    r_far = compute_reward(**base_kwargs(alt=TARGET_ALT + ALT_SIGMA * 3))
    assert r_at_target > r_near > r_far


# UT-MO-04: reward decreases monotonically as altitude deviation increases
# (proves the shaping curve is smooth, not noisy/broken)
def test_reward_monotonic_with_altitude_deviation():
    deviations = [0, 20, 40, 80, 120, 160, 240]
    rewards = [compute_reward(**base_kwargs(alt=TARGET_ALT + d)) for d in deviations]
    for i in range(len(rewards) - 1):
        assert rewards[i] >= rewards[i + 1], (
            f"Reward should not increase as altitude deviation grows: "
            f"dev={deviations[i]} -> {rewards[i]}, dev={deviations[i+1]} -> {rewards[i+1]}"
        )


# UT-MO-05: stalling always makes the reward worse than not stalling.
# Note: stalling costs -1.0 directly, AND it also removes the +0.2
# "clean flight" bonus (since that bonus requires not is_stalled) --
# so the real gap is 1.2, not 1.0. This is correct GDScript behavior,
# the test just needs to expect the right number.
def test_stall_always_penalized():
    r_normal = compute_reward(**base_kwargs(is_stalled=False))
    r_stalled = compute_reward(**base_kwargs(is_stalled=True))
    assert r_stalled == pytest.approx(r_normal - 1.2)


# UT-MO-06: excessive g-force (> 5.0) is penalized
def test_high_g_force_penalized():
    r_normal = compute_reward(**base_kwargs(g_force=1.0))
    r_high_g = compute_reward(**base_kwargs(g_force=6.0))
    assert r_high_g == pytest.approx(r_normal - 0.5)


# UT-MO-07: on the ground, low speed (a controlled stop) earns a small bonus
def test_on_ground_low_speed_gets_bonus():
    r_slow = compute_reward(**base_kwargs(alt=0.0, spd=1.0, is_done=False))
    r_fast = compute_reward(**base_kwargs(alt=0.0, spd=25.0, is_done=False))
    assert r_slow > r_fast


# UT-MO-08: descending too fast near target altitude is penalized
# (vertical speed penalty only applies within 2*ALT_SIGMA of target)
def test_high_vertical_speed_near_target_penalized():
    r_stable = compute_reward(**base_kwargs(alt=TARGET_ALT, vs=0.0))
    r_diving = compute_reward(**base_kwargs(alt=TARGET_ALT, vs=-10.0))
    assert r_diving < r_stable


# UT-MO-09: dropping below ALT_FLOOR (50.0) while still flying adds an extra penalty
def test_below_altitude_floor_penalized_extra():
    r_at_floor = compute_reward(**base_kwargs(alt=50.0))
    r_below_floor = compute_reward(**base_kwargs(alt=20.0))
    assert r_below_floor < r_at_floor


# UT-MO-10: out of fuel / engine off switches the function into "glide/land" mode,
# where deploying the landing gear is rewarded instead of penalized
def test_engine_off_rewards_gear_down():
    r_gear_up = compute_reward(**base_kwargs(fuel=0.0, engine_on=False, gear_down=False))
    r_gear_down = compute_reward(**base_kwargs(fuel=0.0, engine_on=False, gear_down=True))
    assert r_gear_down > r_gear_up


# UT-MO-11: cruising cleanly (mid-altitude, good speed, no stall, engine on)
# earns the small "clean flight" bonus on top of the altitude-tracking reward
def test_clean_cruise_gets_small_bonus():
    r_clean = compute_reward(**base_kwargs(alt=200.0, spd=40.0, is_stalled=False, engine_on=True))
    r_stalled = compute_reward(**base_kwargs(alt=200.0, spd=40.0, is_stalled=True, engine_on=True))
    # stalled version should be about 0.2 (lost bonus) + 1.0 (stall penalty) lower
    assert (r_clean - r_stalled) == pytest.approx(1.2, abs=0.01)
