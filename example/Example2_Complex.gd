extends Node3D

const STATE_DIM = 13
const ACTION_DIM = 7
const HIDDEN = 256
const MAX_EPISODE_STEPS = 2000
const FRAMES_PER_STEP = 4
const BATCH_SIZE = 64
const REPLAY_CAPACITY = 50000
const N_STEPS = 3
const GAMMA = 0.99
var GAMMA_POW = [1.0, GAMMA, GAMMA * GAMMA, GAMMA * GAMMA * GAMMA]
const LR_INIT = 0.001
const LR_DECAY = 0.9995
const LR_MIN = 0.0001
const ADAM_BETA1 = 0.9
const ADAM_BETA2 = 0.999
const ADAM_EPS = 1e-8
const TAU = 0.005
const EPSILON_START = 1.0
const EPSILON_MIN = 0.01
const EPSILON_DECAY = 0.998
const GRAD_CLIP = 1.0
const PRIORITY_ALPHA = 0.6
const PRIORITY_BETA_START = 0.4
const PRIORITY_BETA_STEPS = 100000
const SAVE_PATH = "user://dqn_complex_weights.save"
const META_PATH = "user://dqn_complex_meta.save"
const TRAIN_INTERVAL = 2
const CHUNK_SIZE = 16
const SAVE_INTERVAL = 50
const SAVE_VERSION = 1
const TEST_EPISODES = 10
const TEST_REPORT_PATH = "user://test_report_complex.txt"

const TEMPERATURE_ALTITUDE_DROP_RATE: float = 0.0065
const ALTITUDE_OF_ZERO_DENSITY: float = 100000.0
const REFUEL_RATE: float = 20.0

var template_explosion = preload("res://example/scenes/Explosion/Explosion.tscn")
var explosion_instance = null

var takeoff_phase = true
var has_landed_safely = false
var engine_was_running = false

@onready var aircraft = get_node("Aircraft")
var engine_modules = []
var steering_module: AircraftModule_Steering
var landing_gear_module: AircraftModule_LandingGear
var fuel_containers = []
var battery_container: AircraftModule_EnergyContainer = null

var throttle_level = 0.5
var pitch_level = 0.0
var roll_level = 0.0
var prev_state = []
var episode_step = 0
var episode_reward = 0.0
var is_done = false
var frame_counter = 0

var step_count = 0
var episode_count = 0
var best_reward = -1e9

var sumtree: SumTree
var max_priority = 1.0

var w1 = PackedFloat32Array()
var b1 = PackedFloat32Array()
var wA = PackedFloat32Array()
var bA = PackedFloat32Array()
var wV = PackedFloat32Array()
var bV = 0.0
var w1t = PackedFloat32Array()
var b1t = PackedFloat32Array()
var wAt = PackedFloat32Array()
var bAt = PackedFloat32Array()
var wVt = PackedFloat32Array()
var bVt = 0.0

var epsilon = EPSILON_START

var m_w1 = PackedFloat32Array()
var v_w1 = PackedFloat32Array()
var m_b1 = PackedFloat32Array()
var v_b1 = PackedFloat32Array()
var m_wA = PackedFloat32Array()
var v_wA = PackedFloat32Array()
var m_bA = PackedFloat32Array()
var v_bA = PackedFloat32Array()
var m_wV = PackedFloat32Array()
var v_wV = PackedFloat32Array()
var m_bV = 0.0
var v_bV = 0.0
var adam_step = 0

var nstep_buffer = []
var prev_action = -1
var train_counter = 0

var run_id = 0
var _last_saved_best = -1e9

var chunk_batch = []
var chunk_indices = []
var chunk_is_weights = PackedFloat32Array()
var chunk_ptr = -1
var chunk_dw1 = PackedFloat32Array()
var chunk_db1 = PackedFloat32Array()
var chunk_dwA = PackedFloat32Array()
var chunk_dbA = PackedFloat32Array()
var chunk_dwV = PackedFloat32Array()
var chunk_dbV = 0.0

var test_mode = false
var test_results = []
var test_ep_alt_sum = 0.0
var test_ep_stall_acc = 0

var is_landing_mode = false

var is_reloading_fuel = false
var is_charging_battery = false

class SumTree:
	var tree: PackedFloat32Array
	var data: Array
	var capacity: int
	var size: int = 0
	var write_idx: int = 0

	func _init(cap: int):
		capacity = 1
		while capacity < cap:
			capacity <<= 1
		tree.resize(capacity * 2)
		data.resize(capacity)

	func total() -> float:
		return tree[1]

	func add(item, priority: float):
		var idx = capacity + write_idx
		data[write_idx] = item
		_tree_set(idx, priority)
		write_idx = (write_idx + 1) % capacity
		if size < capacity:
			size += 1

	func _tree_set(idx: int, priority: float):
		tree[idx] = priority
		idx >>= 1
		while idx:
			tree[idx] = tree[idx * 2] + tree[idx * 2 + 1]
			idx >>= 1

	func _retrieve(idx: int, s: float) -> int:
		while idx < capacity:
			var left = idx * 2
			if tree[left] >= s:
				idx = left
			else:
				s -= tree[left]
				idx = left + 1
		return idx

	func sample(n: int) -> Array:
		var batch = []
		var indices = []
		var priorities = []
		var total_p = total()
		if total_p <= 0.0:
			return [[], [], []]
		var seg = total_p / n
		for i in range(n):
			var s = seg * (i + randf())
			var idx = _retrieve(1, s)
			var data_idx = idx - capacity
			indices.append(data_idx)
			batch.append(data[data_idx])
			priorities.append(tree[idx])
		return [batch, indices, priorities]

	func set_priority(idx: int, priority: float):
		if idx < capacity:
			_tree_set(capacity + idx, priority)


func _ready():
	await get_tree().process_frame

	for child in aircraft.get_children():
		if child is AircraftModule_Engine:
			engine_modules.append(child)
		elif child is AircraftModule_EnergyContainer:
			if child.EnergyType == "battery":
				battery_container = child
			else:
				fuel_containers.append(child)
		elif child is AircraftModule_Steering:
			steering_module = child
		elif child is AircraftModule_LandingGear:
			landing_gear_module = child

	if OS.has_feature("headless"):
		Engine.max_fps = -1
		Engine.time_scale = 5.0
		print("Headless mode — FPS uncapped, time scale 5x")

	for arg in OS.get_cmdline_args():
		if arg == "--test" or arg == "-t":
			test_mode = true
			print("TEST MODE — %d episodes, greedy actions, no training" % TEST_EPISODES)
			break

	_init_network()
	sumtree = SumTree.new(REPLAY_CAPACITY)
	_init_run_id()
	var best_path = _find_best_run()
	var loaded = false
	if best_path != "":
		loaded = load_weights(best_path)
		if loaded:
			print("Loaded best model (best_reward %.1f)" % best_reward)
	if not loaded:
		loaded = load_weights()
		if loaded:
			print("Loaded saved weights (episode %d)" % episode_count)
		else:
			print("No saved weights found, starting fresh")
	if test_mode:
		takeoff_phase = false
	initialize_aircraft()
	if takeoff_phase:
		print("Starting takeoff from runway...")
	else:
		print("Spawning at altitude...")


func _init_network():
	var seed_val = randi()
	var rng = RandomNumberGenerator.new()
	rng.seed = seed_val
	w1.resize(HIDDEN * STATE_DIM)
	b1.resize(HIDDEN)
	wA.resize(ACTION_DIM * HIDDEN)
	bA.resize(ACTION_DIM)
	wV.resize(HIDDEN)
	bV = 0.0
	for i in range(w1.size()):
		w1[i] = rng.randfn(0.0, 0.1)
	for i in range(b1.size()):
		b1[i] = 0.0
	for i in range(wA.size()):
		wA[i] = rng.randfn(0.0, 0.1)
	for i in range(bA.size()):
		bA[i] = 0.0
	for i in range(wV.size()):
		wV[i] = rng.randfn(0.0, 0.1)
	_init_adam()
	_copy_to_target()


func _init_adam():
	m_w1.resize(HIDDEN * STATE_DIM)
	v_w1.resize(HIDDEN * STATE_DIM)
	m_b1.resize(HIDDEN)
	v_b1.resize(HIDDEN)
	m_wA.resize(ACTION_DIM * HIDDEN)
	v_wA.resize(ACTION_DIM * HIDDEN)
	m_bA.resize(ACTION_DIM)
	v_bA.resize(ACTION_DIM)
	m_wV.resize(HIDDEN)
	v_wV.resize(HIDDEN)
	m_bV = 0.0
	v_bV = 0.0
	adam_step = 0


func _copy_to_target():
	w1t = w1.duplicate()
	b1t = b1.duplicate()
	wAt = wA.duplicate()
	bAt = bA.duplicate()
	wVt = wV.duplicate()
	bVt = bV


func save_weights():
	_write_weights_to(SAVE_PATH)
	if best_reward > _last_saved_best:
		_write_weights_to("user://dqn_complex_run_%d.save" % run_id)
		_last_saved_best = best_reward
	print("Weights saved (episode %d)" % episode_count)


func _write_weights_to(path: String):
	var file = FileAccess.open(path, FileAccess.WRITE)
	if not file:
		push_error("Failed to open save file for writing")
		return
	file.store_32(SAVE_VERSION)
	file.store_var(w1)
	file.store_var(b1)
	file.store_var(wA)
	file.store_var(bA)
	file.store_var(wV)
	file.store_var(bV)
	file.store_var(epsilon)
	file.store_32(episode_count)
	file.store_32(step_count)
	file.store_var(best_reward)
	file.store_var(m_w1)
	file.store_var(v_w1)
	file.store_var(m_b1)
	file.store_var(v_b1)
	file.store_var(m_wA)
	file.store_var(v_wA)
	file.store_var(m_bA)
	file.store_var(v_bA)
	file.store_var(m_wV)
	file.store_var(v_wV)
	file.store_var(m_bV)
	file.store_var(v_bV)
	file.store_32(adam_step)
	file.close()


func load_weights(path := SAVE_PATH) -> bool:
	if not FileAccess.file_exists(path):
		return false
	var file = FileAccess.open(path, FileAccess.READ)
	if not file:
		return false
	var ver = file.get_32()
	if ver < 1:
		file.close()
		return false
	w1 = file.get_var()
	b1 = file.get_var()
	wA = file.get_var()
	bA = file.get_var()
	wV = file.get_var()
	bV = file.get_var()
	epsilon = file.get_var()
	episode_count = file.get_32()
	step_count = file.get_32()
	best_reward = file.get_var()
	m_w1 = file.get_var()
	v_w1 = file.get_var()
	m_b1 = file.get_var()
	v_b1 = file.get_var()
	m_wA = file.get_var()
	v_wA = file.get_var()
	m_bA = file.get_var()
	v_bA = file.get_var()
	m_wV = file.get_var()
	v_wV = file.get_var()
	m_bV = file.get_var()
	v_bV = file.get_var()
	adam_step = file.get_32()
	file.close()
	_copy_to_target()
	_last_saved_best = best_reward
	return true


func _init_run_id():
	var rid = 0
	if FileAccess.file_exists(META_PATH):
		var f = FileAccess.open(META_PATH, FileAccess.READ)
		if f:
			rid = f.get_32()
			f.close()
	var f = FileAccess.open(META_PATH, FileAccess.WRITE)
	if f:
		f.store_32(rid + 1)
		f.close()
	run_id = rid


func _find_best_run() -> String:
	var best_path = ""
	var best_r = -1e9
	var dir = DirAccess.open("user://")
	if not dir:
		return best_path
	dir.list_dir_begin()
	var fn = dir.get_next()
	while fn != "":
		if fn.begins_with("dqn_complex_run_") and fn.ends_with(".save"):
			var full = "user://" + fn
			var file = FileAccess.open(full, FileAccess.READ)
			if file:
				var ver = file.get_32()
				if ver >= 1:
					file.get_var()
					file.get_var()
					file.get_var()
					file.get_var()
					file.get_var()
					file.get_var()
					file.get_var()
					file.get_32()
					file.get_32()
					var r = file.get_var()
					if r > best_r:
						best_r = r
						best_path = full
				file.close()
		fn = dir.get_next()
	dir.list_dir_end()
	if best_path != "":
		print("Found best complex run model (reward %.1f)" % best_r)
	return best_path


func _on_episode_end():
	if test_mode:
		var alt_avg = test_ep_alt_sum / max(1, episode_step)
		var crashed = is_done and not has_landed_safely
		test_results.append({
			"reward": episode_reward,
			"steps": episode_step,
			"landed": has_landed_safely,
			"crashed": crashed,
			"avg_alt": alt_avg,
			"stalls": test_ep_stall_acc
		})
		var ep_idx = test_results.size()
		var status = "LANDED" if has_landed_safely else ("CRASHED" if crashed else "TIMEOUT")
		print("[%d/%d] ep %d | reward %.1f | steps %d | %s | avg_alt %.0f" %
			[ep_idx, TEST_EPISODES, ep_idx, episode_reward, episode_step, status, alt_avg])
		if ep_idx >= TEST_EPISODES:
			_generate_test_report()
			get_tree().quit()
	else:
		if episode_reward > best_reward:
			best_reward = episode_reward
		print("Episode %d | reward: %.1f | steps: %d | eps: %.3f | best: %.1f" %
			[episode_count, episode_reward, episode_step, epsilon, best_reward])


func _generate_test_report():
	var total_r = 0.0
	var total_steps = 0
	var landings = 0
	var crashes = 0
	var alt_sum = 0.0
	var stall_sum = 0
	var min_r = 1e9
	var max_r = -1e9

	var report = "=== DQN COMPLEX TEST REPORT ===\n"
	report += "Model: dqn_complex_run_%d.save\n" % run_id
	report += "Episodes: %d\n\n" % test_results.size()
	report += "%-8s %8s %6s %8s %7s %7s\n" % ["#", "Reward", "Steps", "Status", "AvgAlt", "Stalls"]

	for i in range(test_results.size()):
		var r = test_results[i]
		var status = "LANDED" if r["landed"] else ("CRASH" if r["crashed"] else "TIMEOUT")
		report += "%-8d %8.1f %6d %8s %7.0f %7d\n" % [i + 1, r["reward"], r["steps"], status, r["avg_alt"], r["stalls"]]
		total_r += r["reward"]
		total_steps += r["steps"]
		alt_sum += r["avg_alt"]
		stall_sum += r["stalls"]
		if r["landed"]: landings += 1
		if r["crashed"]: crashes += 1
		if r["reward"] < min_r: min_r = r["reward"]
		if r["reward"] > max_r: max_r = r["reward"]

	var n = float(test_results.size())
	var avg_r = total_r / n
	var avg_steps = total_steps / n
	var avg_alt = alt_sum / n
	var avg_stalls = stall_sum / n
	var success_rate = landings / n * 100.0

	report += "\n--- Summary ---\n"
	report += "Avg Reward:  %.1f (min %.1f, max %.1f)\n" % [avg_r, min_r, max_r]
	report += "Avg Steps:   %.0f / %d\n" % [avg_steps, MAX_EPISODE_STEPS]
	report += "Avg Alt:     %.0f m\n" % avg_alt
	report += "Avg Stalls:  %.1f / ep\n" % avg_stalls
	report += "Landings:    %d / %d (%.0f%%)\n" % [landings, test_results.size(), success_rate]
	report += "Crashes:     %d / %d\n" % [crashes, test_results.size()]

	var file = FileAccess.open(TEST_REPORT_PATH, FileAccess.WRITE)
	if file:
		file.store_string(report)
		file.close()
		print("\nTest report saved to: %s" % TEST_REPORT_PATH)
	else:
		push_error("Failed to save test report")

	print("\n" + report)


func initialize_aircraft():
	throttle_level = 0.5
	pitch_level = 0.0
	roll_level = 0.0
	episode_step = 0
	episode_reward = 0.0
	is_done = false

	for eng in engine_modules:
		if is_instance_valid(eng):
			eng.engine_start()
	if is_instance_valid(steering_module):
		steering_module.set_x(0.0)
		steering_module.set_y(0.0)
		steering_module.set_z(0.0)
	for fc in fuel_containers:
		if is_instance_valid(fc):
			fc.current_level = fc.MaxCapacity
	if is_instance_valid(battery_container):
		battery_container.current_level = battery_container.MaxCapacity
	if is_instance_valid(aircraft):
		aircraft.angular_velocity = Vector3.ZERO
		var spawn_basis = Basis(Vector3(0.707107, 0, 0.707107), Vector3(0, 1, 0), Vector3(-0.707107, 0, 0.707107))
		if takeoff_phase:
			aircraft.linear_velocity = Vector3.ZERO
			aircraft.global_transform = Transform3D(spawn_basis, Vector3(552.0, 770.0, 2509.0))
			throttle_level = 1.0
			for eng in engine_modules:
				if is_instance_valid(eng):
					eng.engine_set_power(1.0)
		else:
			aircraft.linear_velocity = Vector3(0.0, 0.0, -50.0)
			aircraft.global_transform = Transform3D(spawn_basis, Vector3(552.0, 1068.0, 2509.0))
			for eng in engine_modules:
				if is_instance_valid(eng):
					eng.engine_set_power(0.5)


func reset_episode():
	if explosion_instance and is_instance_valid(explosion_instance):
		explosion_instance.queue_free()
		explosion_instance = null
	has_landed_safely = false
	engine_was_running = false
	takeoff_phase = false
	if not test_mode:
		episode_count += 1
		if episode_count % SAVE_INTERVAL == 0:
			save_weights()
	prev_action = -1
	is_landing_mode = false
	train_counter = 0
	chunk_ptr = -1
	nstep_buffer.clear()
	if test_mode:
		test_ep_alt_sum = 0.0
		test_ep_stall_acc = 0
	initialize_aircraft()
	prev_state = get_state()


func _on_Aircraft_crashed(_iv):
	if is_done:
		return
	is_done = true
	var n = template_explosion.instantiate()
	add_child(n)
	n.global_transform.origin = aircraft.global_transform.origin
	n.explode()
	explosion_instance = n


func _on_Aircraft_parked():
	if is_done or takeoff_phase:
		return
	has_landed_safely = true
	is_done = true
	if not is_instance_valid(aircraft):
		return
	if $FuelArea.overlaps_body(aircraft):
		is_reloading_fuel = true
		print("RELOADING FUEL")
	if $BatteryArea.overlaps_body(aircraft):
		is_charging_battery = true
		print("RECHARGING BATTERY")


func _on_Aircraft_moved():
	if is_reloading_fuel:
		is_reloading_fuel = false
		print("REFUEL STOPPED")
	if is_charging_battery:
		is_charging_battery = false
		print("CHARGE STOPPED")


func get_obstacle_distances() -> Array:
	var space = get_world_3d().direct_space_state
	var origin = aircraft.global_transform.origin
	var fwd = -aircraft.global_transform.basis.z
	var max_dist = 300.0
	var angles = [0.0, deg_to_rad(15.0), deg_to_rad(-15.0)]
	var dists = []
	for a in angles:
		var dir = fwd.rotated(Vector3.UP, a)
		var query = PhysicsRayQueryParameters3D.create(origin, origin + dir * max_dist)
		query.exclude = [aircraft]
		var hit = space.intersect_ray(query)
		if hit:
			dists.append(hit.position.distance_to(origin) / max_dist)
		else:
			dists.append(1.0)
	return dists


func get_state() -> Array:
	if not is_instance_valid(aircraft):
		return _zero_state()
	var s = aircraft.forward_air_speed / 100.0
	var a = max(aircraft.local_altitude, 0.0) / 500.0
	var v = aircraft.linear_velocity.y / 50.0
	var st = 1.0 if aircraft.is_stalled else 0.0
	var p = steering_module.axis_x if is_instance_valid(steering_module) else 0.0
	var r = steering_module.axis_z if is_instance_valid(steering_module) else 0.0
	var obs = get_obstacle_distances()
	var total_fuel = 0.0
	var total_max = 0.0
	for fc in fuel_containers:
		if is_instance_valid(fc):
			total_fuel += fc.current_level
			total_max += fc.MaxCapacity
	var fuel_ratio = total_fuel / total_max if total_max > 0 else 0.0
	var battery_ratio = 0.0
	if is_instance_valid(battery_container):
		battery_ratio = battery_container.current_level / battery_container.MaxCapacity
	var state = [s, a, sin(p), cos(p), sin(r), cos(r), v, st, obs[0], obs[1], obs[2], fuel_ratio, battery_ratio]
	for i in range(state.size()):
		state[i] = clamp(state[i], -1.0, 1.0)
	return state


func _zero_state() -> Array:
	var z = []
	z.resize(STATE_DIM)
	for i in range(STATE_DIM):
		z[i] = 0.0
	return z


const TARGET_ALT = 200.0
const ALT_SIGMA = 80.0
const ALT_FLOOR = 50.0


func compute_reward() -> float:
	if not is_instance_valid(aircraft):
		return -1.0
	var alt = max(aircraft.local_altitude, 0.0)
	var spd = aircraft.forward_air_speed
	var vs = aircraft.linear_velocity.y
	var engine_on = false
	for eng in engine_modules:
		if is_instance_valid(eng) and eng.is_engine_working:
			engine_on = true
			break
	var gear_down = landing_gear_module.is_deployed if is_instance_valid(landing_gear_module) else false
	var total_fuel = 0.0
	var total_max = 0.0
	for fc in fuel_containers:
		if is_instance_valid(fc):
			total_fuel += fc.current_level
			total_max += fc.MaxCapacity
	var fuel_ratio = total_fuel / total_max if total_max > 0 else 0.0
	var battery_ratio = 0.0
	if is_instance_valid(battery_container):
		battery_ratio = battery_container.current_level / battery_container.MaxCapacity
	var rw = 0.0

	var on_ground = alt < 15.0

	if has_landed_safely:
		rw += 15.0
	elif on_ground:
		rw -= 0.5
		if spd < 3.0 and not is_done:
			rw += 2.0
	else:
		rw += min(spd * 0.002, 0.3)
		if fuel_ratio > 0.05 and engine_on:
			var alt_dev = abs(alt - TARGET_ALT)
			rw += exp(-(alt_dev * alt_dev) / (2.0 * ALT_SIGMA * ALT_SIGMA)) * 3.0
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

	if alt > 50.0 and spd > 30.0 and not aircraft.is_stalled and engine_on:
		rw += 0.2
	if aircraft.is_stalled:
		rw -= 1.0
	if aircraft.local_g_force > 5.0:
		rw -= 0.5
	rw += fuel_ratio * 0.5
	rw += battery_ratio * 0.3
	if is_reloading_fuel:
		rw += 1.0
	if is_charging_battery:
		rw += 0.5
	return rw


func apply_action(a: int):
	if not is_instance_valid(aircraft) or engine_modules.is_empty() or not is_instance_valid(steering_module):
		return
	match a:
		0: pass
		1: pitch_level = min(pitch_level + 0.2, 1.0)
		2: pitch_level = max(pitch_level - 0.2, -1.0)
		3: roll_level = max(roll_level - 0.2, -1.0)
		4: roll_level = min(roll_level + 0.2, 1.0)
		5: throttle_level = min(throttle_level + 0.1, 1.0)
		6: throttle_level = max(throttle_level - 0.1, 0.0)
	for eng in engine_modules:
		if is_instance_valid(eng):
			eng.engine_set_power(throttle_level)
	steering_module.set_x(pitch_level)
	steering_module.set_z(roll_level)


func forward(x: PackedFloat32Array, w1p: PackedFloat32Array, b1p: PackedFloat32Array, wAp: PackedFloat32Array, bAp: PackedFloat32Array, wVp: PackedFloat32Array, bVp: float) -> Array:
	var h = PackedFloat32Array()
	h.resize(HIDDEN)
	for hi in range(HIDDEN):
		var s = b1p[hi]
		for j in range(STATE_DIM):
			s += w1p[hi * STATE_DIM + j] * x[j]
		h[hi] = max(s, 0.0)

	var V = bVp
	for hi in range(HIDDEN):
		V += wVp[hi] * h[hi]

	var A = PackedFloat32Array()
	A.resize(ACTION_DIM)
	for ai in range(ACTION_DIM):
		var s = bAp[ai]
		for hi in range(HIDDEN):
			s += wAp[ai * HIDDEN + hi] * h[hi]
		A[ai] = s

	var meanA = 0.0
	for ai in range(ACTION_DIM):
		meanA += A[ai]
	meanA /= ACTION_DIM

	var Q = PackedFloat32Array()
	Q.resize(ACTION_DIM)
	for ai in range(ACTION_DIM):
		Q[ai] = V + A[ai] - meanA

	return [h, V, A, Q]


func predict_q(state: Array) -> PackedFloat32Array:
	var x = PackedFloat32Array(state)
	var result = forward(x, w1, b1, wA, bA, wV, bV)
	return result[3]


func select_action(state: Array) -> int:
	if randf() < epsilon:
		return randi() % ACTION_DIM
	var q = predict_q(state)
	var best = 0
	for i in range(1, q.size()):
		if q[i] > q[best]:
			best = i
	return best


func push_replay(state: Array, action: int, reward: float, next_state: Array, done: bool):
	nstep_buffer.append([state, action, reward, next_state, done])
	if nstep_buffer.size() > N_STEPS:
		nstep_buffer.pop_front()

	var push_ready = done or nstep_buffer.size() == N_STEPS
	if not push_ready:
		return

	var G = 0.0
	var final_idx = nstep_buffer.size() - 1
	for i in range(nstep_buffer.size()):
		G += pow(GAMMA, i) * nstep_buffer[i][2]
		if nstep_buffer[i][4]:
			final_idx = i
			break

	var first = nstep_buffer[0]
	var last = nstep_buffer[final_idx]
	var n_actual = final_idx + 1
	var p = pow(max_priority, PRIORITY_ALPHA)
	sumtree.add([first[0], first[1], G, last[3], last[4], n_actual], p)

	if done:
		nstep_buffer.clear()


func sample_batch(batch_size: int) -> Array:
	var result = sumtree.sample(batch_size)
	var batch = result[0]
	var indices = result[1]
	var priorities = result[2]

	if batch.is_empty():
		return [[], [], []]

	var total_p = sumtree.total()
	var n = sumtree.size
	var beta = min(1.0, PRIORITY_BETA_START + step_count * (1.0 - PRIORITY_BETA_START) / float(PRIORITY_BETA_STEPS))
	var weights = PackedFloat32Array()
	weights.resize(batch_size)
	for i in range(batch_size):
		var prob = priorities[i] / total_p
		weights[i] = pow(1.0 / (n * prob + 1e-8), beta)

	return [batch, indices, weights]


func train_step():
	if sumtree.size < BATCH_SIZE:
		return

	if chunk_ptr < 0:
		var batch_result = sample_batch(BATCH_SIZE)
		chunk_batch = batch_result[0]
		chunk_indices = batch_result[1]
		chunk_is_weights = batch_result[2]

		chunk_dw1.resize(HIDDEN * STATE_DIM)
		chunk_db1.resize(HIDDEN)
		chunk_dwA.resize(ACTION_DIM * HIDDEN)
		chunk_dbA.resize(ACTION_DIM)
		chunk_dwV.resize(HIDDEN)
		chunk_dw1.fill(0.0)
		chunk_db1.fill(0.0)
		chunk_dwA.fill(0.0)
		chunk_dbA.fill(0.0)
		chunk_dwV.fill(0.0)
		chunk_dbV = 0.0

		chunk_ptr = 0

	var chunk_end = mini(chunk_ptr + CHUNK_SIZE, BATCH_SIZE)

	for i in range(chunk_ptr, chunk_end):
		var b = chunk_batch[i]
		var s: Array = b[0]
		var a: int = b[1]
		var r: float = b[2]
		var ns: Array = b[3]
		var d: bool = b[4]
		var n_actual: int = b[5] if b.size() > 5 else 1

		var x_s = PackedFloat32Array(s)
		var x_ns = PackedFloat32Array(ns)

		var fwd = forward(x_s, w1, b1, wA, bA, wV, bV)
		var h = fwd[0]
		var V_s = fwd[1]
		var A_s = fwd[2]
		var Q_s = fwd[3]

		var fwd_on = forward(x_ns, w1, b1, wA, bA, wV, bV)
		var Q_on = fwd_on[3]
		var best_a = 0
		for j in range(1, ACTION_DIM):
			if Q_on[j] > Q_on[best_a]:
				best_a = j

		var fwd_tg = forward(x_ns, w1t, b1t, wAt, bAt, wVt, bVt)
		var Q_tg = fwd_tg[3]
		var target = r + GAMMA_POW[n_actual] * Q_tg[best_a] * (0.0 if d else 1.0)

		var td_err = abs(Q_s[a] - target) + 1e-6
		var p = pow(td_err, PRIORITY_ALPHA)
		sumtree.set_priority(chunk_indices[i], p)
		if td_err > max_priority:
			max_priority = td_err

		var dQ = 2.0 * (Q_s[a] - target) / float(BATCH_SIZE)
		dQ *= chunk_is_weights[i]

		var dV = dQ
		var dA = PackedFloat32Array()
		dA.resize(ACTION_DIM)
		var invN = 1.0 / ACTION_DIM
		for ai in range(ACTION_DIM):
			dA[ai] = -dQ * invN
		dA[a] += dQ

		for hi in range(HIDDEN):
			chunk_dwV[hi] += dV * h[hi]
		chunk_dbV += dV

		for ai in range(ACTION_DIM):
			var dai = dA[ai]
			for hi in range(HIDDEN):
				chunk_dwA[ai * HIDDEN + hi] += dai * h[hi]
			chunk_dbA[ai] += dai

		var grad_h = PackedFloat32Array()
		grad_h.resize(HIDDEN)
		for hi in range(HIDDEN):
			var ssum = dV * wV[hi]
			for ai in range(ACTION_DIM):
				ssum += dA[ai] * wA[ai * HIDDEN + hi]
			grad_h[hi] = ssum if h[hi] > 0 else 0.0

		for hi in range(HIDDEN):
			var ghi = grad_h[hi]
			for j in range(STATE_DIM):
				chunk_dw1[hi * STATE_DIM + j] += ghi * s[j]
			chunk_db1[hi] += ghi

	chunk_ptr = chunk_end

	if chunk_ptr >= BATCH_SIZE:
		if GRAD_CLIP > 0:
			for i in range(chunk_dw1.size()):
				chunk_dw1[i] = clamp(chunk_dw1[i], -GRAD_CLIP, GRAD_CLIP)
			for i in range(chunk_db1.size()):
				chunk_db1[i] = clamp(chunk_db1[i], -GRAD_CLIP, GRAD_CLIP)
			for i in range(chunk_dwA.size()):
				chunk_dwA[i] = clamp(chunk_dwA[i], -GRAD_CLIP, GRAD_CLIP)
			for i in range(chunk_dbA.size()):
				chunk_dbA[i] = clamp(chunk_dbA[i], -GRAD_CLIP, GRAD_CLIP)
			for i in range(chunk_dwV.size()):
				chunk_dwV[i] = clamp(chunk_dwV[i], -GRAD_CLIP, GRAD_CLIP)
			chunk_dbV = clamp(chunk_dbV, -GRAD_CLIP, GRAD_CLIP)

		step_count += 1
		adam_step += 1
		var current_lr = max(LR_MIN, LR_INIT * pow(LR_DECAY, step_count))
		var b1_corr = 1.0 - pow(ADAM_BETA1, adam_step)
		var b2_corr = 1.0 - pow(ADAM_BETA2, adam_step)

		_apply_adam(chunk_dw1, w1, m_w1, v_w1, current_lr, b1_corr, b2_corr)
		_apply_adam(chunk_db1, b1, m_b1, v_b1, current_lr, b1_corr, b2_corr)
		_apply_adam(chunk_dwA, wA, m_wA, v_wA, current_lr, b1_corr, b2_corr)
		_apply_adam(chunk_dbA, bA, m_bA, v_bA, current_lr, b1_corr, b2_corr)
		_apply_adam(chunk_dwV, wV, m_wV, v_wV, current_lr, b1_corr, b2_corr)
		m_bV = ADAM_BETA1 * m_bV + (1.0 - ADAM_BETA1) * chunk_dbV
		v_bV = ADAM_BETA2 * v_bV + (1.0 - ADAM_BETA2) * chunk_dbV * chunk_dbV
		var mbV_hat = m_bV / b1_corr
		var vbV_hat = v_bV / b2_corr
		bV -= current_lr * mbV_hat / (sqrt(vbV_hat) + ADAM_EPS)

		var tau = TAU
		_polyak(w1, w1t, tau)
		_polyak(b1, b1t, tau)
		_polyak(wA, wAt, tau)
		_polyak(bA, bAt, tau)
		_polyak(wV, wVt, tau)
		bVt = tau * bV + (1.0 - tau) * bVt

		epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

		chunk_ptr = -1


func _apply_adam(dw: PackedFloat32Array, w: PackedFloat32Array, m: PackedFloat32Array, v: PackedFloat32Array, lr: float, b1c: float, b2c: float):
	for i in range(w.size()):
		var g = dw[i]
		m[i] = ADAM_BETA1 * m[i] + (1.0 - ADAM_BETA1) * g
		v[i] = ADAM_BETA2 * v[i] + (1.0 - ADAM_BETA2) * g * g
		var m_hat = m[i] / b1c
		var v_hat = v[i] / b2c
		w[i] -= lr * m_hat / (sqrt(v_hat) + ADAM_EPS)


func _polyak(src: PackedFloat32Array, dst: PackedFloat32Array, tau: float):
	for i in range(src.size()):
		dst[i] = tau * src[i] + (1.0 - tau) * dst[i]


func do_takeoff_step():
	for eng in engine_modules:
		if is_instance_valid(eng):
			eng.engine_set_power(1.0)
	var speed = aircraft.forward_air_speed
	var alt = aircraft.local_altitude

	if alt > 70.0 and speed > 30.0:
		takeoff_phase = false
		episode_count = 1
		print("Takeoff complete — DQN taking over")
		if is_instance_valid(landing_gear_module):
			landing_gear_module.stow()
		print("Episode %d | reward: %.1f | steps: %d | eps: %.3f | best: %.1f" % [episode_count, episode_reward, episode_step, epsilon, best_reward])
		prev_state = get_state()
		return

	if speed > 30.0:
		pitch_level = 0.3
	else:
		pitch_level = 0.0
	roll_level = 0.0
	steering_module.set_x(pitch_level)
	steering_module.set_z(roll_level)


func _do_landing():
	if not is_instance_valid(aircraft) or engine_modules.is_empty() or not is_instance_valid(steering_module):
		return

	var pos = aircraft.global_transform.origin
	var fwd = -aircraft.global_transform.basis.z
	var alt = max(aircraft.local_altitude, 0.0)

	var home_pos = Vector3(552.0, 0.0, 2509.0)
	var to_home = home_pos - pos
	to_home.y = 0
	var dist = to_home.length()

	if dist > 1.0:
		to_home = to_home.normalized()
		var cross = fwd.cross(to_home).y
		roll_level = clamp(cross * 3.0, -0.6, 0.6)
	elif alt > 20:
		roll_level = 0.0

	if alt > 80:
		pitch_level = -0.15
		throttle_level = 0.3
	elif alt > 15:
		pitch_level = -0.08
		throttle_level = 0.2
		if is_instance_valid(landing_gear_module) and not landing_gear_module.is_deployed and not landing_gear_module.is_deploying:
			landing_gear_module.deploy()
	elif alt > 5:
		pitch_level = 0.2
		throttle_level = 0.05
	else:
		pitch_level = 0.0
		throttle_level = 0.0

	if alt < 30:
		roll_level *= 0.3

	steering_module.set_x(pitch_level)
	steering_module.set_z(roll_level)
	for eng in engine_modules:
		if is_instance_valid(eng):
			eng.engine_set_power(throttle_level)


func _physics_process(delta):
	if is_reloading_fuel and is_instance_valid(aircraft):
		var amount_per_second = REFUEL_RATE
		var is_aircraft_full = aircraft.load_energy("fuel", amount_per_second * delta)
		if is_aircraft_full:
			is_reloading_fuel = false
			print("REFUEL COMPLETE")
	if is_charging_battery and is_instance_valid(aircraft):
		var charge_per_second = REFUEL_RATE
		var is_aircraft_full = aircraft.load_energy("battery", charge_per_second * delta)
		if is_aircraft_full:
			is_charging_battery = false
			print("CHARGE COMPLETE")

	if not is_instance_valid(aircraft):
		return

	if engine_modules.is_empty() or not is_instance_valid(steering_module):
		return

	var any_engine_running = false
	for eng in engine_modules:
		if is_instance_valid(eng) and eng.is_engine_working:
			any_engine_running = true
			break
	if any_engine_running:
		engine_was_running = true
	elif engine_was_running:
		if is_instance_valid(landing_gear_module) and not landing_gear_module.is_deployed and not landing_gear_module.is_deploying:
			landing_gear_module.deploy()

	if is_done:
		if has_landed_safely:
			episode_reward += 15.0
		_on_episode_end()
		reset_episode()
		return

	if takeoff_phase:
		if test_mode:
			takeoff_phase = false
			episode_step = 0
			episode_reward = 0.0
			prev_action = -1
			prev_state = get_state()
		else:
			do_takeoff_step()
		return

	frame_counter += 1

	if frame_counter < FRAMES_PER_STEP:
		return

	frame_counter = 0

	if not is_landing_mode and episode_step >= int(MAX_EPISODE_STEPS * 0.85):
		is_landing_mode = true
		print("Landing approach at step %d" % episode_step)

	if is_landing_mode:
		_do_landing()
		episode_step += 1
		episode_reward += compute_reward()
		var done = is_done or episode_step >= MAX_EPISODE_STEPS
		if done:
			_on_episode_end()
			reset_episode()
	else:
		var state = get_state()
		var action = select_action(state)
		apply_action(action)

		if test_mode:
			test_ep_alt_sum += aircraft.local_altitude
			if aircraft.is_stalled:
				test_ep_stall_acc += 1

		var next_state = get_state()
		var reward = compute_reward()
		if prev_action >= 0 and action != prev_action:
			reward -= 0.02
		prev_action = action
		episode_step += 1
		episode_reward += reward
		var done = is_done or episode_step >= MAX_EPISODE_STEPS

		if test_mode:
			epsilon = 0.0
		else:
			push_replay(state, action, reward, next_state, done)
			train_counter += 1
			if train_counter >= TRAIN_INTERVAL:
				train_counter = 0
				train_step()

		if done:
			_on_episode_end()
			reset_episode()


func _on_BtnBack_pressed():
	get_tree().change_scene_to_file("res://example/ExampleList.tscn")
