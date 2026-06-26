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
const LR_INIT = 0.001
const LR_DECAY = 0.9995
const LR_MIN = 0.0001
const EPSILON_START = 1.0
const EPSILON_MIN = 0.01
const EPSILON_DECAY = 0.998
const GRAD_CLIP = 1.0
const SAVE_PATH = "user://dqn_complex_weights.save"
const META_PATH = "user://dqn_complex_meta.save"
const TRAIN_INTERVAL = 2
const SAVE_INTERVAL = 50
const SAVE_VERSION = 5
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

var episode_count = 0
var best_reward = -1e9

var agent: DQNRust

var epsilon = EPSILON_START
var prev_action = -1
var train_counter = 0

var run_id = 0
var _last_saved_best = -1e9

var test_mode = false
var test_results = []
var test_ep_alt_sum = 0.0
var test_ep_stall_acc = 0

var is_landing_mode = false

var is_reloading_fuel = false
var is_charging_battery = false

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

	agent = DQNRust.new()
	agent.init(STATE_DIM, ACTION_DIM, HIDDEN, REPLAY_CAPACITY, N_STEPS, GAMMA)
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
	file.store_var(agent.get_weights_online())
	file.store_var(epsilon)
	file.store_32(episode_count)
	file.store_32(agent.get_step_count())
	file.store_var(best_reward)
	file.store_var(agent.get_adam_state())
	file.close()


func load_weights(path := SAVE_PATH) -> bool:
	if not FileAccess.file_exists(path):
		return false
	var file = FileAccess.open(path, FileAccess.READ)
	if not file:
		return false
	var ver = file.get_32()
	if ver < 5:
		file.close()
		return false
	agent.set_weights_online(file.get_var())
	epsilon = file.get_var()
	episode_count = file.get_32()
	agent.set_step_count(file.get_32())
	best_reward = file.get_var()
	agent.set_adam_state(file.get_var())
	file.close()
	agent.copy_to_target()
	print("Weights loaded (episode %d, step %d, best_reward %.1f)" % [episode_count, agent.get_step_count(), best_reward])
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
			if ver >= 5:
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

		if test_mode:
			agent.set_epsilon(0.0)
		else:
			agent.set_epsilon(epsilon)

		var action = agent.select_action(PackedFloat32Array(state))
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

		if not test_mode:
			agent.push_replay(PackedFloat32Array(state), action, reward, PackedFloat32Array(next_state), done)
			train_counter += 1
			if train_counter >= TRAIN_INTERVAL and agent.get_replay_size() >= BATCH_SIZE:
				train_counter = 0
				var current_lr = max(LR_MIN, LR_INIT * pow(LR_DECAY, agent.get_step_count()))
				agent.train(BATCH_SIZE, GAMMA, GRAD_CLIP, current_lr)
				epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

		if done:
			_on_episode_end()
			reset_episode()


func _on_BtnBack_pressed():
	get_tree().change_scene_to_file("res://example/ExampleList.tscn")
