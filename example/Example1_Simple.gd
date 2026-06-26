extends Node3D

const STATE_DIM = 12
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
const SAVE_PATH = "user://dqn_weights.save"
const META_PATH = "user://dqn_meta.save"
const TRAIN_INTERVAL = 2
const SAVE_INTERVAL = 50
const SAVE_VERSION = 5
const TEST_EPISODES = 10
const TEST_REPORT_PATH = "user://test_report.txt"
const TEST_REPORT_HTML_PATH = "user://test_report.html"

var template_explosion = preload("res://example/scenes/Explosion/Explosion.tscn")
var explosion_instance = null

var takeoff_phase = true
var has_landed_safely = false
var engine_was_running = false

@onready var aircraft = get_node("Aircraft")
@onready var engine_module: AircraftModule_Engine
@onready var steering_module: AircraftModule_Steering
@onready var landing_gear_module: AircraftModule_LandingGear
@onready var energy_container: AircraftModule_EnergyContainer

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


func _ready():
	aircraft.connect("crashed", Callable(self, "_on_Aircraft_crashed"))
	aircraft.connect("parked", Callable(self, "_on_Aircraft_parked"))
	aircraft.connect("moved", Callable(self, "_on_Aircraft_moved"))
	$Aircraft/Engine.connect("update_interface", Callable($Aircraft/Model/MovingParts/Engine, "_on_Engine_update_interface"))
	$Aircraft/Steering.connect("update_interface", Callable($Aircraft/Model/MovingParts/Steering, "_on_Steering_update_interface"))
	$Aircraft/Flaps.connect("update_interface", Callable($Aircraft/Model/MovingParts/Flaps, "_on_Flaps_update_interface"))
	$Aircraft/LandingGear.connect("update_interface", Callable($Aircraft/Model/MovingParts/LandingGear, "_on_LandingGear_update_interface"))

	await get_tree().process_frame

	engine_module = aircraft.find_modules_by_type("engine").pop_front()
	steering_module = aircraft.find_modules_by_type("steering").pop_front()
	landing_gear_module = aircraft.find_modules_by_type("landing_gear").pop_front()
	energy_container = aircraft.find_modules_by_type("energy_container").pop_front()

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
		_write_weights_to("user://dqn_run_%d.save" % run_id)
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
		if fn.begins_with("dqn_run_") and fn.ends_with(".save"):
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
		print("Found best run model (reward %.1f)" % best_r)
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

	var report = "=== DQN TEST REPORT ===\n"
	report += "Model: dqn_run_%d.save\n" % run_id
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
	report += "Avg Alt:     %.0f m (target %d m)\n" % [avg_alt, TARGET_ALT]
	report += "Avg Stalls:  %.1f / ep\n" % avg_stalls
	report += "Landings:    %d / %d (%.0f%%)\n" % [landings, test_results.size(), success_rate]
	report += "Crashes:     %d / %d\n" % [crashes, test_results.size()]

	var file = FileAccess.open(TEST_REPORT_PATH, FileAccess.WRITE)
	if file:
		file.store_string(report)
		file.close()
		print("\nTest report saved to: %s" % TEST_REPORT_PATH)
		_generate_html_report(avg_r, min_r, max_r, avg_steps, avg_alt, avg_stalls, landings, crashes, success_rate)
	else:
		push_error("Failed to save test report")

	print("\n" + report)


func _generate_html_report(avg_r: float, min_r: float, max_r: float, avg_steps: float, avg_alt: float, avg_stalls: float, landings: int, crashes: int, success_rate: float):
	var svg_w = 800
	var svg_h = 300
	var margin_l = 60
	var margin_r = 20
	var margin_t = 30
	var margin_b = 50
	var plot_w = svg_w - margin_l - margin_r
	var plot_h = svg_h - margin_t - margin_b

	var ep_count = test_results.size()
	var max_r_chart = max_r
	var min_r_chart = min_r
	var r_range = max_r_chart - min_r_chart
	if r_range < 1.0:
		r_range = 1.0

	var svg_bars = ""
	for i in range(ep_count):
		var r = test_results[i]
		var x = margin_l + (float(i) + 0.15) * plot_w / ep_count
		var bar_w = plot_w / ep_count * 0.7
		var val_h = (r["reward"] - min_r_chart) / r_range * plot_h
		var y = margin_t + plot_h - val_h
		var color = "#22c55e" if r["landed"] else ("#ef4444" if r["crashed"] else "#f59e0b")
		svg_bars += "<rect x='%.1f' y='%.1f' width='%.1f' height='%.1f' fill='%s'/>\n" % [x, y, bar_w, val_h, color]

	var svg = "<svg xmlns='http://www.w3.org/2000/svg' width='%d' height='%d' style='background:#fff;border-radius:8px'>\n" % [svg_w, svg_h]
	svg += "<line x1='%.1f' y1='%d' x2='%.1f' y2='%d' stroke='#ccc' stroke-width='1'/>\n" % [margin_l, margin_t, margin_l, margin_t + plot_h]
	svg += "<line x1='%.1f' y1='%d' x2='%d' y2='%d' stroke='#ccc' stroke-width='1'/>\n" % [margin_l, margin_t + plot_h, svg_w - margin_r, margin_t + plot_h]
	var y_ticks = 5
	for t in range(y_ticks + 1):
		var frac = float(t) / y_ticks
		var val = min_r_chart + frac * r_range
		var y = margin_t + plot_h - frac * plot_h
		svg += "<text x='%.1f' y='%.1f' text-anchor='end' fill='#666' font-size='12'>%.0f</text>\n" % [margin_l - 8, y + 4, val]
		if t > 0 and t < y_ticks:
			svg += "<line x1='%.1f' y1='%.1f' x2='%d' y2='%.1f' stroke='#eee' stroke-width='1'/>\n" % [margin_l, y, svg_w - margin_r, y]
	for i in range(ep_count):
		var x = margin_l + (float(i) + 0.5) * plot_w / ep_count
		svg += "<text x='%.1f' y='%.1f' text-anchor='middle' fill='#666' font-size='11'>%d</text>\n" % [x, margin_t + plot_h + 18, i + 1]
	svg += "<text x='%d' y='%d' text-anchor='middle' fill='#666' font-size='12'>Episode</text>\n" % [svg_w / 2, margin_t + plot_h + 38]
	svg += "<text x='12' y='%d' text-anchor='middle' fill='#666' font-size='12' transform='rotate(-90,12,%d)'>Reward</text>\n" % [margin_t + plot_h / 2, margin_t + plot_h / 2]
	svg += svg_bars
	var lx = svg_w - 200
	var ly = margin_t + 5
	svg += "<rect x='%d' y='%d' width='12' height='12' fill='#22c55e'/><text x='%d' y='%d' fill='#666' font-size='12'>Landed</text>\n" % [lx, ly, lx + 16, ly + 11]
	svg += "<rect x='%d' y='%d' width='12' height='12' fill='#ef4444'/><text x='%d' y='%d' fill='#666' font-size='12'>Crashed</text>\n" % [lx, ly + 18, lx + 16, ly + 29]
	svg += "<rect x='%d' y='%d' width='12' height='12' fill='#f59e0b'/><text x='%d' y='%d' fill='#666' font-size='12'>Timeout</text>\n" % [lx, ly + 36, lx + 16, ly + 47]
	svg += "</svg>"

	var status_colors = {"LANDED": "#22c55e", "CRASH": "#ef4444", "TIMEOUT": "#f59e0b"}
	var ep_rows = ""
	for i in range(test_results.size()):
		var r = test_results[i]
		var status = "LANDED" if r["landed"] else ("CRASH" if r["crashed"] else "TIMEOUT")
		var sc = status_colors[status]
		ep_rows += "<tr><td>%d</td><td>%.1f</td><td>%d</td><td><span style='color:%s;font-weight:600'>%s</span></td><td>%.0f</td><td>%d</td></tr>\n" % [i + 1, r["reward"], r["steps"], sc, status, r["avg_alt"], r["stalls"]]

	var html = "<!DOCTYPE html>\n<html lang='en'>\n<head>\n<meta charset='UTF-8'>\n"
	html += "<meta name='viewport' content='width=device-width,initial-scale=1.0'>\n"
	html += "<title>DQN Test Report</title>\n"
	html += "<style>\n"
	html += "*{margin:0;padding:0;box-sizing:border-box}\n"
	html += "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f5f5f5;color:#333;padding:20px}\n"
	html += ".container{max-width:900px;margin:0 auto}\n"
	html += "h1{text-align:center;margin-bottom:5px;color:#1a1a2e}\n"
	html += ".subtitle{text-align:center;color:#666;margin-bottom:25px}\n"
	html += ".section{background:#fff;border-radius:8px;padding:20px;margin-bottom:20px;box-shadow:0 2px 4px rgba(0,0,0,0.1)}\n"
	html += "h2{margin-bottom:15px;color:#1a1a2e;border-bottom:2px solid #e0e0e0;padding-bottom:8px}\n"
	html += "table{width:100%;border-collapse:collapse;margin-bottom:10px}\n"
	html += "th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #e0e0e0}\n"
	html += "th{background:#f8f9fa;font-weight:600}\n"
	html += "tr:hover{background:#f1f3f5}\n"
	html += ".chart{text-align:center;margin:15px 0}\n"
	html += "</style>\n</head>\n<body>\n"
	html += "<div class='container'>\n"
	html += "<h1>DQN Flight Sim — Test Report</h1>\n"
	html += "<p class='subtitle'>Model: dqn_run_%d.save &middot; %d episodes</p>\n" % [run_id, ep_count]

	html += "<div class='section'><h2>Summary</h2>\n<table>\n"
	html += "<tr><th>Metric</th><th>Value</th></tr>\n"
	html += "<tr><td>Avg Reward</td><td>%.1f (min %.1f, max %.1f)</td></tr>\n" % [avg_r, min_r, max_r]
	html += "<tr><td>Avg Steps</td><td>%.0f / %d</td></tr>\n" % [avg_steps, MAX_EPISODE_STEPS]
	html += "<tr><td>Avg Altitude</td><td>%.0f m (target %d m)</td></tr>\n" % [avg_alt, TARGET_ALT]
	html += "<tr><td>Avg Stalls / ep</td><td>%.1f</td></tr>\n" % avg_stalls
	html += "<tr><td>Landings</td><td>%d / %d (%.0f%%)</td></tr>\n" % [landings, ep_count, success_rate]
	html += "<tr><td>Crashes</td><td>%d / %d</td></tr>\n" % [crashes, ep_count]
	html += "</table>\n</div>\n"

	html += "<div class='section'><h2>Per-Episode Rewards</h2>\n<div class='chart'>\n"
	html += svg + "\n</div>\n</div>\n"

	html += "<div class='section'><h2>Episode Details</h2>\n<table>\n"
	html += "<tr><th>#</th><th>Reward</th><th>Steps</th><th>Status</th><th>Avg Alt</th><th>Stalls</th></tr>\n"
	html += ep_rows
	html += "</table>\n</div>\n"

	html += "</div>\n</body>\n</html>"

	var hfile = FileAccess.open(TEST_REPORT_HTML_PATH, FileAccess.WRITE)
	if hfile:
		hfile.store_string(html)
		hfile.close()
		print("HTML report saved to: %s" % TEST_REPORT_HTML_PATH)
	else:
		push_error("Failed to save HTML report")


func initialize_aircraft():
	throttle_level = 0.5
	pitch_level = 0.0
	roll_level = 0.0
	episode_step = 0
	episode_reward = 0.0
	is_done = false

	if is_instance_valid(engine_module):
		engine_module.engine_start()
	if is_instance_valid(steering_module):
		steering_module.set_x(0.0)
		steering_module.set_y(0.0)
		steering_module.set_z(0.0)
	if is_instance_valid(energy_container):
		energy_container.current_level = energy_container.MaxCapacity
	if is_instance_valid(aircraft):
		aircraft.angular_velocity = Vector3.ZERO
		if takeoff_phase:
			aircraft.linear_velocity = Vector3.ZERO
			aircraft.global_transform.origin = Vector3(0.0, 2.0, -250.0)
			throttle_level = 1.0
			engine_module.engine_set_power(1.0)
		else:
			aircraft.linear_velocity = Vector3(0.0, 0.0, -50.0)
			aircraft.global_transform.origin = Vector3(0.0, 300.0, -200.0)
			engine_module.engine_set_power(0.5)


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


func _on_Aircraft_moved():
	pass


func get_obstacle_distances() -> Array:
	var space = get_world_3d().direct_space_state
	var origin = aircraft.global_transform.origin
	var fwd = -aircraft.global_transform.basis.z
	var max_dist = 200.0
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
	var fuel = energy_container.current_level / energy_container.MaxCapacity if is_instance_valid(energy_container) else 1.0
	var state = [s, a, sin(p), cos(p), sin(r), cos(r), v, st, obs[0], obs[1], obs[2], fuel]
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

func get_fuel_soc() -> float:
	if is_instance_valid(energy_container):
		return energy_container.current_level / energy_container.MaxCapacity
	return 1.0


func compute_reward() -> float:
	if not is_instance_valid(aircraft):
		return -1.0
	var alt = max(aircraft.local_altitude, 0.0)
	var spd = aircraft.forward_air_speed
	var vs = aircraft.linear_velocity.y
	var fuel = get_fuel_soc()
	var engine_on = engine_module.is_engine_working if is_instance_valid(engine_module) else true
	var gear_down = landing_gear_module.is_deployed if is_instance_valid(landing_gear_module) else false
	var rw = 0.0

	var on_ground = alt < 10.0

	if has_landed_safely:
		rw += 15.0
	elif on_ground:
		rw -= 0.5
		if spd < 3.0 and not is_done:
			rw += 2.0
	else:
		rw += min(spd * 0.002, 0.3)
		if fuel > 0.1 and engine_on:
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
	return rw


func apply_action(a: int):
	if not is_instance_valid(aircraft) or not is_instance_valid(engine_module) or not is_instance_valid(steering_module):
		return
	match a:
		0: pass
		1: pitch_level = min(pitch_level + 0.2, 1.0)
		2: pitch_level = max(pitch_level - 0.2, -1.0)
		3: roll_level = max(roll_level - 0.2, -1.0)
		4: roll_level = min(roll_level + 0.2, 1.0)
		5: throttle_level = min(throttle_level + 0.1, 1.0)
		6: throttle_level = max(throttle_level - 0.1, 0.0)
	engine_module.engine_set_power(throttle_level)
	steering_module.set_x(pitch_level)
	steering_module.set_z(roll_level)


func do_takeoff_step():
	engine_module.engine_set_power(1.0)
	var speed = aircraft.forward_air_speed
	var alt = aircraft.local_altitude

	if alt > 50.0 and speed > 30.0:
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
	if not is_instance_valid(aircraft) or not is_instance_valid(engine_module) or not is_instance_valid(steering_module):
		return

	var pos = aircraft.global_transform.origin
	var fwd = -aircraft.global_transform.basis.z
	var alt = max(aircraft.local_altitude, 0.0)

	var to_home = -pos
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
	engine_module.engine_set_power(throttle_level)


func _physics_process(_delta):
	if not is_instance_valid(aircraft):
		return

	if not is_instance_valid(engine_module) or not is_instance_valid(steering_module):
		return

	if engine_module.is_engine_working:
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
