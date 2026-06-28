extends Node

@export var AircraftNode: NodePath
@export var DQNAgentNode: NodePath
@export var ExportIntervalFrames: int = 3
@export var WeightSaveIntervalEpisodes: int = 10
@export var MaxCsvRows: int = 50000
@export var DQNStateDim: int = 12
@export var DQNActionDim: int = 7

var aircraft: Aircraft = null
var dqn_agent = null
var aircraft_ref: Aircraft = null  # direct reference, set by training script
var engine_module = null
var steering_module = null
var flaps_module = null
var landing_gear_module = null
var energy_container = null
var instrument_attitude = null

var _frame_counter: int = 0
var _row_count: int = 0
var _csv_file: FileAccess = null
var _csv_path: String = ""
var _output_dir: String = ""
var _weights_dir: String = ""
var _episode_count: int = 0
var _prev_episode: int = -1
var _header_written: bool = false
var _fields: Array[String] = []


func _ready():
	await get_tree().process_frame
	aircraft = aircraft_ref
	if not aircraft and AircraftNode:
		aircraft = get_node_or_null(AircraftNode)
	if not aircraft:
		print("TelemetryCsvExporter: Aircraft node not configured — skipping")
		return
	engine_module = aircraft.find_modules_by_type("engine").pop_front()
	steering_module = aircraft.find_modules_by_type("steering").pop_front()
	flaps_module = aircraft.find_modules_by_type("flaps").pop_front()
	landing_gear_module = aircraft.find_modules_by_type("landing_gear").pop_front()
	energy_container = aircraft.find_modules_by_type("energy_container").pop_front()
	instrument_attitude = aircraft.find_modules_by_type("instrument_attitude").pop_front()

	if DQNAgentNode:
		dqn_agent = get_node_or_null(DQNAgentNode)

	var project_root = ProjectSettings.globalize_path("res://")
	_output_dir = project_root + "telemetry/csv"
	_weights_dir = project_root + "telemetry/weights"

	var dir = DirAccess.open(project_root)
	if dir:
		if not dir.dir_exists("telemetry"):
			dir.make_dir("telemetry")
		if not dir.dir_exists("telemetry/csv"):
			dir.make_dir("telemetry/csv")
		if not dir.dir_exists("telemetry/weights"):
			dir.make_dir("telemetry/weights")

	var timestamp = Time.get_datetime_string_from_system().replace(":", "-")
	_csv_path = _output_dir + "/telemetry_" + timestamp + ".csv"
	_csv_file = FileAccess.open(_csv_path, FileAccess.WRITE)
	if _csv_file:
		print("TelemetryCsvExporter: writing to " + _csv_path)
	else:
		push_error("TelemetryCsvExporter: failed to open " + _csv_path)

	_build_fields()
	_write_header()


func _build_fields():
	_fields = [
		"t_ms", "alt_m", "spd_ms", "fwd_spd_ms", "vspd_ms",
		"g_force", "load_factor", "stall",
		"px", "py", "pz",
		"roll_deg", "pitch_deg", "hdg_deg",
		"epow", "eact", "flap", "gear", "fuel",
	]
	if dqn_agent != null:
		_fields.append("action")
		for i in range(DQNStateDim):
			_fields.append("state_%d" % i)
		for i in range(DQNActionDim):
			_fields.append("Q_%d" % i)
		_fields.append("reward")
		_fields.append("epsilon")
		_fields.append("episode")
		_fields.append("step_count")


func _write_header():
	if _csv_file and not _header_written:
		_csv_file.store_line(",".join(_fields))
		_header_written = true


func _physics_process(_delta):
	if not aircraft or not is_instance_valid(aircraft):
		return
	# Auto-write only when no DQN agent is configured (no DQN data to log).
	# When DQN agent is present, the training script drives writes via set_dqn_data().
	if dqn_agent and is_instance_valid(dqn_agent):
		return
	_frame_counter += 1
	if _frame_counter < ExportIntervalFrames:
		return
	_frame_counter = 0

	if _csv_file == null:
		return

	var data = _collect_data()
	if data.is_empty():
		return

	var row = PackedStringArray()
	for field in _fields:
		var val = data.get(field, "")
		row.append(str(val))
	_csv_file.store_line(",".join(row))
	_row_count += 1

	if _row_count >= MaxCsvRows:
		_rotate_file()


func _collect_data() -> Dictionary:
	var pos = aircraft.global_transform.origin
	var vel = aircraft.linear_velocity

	var d = {
		"t_ms": Time.get_ticks_msec(),
		"alt_m": aircraft.local_altitude,
		"spd_ms": aircraft.air_velocity,
		"fwd_spd_ms": aircraft.forward_air_speed,
		"vspd_ms": vel.y,
		"g_force": aircraft.local_g_force,
		"load_factor": aircraft.local_load_factor,
		"stall": 1 if aircraft.is_stalled else 0,
		"px": pos.x,
		"py": pos.y,
		"pz": pos.z,
	}

	if instrument_attitude and is_instance_valid(instrument_attitude):
		d["roll_deg"] = rad_to_deg(instrument_attitude.current_roll)
		d["pitch_deg"] = rad_to_deg(instrument_attitude.current_pitch)
		d["hdg_deg"] = rad_to_deg(instrument_attitude.current_bearing)

	if engine_module and is_instance_valid(engine_module):
		d["epow"] = engine_module.current_power
		d["eact"] = 1 if engine_module.is_engine_working else 0

	if flaps_module and is_instance_valid(flaps_module):
		d["flap"] = flaps_module.flap_position

	if landing_gear_module and is_instance_valid(landing_gear_module):
		d["gear"] = 1 if landing_gear_module.is_deployed else 0

	if energy_container and is_instance_valid(energy_container):
		var soc = 0.0
		if energy_container.MaxCapacity > 0:
			soc = energy_container.current_level / energy_container.MaxCapacity
		d["fuel"] = soc

	if dqn_agent and is_instance_valid(dqn_agent):
		# DQN data added by the training script via set_dqn_data()
		pass

	return d


func set_dqn_data(action: int, state, q_values, reward: float, epsilon: float, episode: int, step: int):
	if _csv_file == null:
		return
	var data = _collect_data()
	data["action"] = action
	for i in range(state.size()):
		if i < DQNStateDim:
			data["state_%d" % i] = state[i]
	for i in range(q_values.size()):
		if i < DQNActionDim:
			data["Q_%d" % i] = q_values[i]
	data["reward"] = reward
	data["epsilon"] = epsilon
	data["episode"] = episode
	data["step_count"] = step

	var row = PackedStringArray()
	for field in _fields:
		var val = data.get(field, "")
		row.append(str(val))
	_csv_file.store_line(",".join(row))
	_row_count += 1
	if episode != _episode_count and episode % WeightSaveIntervalEpisodes == 0:
		_save_weight_snapshot()
	_episode_count = episode

	if _row_count >= MaxCsvRows:
		_rotate_file()


func _save_weight_snapshot():
	if not dqn_agent or not is_instance_valid(dqn_agent):
		return
	var weights = dqn_agent.get_weights_online()
	if weights.is_empty():
		return
	var layer_names = ["w1", "b1", "w2", "b2", "wA", "bA", "wV", "bV"]
	var timestamp = Time.get_datetime_string_from_system().replace(":", "-")
	var ep_str = "ep%04d" % _episode_count

	for i in range(min(weights.size(), layer_names.size())):
		var data = weights[i]
		var wpath = _weights_dir + "/%s_%s_%s.csv" % [timestamp, ep_str, layer_names[i]]
		var f = FileAccess.open(wpath, FileAccess.WRITE)
		if f:
			if data is PackedFloat32Array:
				f.store_line("index,value")
				for idx in range(data.size()):
					f.store_line("%d,%s" % [idx, String.num(data[idx], 10)])
			elif data is float or data is int:
				f.store_line("value")
				f.store_line(String.num(data as float, 10))
			f.close()
	print("Weight snapshot saved (episode %d)" % _episode_count)


func _rotate_file():
	if _csv_file:
		_csv_file.close()
	var timestamp = Time.get_datetime_string_from_system().replace(":", "-")
	_csv_path = _output_dir + "/telemetry_" + timestamp + ".csv"
	_csv_file = FileAccess.open(_csv_path, FileAccess.WRITE)
	_row_count = 0
	_header_written = false
	_write_header()
	print("TelemetryCsvExporter: rotated to " + _csv_path)


func _exit_tree():
	if _csv_file:
		_csv_file.close()
