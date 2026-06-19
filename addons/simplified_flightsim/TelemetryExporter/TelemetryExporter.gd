extends Node

@export var AircraftNode: NodePath
@export var ExportIntervalFrames: int = 3

var aircraft: Aircraft = null
var engine_module = null
var steering_module = null
var flaps_module = null
var landing_gear_module = null
var energy_container = null
var instrument_attitude = null

var _frame_counter: int = 0
var _output_path: String = ""
var _output_dir: String = ""

func _ready():
	await get_tree().process_frame
	aircraft = get_node_or_null(AircraftNode)
	if not aircraft:
		printerr("TelemetryExporter: Aircraft node not found")
		return
	engine_module = aircraft.find_modules_by_type("engine").pop_front()
	steering_module = aircraft.find_modules_by_type("steering").pop_front()
	flaps_module = aircraft.find_modules_by_type("flaps").pop_front()
	landing_gear_module = aircraft.find_modules_by_type("landing_gear").pop_front()
	energy_container = aircraft.find_modules_by_type("energy_container").pop_front()
	instrument_attitude = aircraft.find_modules_by_type("instrument_attitude").pop_front()

	var project_root = ProjectSettings.globalize_path("res://")
	_output_dir = project_root + "telemetry"
	_output_path = _output_dir + "/telemetry.jsonl"

	var dir = DirAccess.open(project_root)
	if dir and not dir.dir_exists("telemetry"):
		dir.make_dir("telemetry")

	var f = FileAccess.open(_output_path, FileAccess.WRITE)
	if f:
		f.store_line("")
		f.close()


func _physics_process(_delta):
	if not aircraft or not is_instance_valid(aircraft):
		return
	_frame_counter += 1
	if _frame_counter < ExportIntervalFrames:
		return
	_frame_counter = 0

	var data = _collect_data()
	var json_str = JSON.stringify(data)
	var f = FileAccess.open(_output_path, FileAccess.READ_WRITE)
	if f:
		f.seek_end()
		f.store_line(json_str)
		f.close()


func _collect_data() -> Dictionary:
	var pos = aircraft.global_transform.origin
	var vel = aircraft.linear_velocity

	var d = {
		"t": Time.get_ticks_msec(),
		"alt": aircraft.local_altitude,
		"spd": aircraft.air_velocity,
		"fwd_spd": aircraft.forward_air_speed,
		"vspd": vel.y,
		"g": aircraft.local_g_force,
		"load": aircraft.local_load_factor,
		"stall": 1 if aircraft.is_stalled else 0,
		"px": pos.x,
		"py": pos.y,
		"pz": pos.z,
	}

	if instrument_attitude and is_instance_valid(instrument_attitude):
		d["roll"] = rad_to_deg(instrument_attitude.current_roll)
		d["pitch"] = rad_to_deg(instrument_attitude.current_pitch)
		d["hdg"] = rad_to_deg(instrument_attitude.current_bearing)
		d["lat"] = instrument_attitude.current_latitude
		d["lon"] = instrument_attitude.current_longitude

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

	if aircraft.EnableTemperatureCalculations:
		d["temp"] = aircraft.local_temperature

	return d
