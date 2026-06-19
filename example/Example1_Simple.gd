extends Node3D

const STATE_DIM = 12
const ACTION_DIM = 7
const HIDDEN = 128
const MAX_EPISODE_STEPS = 2000
const FRAMES_PER_STEP = 4
const BATCH_SIZE = 32
const REPLAY_CAPACITY = 20000
const GAMMA = 0.99
const LR_INIT = 0.001
const LR_DECAY = 0.9995
const LR_MIN = 0.0001
const TAU = 0.005
const EPSILON_START = 1.0
const EPSILON_MIN = 0.01
const EPSILON_DECAY = 0.998
const GRAD_CLIP = 1.0
const PRIORITY_ALPHA = 0.6
const PRIORITY_BETA_START = 0.4
const PRIORITY_BETA_STEPS = 100000
const SAVE_PATH = "user://dqn_weights.save"
const SAVE_INTERVAL = 50
const SAVE_VERSION = 3

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

# Prioritized replay
var sumtree: SumTree
var max_priority = 1.0

# Dueling network weights
var w1 = PackedFloat32Array()
var b1 = PackedFloat32Array()
var wA = PackedFloat32Array()
var bA = PackedFloat32Array()
var wV = PackedFloat32Array()
var bV = 0.0
# Target network
var w1t = PackedFloat32Array()
var b1t = PackedFloat32Array()
var wAt = PackedFloat32Array()
var bAt = PackedFloat32Array()
var wVt = PackedFloat32Array()
var bVt = 0.0

var epsilon = EPSILON_START


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

	_init_network()
	sumtree = SumTree.new(REPLAY_CAPACITY)
	var loaded = load_weights()
	if loaded:
		print("Loaded saved weights (episode %d)" % episode_count)
	else:
		print("No saved weights found, starting fresh")
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
	_copy_to_target()


func _copy_to_target():
	w1t = w1.duplicate()
	b1t = b1.duplicate()
	wAt = wA.duplicate()
	bAt = bA.duplicate()
	wVt = wV.duplicate()
	bVt = bV


func save_weights():
	var file = FileAccess.open(SAVE_PATH, FileAccess.WRITE)
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
	file.close()
	print("Weights saved (episode %d)" % episode_count)


func load_weights() -> bool:
	if not FileAccess.file_exists(SAVE_PATH):
		return false
	var file = FileAccess.open(SAVE_PATH, FileAccess.READ)
	if not file:
		return false
	var ver = file.get_32()
	if ver < 3:
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
	file.close()
	_copy_to_target()
	print("Weights loaded from save (episode %d, best_reward %.1f)" % [episode_count, best_reward])
	return true


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
	episode_count += 1
	if episode_count % SAVE_INTERVAL == 0:
		save_weights()
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
const ALT_BAND = 50.0

func get_fuel_soc() -> float:
	if is_instance_valid(energy_container):
		return energy_container.current_level / energy_container.MaxCapacity
	return 1.0


func compute_reward() -> float:
	if not is_instance_valid(aircraft):
		return -1.0
	var alt = max(aircraft.local_altitude, 0.0)
	var spd = aircraft.forward_air_speed
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
		rw += spd * 0.005
		if fuel > 0.1 and engine_on:
			var alt_dev = abs(alt - TARGET_ALT)
			if alt_dev < ALT_BAND:
				rw += 1.0
			else:
				rw += max(0.0, 1.0 - alt_dev / TARGET_ALT) * 0.5
		else:
			rw += 0.5 if gear_down else -0.5
			rw += 0.3 if aircraft.linear_velocity.y < -1.0 else 0.0
			if alt < 50.0 and spd < 20.0:
				rw += 2.0
			if not engine_on and gear_down:
				rw += 0.5

	if alt > 50.0 and spd > 30.0 and not aircraft.is_stalled and engine_on:
		rw += 0.2
	if aircraft.is_stalled:
		rw -= 0.5
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
	var p = pow(max_priority, PRIORITY_ALPHA)
	sumtree.add([state, action, reward, next_state, done], p)


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

	var batch_result = sample_batch(BATCH_SIZE)
	var batch = batch_result[0]
	var indices = batch_result[1]
	var is_weights = batch_result[2]

	var dw1 = PackedFloat32Array()
	var db1 = PackedFloat32Array()
	var dwA = PackedFloat32Array()
	var dbA = PackedFloat32Array()
	var dwV = PackedFloat32Array()
	var dbV = 0.0
	dw1.resize(HIDDEN * STATE_DIM)
	db1.resize(HIDDEN)
	dwA.resize(ACTION_DIM * HIDDEN)
	dbA.resize(ACTION_DIM)
	dwV.resize(HIDDEN)

	for i in range(BATCH_SIZE):
		var b = batch[i]
		var s: Array = b[0]
		var a: int = b[1]
		var r: float = b[2]
		var ns: Array = b[3]
		var d: bool = b[4]

		var x_s = PackedFloat32Array(s)
		var x_ns = PackedFloat32Array(ns)

		var fwd = forward(x_s, w1, b1, wA, bA, wV, bV)
		var h = fwd[0]
		var V_s = fwd[1]
		var A_s = fwd[2]
		var Q_s = fwd[3]

		# Double DQN: online selects action, target evaluates
		var fwd_on = forward(x_ns, w1, b1, wA, bA, wV, bV)
		var Q_on = fwd_on[3]
		var best_a = 0
		for j in range(1, ACTION_DIM):
			if Q_on[j] > Q_on[best_a]:
				best_a = j

		var fwd_tg = forward(x_ns, w1t, b1t, wAt, bAt, wVt, bVt)
		var Q_tg = fwd_tg[3]
		var target = r + GAMMA * Q_tg[best_a] * (0.0 if d else 1.0)

		# Update priority with TD error
		var td_err = abs(Q_s[a] - target) + 1e-6
		var p = pow(td_err, PRIORITY_ALPHA)
		sumtree.set_priority(indices[i], p)
		if td_err > max_priority:
			max_priority = td_err

		var dQ = 2.0 * (Q_s[a] - target) / float(BATCH_SIZE)
		dQ *= is_weights[i]

		var dV = dQ
		var dA = PackedFloat32Array()
		dA.resize(ACTION_DIM)
		var invN = 1.0 / ACTION_DIM
		for ai in range(ACTION_DIM):
			dA[ai] = -dQ * invN
		dA[a] += dQ

		for hi in range(HIDDEN):
			dwV[hi] += dV * h[hi]
		dbV += dV

		for ai in range(ACTION_DIM):
			var dai = dA[ai]
			for hi in range(HIDDEN):
				dwA[ai * HIDDEN + hi] += dai * h[hi]
			dbA[ai] += dai

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
				dw1[hi * STATE_DIM + j] += ghi * s[j]
			db1[hi] += ghi

	if GRAD_CLIP > 0:
		for i in range(dw1.size()):
			dw1[i] = clamp(dw1[i], -GRAD_CLIP, GRAD_CLIP)
		for i in range(db1.size()):
			db1[i] = clamp(db1[i], -GRAD_CLIP, GRAD_CLIP)
		for i in range(dwA.size()):
			dwA[i] = clamp(dwA[i], -GRAD_CLIP, GRAD_CLIP)
		for i in range(dbA.size()):
			dbA[i] = clamp(dbA[i], -GRAD_CLIP, GRAD_CLIP)
		for i in range(dwV.size()):
			dwV[i] = clamp(dwV[i], -GRAD_CLIP, GRAD_CLIP)
		dbV = clamp(dbV, -GRAD_CLIP, GRAD_CLIP)

	step_count += 1
	var current_lr = max(LR_MIN, LR_INIT * pow(LR_DECAY, step_count))

	for i in range(w1.size()):
		w1[i] -= current_lr * dw1[i]
	for i in range(b1.size()):
		b1[i] -= current_lr * db1[i]
	for i in range(wA.size()):
		wA[i] -= current_lr * dwA[i]
	for i in range(bA.size()):
		bA[i] -= current_lr * dbA[i]
	for i in range(wV.size()):
		wV[i] -= current_lr * dwV[i]
	bV -= current_lr * dbV

	var tau = TAU
	for i in range(w1.size()):
		w1t[i] = tau * w1[i] + (1.0 - tau) * w1t[i]
	for i in range(b1.size()):
		b1t[i] = tau * b1[i] + (1.0 - tau) * b1t[i]
	for i in range(wA.size()):
		wAt[i] = tau * wA[i] + (1.0 - tau) * wAt[i]
	for i in range(bA.size()):
		bAt[i] = tau * bA[i] + (1.0 - tau) * bAt[i]
	for i in range(wV.size()):
		wVt[i] = tau * wV[i] + (1.0 - tau) * wVt[i]
	bVt = tau * bV + (1.0 - tau) * bVt

	epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)


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
		if episode_reward > best_reward:
			best_reward = episode_reward
		print("Episode %d | reward: %.1f | steps: %d | eps: %.3f | best: %.1f" % [episode_count, episode_reward, episode_step, epsilon, best_reward])
		reset_episode()
		return

	if takeoff_phase:
		do_takeoff_step()
		return

	frame_counter += 1

	if frame_counter < FRAMES_PER_STEP:
		return

	frame_counter = 0

	var state = get_state()
	var action = select_action(state)
	apply_action(action)

	var next_state = get_state()
	var reward = compute_reward()
	episode_step += 1
	episode_reward += reward
	var done = is_done or episode_step >= MAX_EPISODE_STEPS

	push_replay(state, action, reward, next_state, done)
	train_step()

	if done:
		if episode_reward > best_reward:
			best_reward = episode_reward
		print("Episode %d | reward: %.1f | steps: %d | eps: %.3f | best: %.1f" % [episode_count, episode_reward, episode_step, epsilon, best_reward])
		reset_episode()


func _on_BtnBack_pressed():
	get_tree().change_scene_to_file("res://example/ExampleList.tscn")
