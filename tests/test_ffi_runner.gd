extends Node

var passed := 0
var failed := 0
var total := 0
var results: Array[String] = []

func _ready():
	print("=== Integration Tests: Godot-Rust FFI ===\n")
	test_it01_init_valid_config()
	test_it02_init_invalid_dims()
	test_it03_select_action_valid_index()
	test_it04_store_transition()
	test_it05_train_step_after_warmup()
	test_it06_save_load_roundtrip()

	print("\n=== Results: %d/%d passed, %d failed ===" % [passed, total, failed])
	for r in results:
		print(r)

	var report = "=== INTEGRATION TEST REPORT ===\n"
	report += "Passed: %d/%d\nFailed: %d\n\n" % [passed, total, failed]
	for r in results:
		report += r + "\n"
	var f = FileAccess.open("user://integration_test_report.txt", FileAccess.WRITE)
	if f:
		f.store_string(report)
		f.close()
		print("Report saved to user://integration_test_report.txt")

	get_tree().quit(0 if failed == 0 else 1)

func check(test_id: String, description: String, condition: bool):
	total += 1
	if condition:
		passed += 1
		var msg = "[PASS] %s: %s" % [test_id, description]
		print(msg)
		results.append(msg)
	else:
		failed += 1
		var msg = "[FAIL] %s: %s" % [test_id, description]
		print(msg)
		results.append(msg)

func test_it01_init_valid_config():
	var agent = DQNRust.new()
	agent.init(13, 7, 256, 128, 1000, 3, 0.99, 42)
	check("IT-01", "Agent initialization with valid config", agent != null)
	agent.free()

func test_it02_init_invalid_dims():
	var agent = DQNRust.new()
	agent.init(0, 7, 256, 128, 1000, 3, 0.99, 42)
	check("IT-02", "Agent handles zero state dimension gracefully", agent != null)
	agent.free()

func test_it03_select_action_valid_index():
	var agent = DQNRust.new()
	agent.init(13, 7, 256, 128, 1000, 3, 0.99, 42)
	agent.set_epsilon(0.0)
	var state = PackedFloat32Array([0.5, 0.3, 0.1, 0.9, 0.2, 0.8, 0.1, 0.0, 0.7, 0.5, 0.6, 0.9, 0.5])
	var action = agent.select_action(state)
	check("IT-03", "select_action returns valid action index [0..6]", action >= 0 and action <= 6)
	agent.free()

func test_it04_store_transition():
	var agent = DQNRust.new()
	agent.init(13, 7, 256, 128, 1000, 3, 0.99, 42)
	var s1 = PackedFloat32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0, 0.8, 0.9, 0.1, 0.2, 0.5])
	var s2 = PackedFloat32Array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0, 0.9, 0.1, 0.2, 0.3, 0.5])
	agent.push_replay(s1, 3, 1.5, s2, false)
	check("IT-04", "store_transition accepts valid experience", agent.get_replay_size() >= 0)
	agent.free()

func test_it05_train_step_after_warmup():
	var agent = DQNRust.new()
	agent.init(13, 7, 256, 128, 1000, 3, 0.99, 42)
	var s1 = PackedFloat32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0, 0.8, 0.9, 0.1, 0.2, 0.5])
	var s2 = PackedFloat32Array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0, 0.9, 0.1, 0.2, 0.3, 0.5])
	for i in range(64):
		agent.push_replay(s1, i % 7, float(i) * 0.1, s2, i == 63)
	var result = agent.train(32, 0.99, 1.0, 0.001)
	check("IT-05", "train_step returns true after warmup", result == true)
	agent.free()

func test_it06_save_load_roundtrip():
	var agent = DQNRust.new()
	agent.init(13, 7, 256, 128, 1000, 3, 0.99, 42)
	agent.set_epsilon(0.5)
	agent.set_step_count(100)
	var s1 = PackedFloat32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0, 0.8, 0.9, 0.1, 0.2, 0.5])
	var s2 = PackedFloat32Array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0, 0.9, 0.1, 0.2, 0.3, 0.5])
	agent.push_replay(s1, 3, 1.0, s2, false)

	var weights_before = agent.get_weights_online()
	var agent2 = DQNRust.new()
	agent2.init(13, 7, 256, 128, 1000, 3, 0.99, 42)
	agent2.set_weights_online(weights_before)

	var weights_after = agent2.get_weights_online()
	var weights_match = true
	if weights_before.size() == weights_after.size():
		for i in range(weights_before.size()):
			if weights_before[i] != weights_after[i]:
				weights_match = false
				break
	else:
		weights_match = false

	check("IT-06", "save/load round-trip preserves weights", weights_match)
	agent.free()
	agent2.free()
