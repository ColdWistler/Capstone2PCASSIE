extends Control

# Target the child node directly
# Drag the child node while holding Ctrl to verify the path
@onready var speed_label = $AltSpeed 

func update_interface(values: Dictionary):
	# Check if the node exists and the value exists in the dictionary
	if speed_label and values.has("engine_active"):
		# Update the color based on active state (assuming the node has a modulate property)
		speed_label.modulate = Color.GREEN if values["engine_active"] else Color.RED
	
	# If you need to update a position or text based on power
	if speed_label and values.has("engine_power"):
		# Example: update text or position based on your needs
		speed_label.text = str(int(values["engine_power"] * 100)) + " SPD"
