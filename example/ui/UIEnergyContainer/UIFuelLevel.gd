extends Control

@export var ActiveColor: Color = Color(0.9, 0.9, 0.5, 1.0)
@export var Caption: String = "SPD" # Changed to match your screenshot

# Target the child node directly. 
# If your child node is named "AltSpeed", use $AltSpeed
@onready var display_label = $AltSpeed 

func _ready():
	# Set initial text
	if display_label:
		display_label.text = Caption

func update_interface(values: Dictionary):
	# Adjust this based on how you want to display the data
	# Assuming 'values' contains speed info
	if display_label and values.has("speed"):
		display_label.text = str(values["speed"])
