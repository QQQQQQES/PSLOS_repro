uint8 MODE_INVALID=0
uint8 MODE_TRACK=1
uint8 MODE_CAPTURE=2
uint8 MODE_OUTSIDE_SECTOR=3
uint8 MODE_STALE_STATE=4

std_msgs/Header header
geometry_msgs/Vector3 desired_force_world
geometry_msgs/Quaternion desired_attitude
geometry_msgs/Vector3 desired_body_rate
float64 normalized_thrust
uint8 mode
bool valid
string reason

