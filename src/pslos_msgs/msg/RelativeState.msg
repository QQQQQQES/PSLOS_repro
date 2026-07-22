uint8 SOURCE_INVALID=0
uint8 SOURCE_GAZEBO_TRUTH=1
uint8 SOURCE_DC_EKF=2

std_msgs/Header header
geometry_msgs/Vector3 position
geometry_msgs/Vector3 velocity
geometry_msgs/Vector3 los
geometry_msgs/Quaternion interceptor_attitude
geometry_msgs/Vector3 interceptor_velocity
geometry_msgs/Vector3 interceptor_body_rate
float64 range
float64 vision_age
float64[36] covariance
uint32 measurement_count
uint8 source
bool measurement_updated
bool valid

