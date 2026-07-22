cmake_minimum_required(VERSION 3.0.2)
project(pslos_ros)

find_package(catkin REQUIRED COMPONENTS geometry_msgs mavros_msgs pslos_core pslos_msgs rospy sensor_msgs std_msgs)
catkin_package(CATKIN_DEPENDS geometry_msgs mavros_msgs pslos_core pslos_msgs rospy sensor_msgs std_msgs)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_px4_adapter.py)
  catkin_add_nosetests(test/test_dcekf_node.py)
  catkin_add_nosetests(test/test_reference_node.py)
endif()

catkin_install_python(
  PROGRAMS
  scripts/dcekf_node.py
  scripts/feature_adapter_node.py
  scripts/px4_adapter_node.py
  scripts/reference_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
