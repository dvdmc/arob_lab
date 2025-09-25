import rclpy
import os
from rclpy.node import Node
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Header, ColorRGBA
from builtin_interfaces.msg import Duration
import tf_transformations
import copy
from time import sleep
from geometry_msgs.msg import PoseStamped
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from std_srvs.srv import SetBool
from as2_python_api.drone_interface import DroneInterface

class DroneRaceNode(Node):
    def __init__(self):
        super().__init__('drone_race_node')

        # Drone interface
        self.uav = DroneInterface(
            drone_id='drone0',
            use_sim_time=True,
            verbose=True)
        self.pose_command_publisher = self.create_publisher(PoseStamped, '/drone0/motion_reference/pose', 10)
        # Create subscriber
        # QoS compatible con publicadores "best effort"
        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        self.pose_suscriber = self.create_subscription(PoseStamped, '/drone0/self_localization/pose',
            self.drone_pose,   # callback
            qos_profile        # QoS queue size
        )
        self.current_gate = PoseStamped()
        self.current_gate_id = 0
        self.current_pose = PoseStamped()

        # Publishers for markers
        self.pub_gate_markers = self.create_publisher(MarkerArray, 'gate_markers', 10)
        
        self.id_marker = 0
        self.gates = []
    
    def start_drone(self):
        ##### ARM OFFBOARD #####
        print("Offboard")
        self.uav.offboard()
        print("Arm")
        self.uav.arm()

        ##### TAKE OFF #####
        print("Take Off")
        self.uav.takeoff(height=1.0, speed=1.0)
        sleep(1.0)
        print('Drone ready to fly')
    
    def read_gates(self, filepath):
        self.gates.clear()
        try:
            with open(filepath, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        x, y, z, yaw = map(float, parts[0:4])
                        pose = Pose()
                        pose.position.x = x
                        pose.position.y = y
                        pose.position.z = z
                        # Orientation from yaw only, assuming no roll/pitch
                        quat = tf_transformations.quaternion_from_euler(0, 0, yaw)
                        pose.orientation.x = quat[0]
                        pose.orientation.y = quat[1]
                        pose.orientation.z = quat[2]
                        pose.orientation.w = quat[3]
                        self.gates.append(pose)
            self.get_logger().info(f'Read {len(self.gates)} gates from {filepath}')

            # Initialize the first goal
            gate = self.gates[0]
            self.current_gate.header.stamp = self.get_clock().now().to_msg()
            self.current_gate.header.frame_id = 'earth'

            # Pose
            self.current_gate.pose.position.x = gate.position.x
            self.current_gate.pose.position.y = gate.position.y
            self.current_gate.pose.position.z = gate.position.z

            # Orientation
            self.current_gate.pose.orientation.x = gate.orientation.x
            self.current_gate.pose.orientation.y = gate.orientation.y
            self.current_gate.pose.orientation.z = gate.orientation.z
            self.current_gate.pose.orientation.w = gate.orientation.w
            self.current_gate_id = 0
            return True
        except Exception as e:
            self.get_logger().error(f'Error reading gates file: {e}')
            return False
    
    def drone_pose(self, msg: PoseStamped):
        self.current_pose = copy.deepcopy(msg)
        # print('Pose received')
    
    def RPY_to_R_matrix(roll, pitch, yaw):
        """
        Converts the roll, pitch, and yaw angles to a 3×3 rotation matrix (NumPy).
        Similar to Eigen::AngleAxis and Eigen::Quaternion used in C++.
        """
        q = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
        R = tf_transformations.quaternion_matrix(q)[:3, :3]
        return R

    def quat_to_R_matrix(self, q):
        """
        Converts a geometry_msgs.msg.Quaternion to a 3×3 rotation matrix (NumPy).
        Uses tf_transformations for the conversion.
        """
        quat = [q.x, q.y, q.z, q.w]
        R = tf_transformations.quaternion_matrix(quat)[:3, :3]
        return R

    def RPY_to_quat(self, roll, pitch, yaw):
        """
        Converts roll, pitch, and yaw to a geometry_msgs.msg.Quaternion
        """
        q = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
        quat_msg = Quaternion()
        quat_msg.x = q[0]
        quat_msg.y = q[1]
        quat_msg.z = q[2]
        quat_msg.w = q[3]
        return quat_msg
    
    def draw_gate_markers(self):
        marker_array = MarkerArray()
        for idx, gate in enumerate(self.gates):

            rotate_gate = self.quat_to_R_matrix(gate.orientation)
            pos_gate = np.array([gate.position.x, gate.position.y, gate.position.z])

            marker = Marker()
            marker.header.frame_id = 'earth'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'gates'
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            marker.lifetime = Duration(sec=0)

            # Line Marker for the green lines
            line_marker = Marker()
            line_marker.header.frame_id = "earth"
            line_marker.header.stamp = self.get_clock().now().to_msg()
            line_marker.ns = "line"
            line_marker.id = self.id_marker
            self.id_marker += 1
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.05
            line_marker.pose.orientation.w = 1.0
            line_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            line_marker.lifetime = Duration(sec=0)

            gate_size = 0.75
            points = []

            # Gate Corners
            corners = [
                np.array([0.0, gate_size, gate_size]),
                np.array([0.0, -gate_size, gate_size]),
                np.array([0.0, -gate_size, -gate_size]),
                np.array([0.0, gate_size, -gate_size])
            ]

            for corner in corners:
                pos = pos_gate + rotate_gate @ corner
                marker.pose.position.x = float(pos[0])
                marker.pose.position.y = float(pos[1])
                marker.pose.position.z = float(pos[2])
                marker.id = self.id_marker
                self.id_marker += 1
                marker_array.markers.append(copy.deepcopy(marker))  # We add each cube

                # Last point in the line
                point = Point()
                point.x = float(pos[0])
                point.y = float(pos[1])
                point.z = float(pos[2])
                points.append(point)

            # Close the square by connecting the last point to the first.
            points.append(points[0])
            line_marker.points = points

            marker_array.markers.append(copy.deepcopy(line_marker))

        self.pub_gate_markers.publish(marker_array)
        # self.get_logger().info('Published gate markers')
    
    # Placeholder for trajectory generation (LAB 3)
    '''
    def generate_trajectory(self):
        # Your mav_trajectory_generation code (commented out here)
        pass
    '''
    
    # Draw trajectory markers example placeholder (LAB 3)
    '''
    def draw_trajectory_markers(self):
        # Create MarkerArray of trajectory points or vectors
        pass
    '''
    
    def timer_callback(self):
        self.get_logger().info('timer callback')
        if self.current_pose is None:
            return
        
        #Complete the code to go through all the gates
        
    
def main(args=None):
    rclpy.init(args=args)
    node = DroneRaceNode()

    print('Node launched')

    node.start_drone()


    path = '/root/arob_ws/src/arob_lab_drones/data/gates.txt'
    filepath = os.path.expanduser(path)
    filepath = os.path.abspath(filepath)
    success = node.read_gates(filepath)
    node.draw_gate_markers()
    
    if success:
        # Periodic timer to publish markers every 500ms
        node.create_timer(0.5, node.timer_callback)
        rclpy.spin(node)
    else:
        node.get_logger().error('Failed to start node due to gates file error')
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
