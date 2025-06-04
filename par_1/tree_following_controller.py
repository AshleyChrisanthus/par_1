#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class TreeFollowingController(Node):
    """
    Simple controller that makes robot go straight when trees are detected.
    Integrates with the existing TreeDetector node.
    """
    
    def __init__(self):
        super().__init__('tree_following_controller')
        
        # Subscribe to tree detection messages
        self.tree_sub = self.create_subscription(
            String,
            '/detected_trees',  # From your TreeDetector
            self.tree_callback,
            10)
        
        # Publisher for robot movement commands
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',  # Standard robot velocity topic
            10)
        
        # Movement parameters
        self.forward_speed = 0.3  # m/s - adjust as needed
        self.trees_detected = False
        
        # Timer to continuously publish movement commands
        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz
        
        self.get_logger().info('Tree Following Controller initialized!')
        self.get_logger().info('Robot will go straight when trees are detected')
        
    def tree_callback(self, msg):
        """Handle tree detection messages from TreeDetector."""
        if msg.data.startswith("trees_detected:"):
            # Extract number of trees from message
            num_trees = int(msg.data.split(":")[1])
            self.trees_detected = True
            self.get_logger().info(f'Trees in vision! Going straight... ({num_trees} trees detected)')
        else:
            self.trees_detected = False
    
    def control_loop(self):
        """Main control loop - runs at 10 Hz."""
        cmd = Twist()
        
        if self.trees_detected:
            # Trees detected - go straight
            cmd.linear.x = self.forward_speed
            cmd.angular.z = 0.0
        else:
            # No trees - stop
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        
        # Publish movement command
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        controller = TreeFollowingController()
        controller.get_logger().info('Starting tree following controller...')
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Tree following controller shutting down...')
    finally:
        if 'controller' in locals():
            controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()