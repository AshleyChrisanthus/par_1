#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Twist
import time
from macadamia_scanner import MacadamiaScanner, NoFrontTreeError

class MacadamiaStateMachine:
    """Simple state machine for macadamia field navigation"""
    
    def __init__(self):
        self.states = {}
        self.start_state = None
        self.end_states = []
    
    def add_state(self, name, handler, end_state=False):
        self.states[name] = handler
        if end_state:
            self.end_states.append(name)
    
    def set_start(self, name):
        self.start_state = name
    
    def run(self, cargo):
        current_state = self.start_state
        
        while current_state not in self.end_states:
            try:
                # Call the handler for current state
                new_state, cargo, transition = self.states[current_state](cargo)
                current_state = new_state
            except Exception as e:
                print(f"State machine error: {e}")
                break
        
        # Execute end state
        if current_state in self.end_states:
            self.states[current_state](cargo)

class MacadamiaFieldRobot(Node):
    """
    Navigation system for 2x2 macadamia field using brown cylinders
    Adapted from agricultural tree row following algorithm
    """
    
    def __init__(self):
        super().__init__('macadamia_field_robot')
        
        self.get_logger().info("🌰 Macadamia Field Robot Initializing...")
        self.get_logger().info("🎯 Mission: Navigate 2x2 brown cylinder field")
        self.get_logger().info("To stop robot: CTRL + C")
        
        # Create scanner for tree detection and navigation
        self.scanner = MacadamiaScanner()
        
        # Create publisher for robot movement (ROSbot uses cmd_vel)
        self.cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Movement commands
        self.move_cmd_straight = Twist()
        self.move_cmd_straight.linear.x = 0.2  # Slower for precision
        self.move_cmd_straight.angular.z = 0.0
        
        self.move_cmd_right = Twist()
        self.move_cmd_right.linear.x = 0.0
        self.move_cmd_right.angular.z = -0.3
        
        self.move_cmd_left = Twist()
        self.move_cmd_left.linear.x = 0.0
        self.move_cmd_left.angular.z = 0.3
        
        self.stop_cmd = Twist()  # All zeros
        
        # Navigation parameters for 2x2 field
        self.keep_from_wall_max = 0.6  # Maximum distance from tree row
        self.keep_from_wall_min = 0.4  # Minimum distance from tree row
        
        # Mission parameters for 2x2 grid
        self.tree_counter = 0
        self.target_trees_per_row = 2  # 2 trees per row in 2x2 grid
        self.total_target_trees = 4   # Total trees in 2x2 grid
        self.current_row = 1          # Start with row 1
        self.found_tree = False
        self.found_and_stopped = False
        
        # Timing parameters
        self.rate = self.create_rate(10)  # 10 Hz control loop
        
        self.get_logger().info("🚀 Macadamia Field Robot Ready!")
        
    def move_forward_handler(self, system_state):
        """Handle forward movement with tree row following"""
        
        if self.is_parallel():  # Check if robot is parallel to tree row
            self.cmd_vel.publish(self.move_cmd_straight)
            time.sleep(0.2)  # Move forward briefly
            
            new_state, transition = self.adapt_distance()
            return new_state, system_state, transition
        else:  # Not parallel - need angle correction
            new_state, transition = self.adapt_angle()
            return new_state, system_state, transition
    
    def correct_clockwise_handler(self, system_state):
        """Rotate clockwise to correct angle"""
        self.cmd_vel.publish(self.move_cmd_right)
        time.sleep(0.1)
        self.cmd_vel.publish(self.stop_cmd)
        return "move_forward", system_state, "corrected clockwise"
    
    def correct_counterclockwise_handler(self, system_state):
        """Rotate counterclockwise to correct angle"""
        self.cmd_vel.publish(self.move_cmd_left)
        time.sleep(0.1)
        self.cmd_vel.publish(self.stop_cmd)
        return "move_forward", system_state, "corrected counterclockwise"
    
    def tree_from_side_handler(self, system_state):
        """
        Handle tree detection from side - perform sampling action
        Adapted for macadamia brown cylinder detection
        """
        if not self.found_and_stopped:
            self.get_logger().info(f"🌰 MACADAMIA TREE #{self.tree_counter + 1} DETECTED!")
            
            # Sampling action: move toward tree and back
            self.get_logger().info("📍 Performing tree sampling approach...")
            
            # Move left toward tree
            self.cmd_vel.publish(self.move_cmd_left)
            time.sleep(2.0)  # Approach tree
            
            # Move right away from tree  
            self.cmd_vel.publish(self.move_cmd_right)
            time.sleep(2.0)  # Move away
            
            # Stop
            self.cmd_vel.publish(self.stop_cmd)
            
            self.found_and_stopped = True
            self.tree_counter += 1
            
            self.get_logger().info(f"✅ Tree #{self.tree_counter} sampled! Total: {self.tree_counter}/{self.total_target_trees}")
            
            # Check if finished current row
            if self.tree_counter % self.target_trees_per_row == 0:
                if self.tree_counter >= self.total_target_trees:
                    # Mission complete!
                    self.get_logger().info("🎉 MISSION COMPLETE! All 4 trees in 2x2 field sampled!")
                    return "end_state", "Mission accomplished!"
                else:
                    # Move to next row
                    self.current_row += 1
                    self.get_logger().info(f"📍 Moving to row {self.current_row}")
                    return "navigate_to_next_row", system_state, "row completed"
        
        return "move_forward", system_state, "continuing after tree"
    
    def navigate_to_next_row_handler(self, system_state):
        """
        Navigate from end of current row to start of next row
        For 2x2 grid: turn around and position for second row
        """
        self.get_logger().info("🔄 Navigating to next row in 2x2 grid...")
        
        # Turn around to face opposite direction for next row
        self.cmd_vel.publish(self.move_cmd_right)
        time.sleep(4.0)  # 180 degree turn approximately
        
        # Move forward to position for next row
        self.cmd_vel.publish(self.move_cmd_straight)
        time.sleep(3.0)  # Move to next row position
        
        # Final positioning turn
        self.cmd_vel.publish(self.move_cmd_right)
        time.sleep(2.0)
        
        self.cmd_vel.publish(self.stop_cmd)
        
        self.get_logger().info(f"✅ Positioned for row {self.current_row}")
        return "move_forward", system_state, "ready for next row"
    
    def correct_right_handler(self, system_state):
        """Move away from tree row (too close)"""
        
        if self.scanner.tree_from_side():
            return "tree_from_side", system_state, "tree detected during right correction"
        else:
            self.found_and_stopped = False
        
        # Move right (away from trees)
        self.cmd_vel.publish(self.move_cmd_right)
        time.sleep(0.15)
        self.cmd_vel.publish(self.stop_cmd)
        
        # Move forward slightly
        self.cmd_vel.publish(self.move_cmd_straight)
        time.sleep(0.1)
        self.cmd_vel.publish(self.stop_cmd)
        
        # Counter-correct slightly
        self.cmd_vel.publish(self.move_cmd_left)
        time.sleep(0.05)
        self.cmd_vel.publish(self.stop_cmd)
        
        return "move_forward", system_state, "corrected distance right"
    
    def correct_left_handler(self, system_state):
        """Move toward tree row (too far)"""
        
        if self.scanner.tree_from_side():
            return "tree_from_side", system_state, "tree detected during left correction"
        else:
            self.found_and_stopped = False
        
        # Move left (toward trees)
        self.cmd_vel.publish(self.move_cmd_left)
        time.sleep(0.15)
        self.cmd_vel.publish(self.stop_cmd)
        
        # Move forward slightly
        self.cmd_vel.publish(self.move_cmd_straight)
        time.sleep(0.1)
        self.cmd_vel.publish(self.stop_cmd)
        
        # Counter-correct slightly
        self.cmd_vel.publish(self.move_cmd_right)
        time.sleep(0.05)
        self.cmd_vel.publish(self.stop_cmd)
        
        return "move_forward", system_state, "corrected distance left"
    
    def terminate_handler(self, system_state):
        """Clean shutdown"""
        self.get_logger().info("🛑 Stopping Macadamia Field Robot")
        self.cmd_vel.publish(self.stop_cmd)
        time.sleep(0.2)
        self.get_logger().info("✅ Mission completed successfully!")
        return "end_state", "Robot stopped"
    
    def adapt_distance(self):
        """Determine if robot needs distance correction from tree row"""
        
        # Check for tree from side first
        if self.scanner.tree_from_side():
            return "tree_from_side", "tree detected from side!"
        else:
            self.found_and_stopped = False
        
        try:
            left_data = self.scanner.get_generated_data()
        except NoFrontTreeError:
            self.get_logger().warn("No front tree found - turning counterclockwise")
            return 'counterclockwise', 'no front tree error'
        
        # Calculate average distance from tree row
        avg_actual_dist = 0
        valid_readings = 0
        
        for range_val in left_data[15:25]:  # Middle section of scan
            if 0 < range_val < 5.0:  # Valid reading
                avg_actual_dist += range_val
                valid_readings += 1
        
        if valid_readings == 0:
            return "move_forward", "no valid distance readings"
        
        avg_actual_dist = avg_actual_dist / valid_readings
        
        self.get_logger().debug(f"Average distance from tree row: {avg_actual_dist:.2f}m")
        
        # Distance-based decisions
        if avg_actual_dist < self.keep_from_wall_min:
            return "correctright", "too close to trees"
        elif avg_actual_dist > self.keep_from_wall_max:
            return "correctleft", "too far from trees"
        else:
            return "move_forward", "good distance from trees"
    
    def adapt_angle(self):
        """Determine angle correction needed for parallel alignment"""
        
        try:
            left_data = self.scanner.get_generated_data()
        except NoFrontTreeError:
            self.get_logger().warn("No tree line found for angle adaptation")
            return 'counterclockwise', 'no tree line for angle'
        
        # Find minimum distance point
        valid_data = [(i, val) for i, val in enumerate(left_data) if 0 < val < 5.0]
        
        if not valid_data:
            return "move_forward", "no valid angle data"
        
        min_value = min(valid_data, key=lambda x: x[1])[1]
        min_indices = [i for i, val in valid_data if abs(val - min_value) < 0.1]
        min_index = int(np.mean(min_indices))  # Average if multiple minimums
        
        self.get_logger().debug(f"Minimum distance at index {min_index}")
        
        # Angle correction decisions
        if min_index < 15:
            return 'clockwise', 'angle correction clockwise'
        elif min_index > 25:
            return 'counterclockwise', 'angle correction counterclockwise'
        else:
            return "move_forward", "good angle alignment"
    
    def is_parallel(self):
        """Check if robot is parallel to tree row"""
        
        try:
            left_data = self.scanner.get_generated_data()
        except NoFrontTreeError:
            return False
        
        # Find minimum distance point
        valid_data = [(i, val) for i, val in enumerate(left_data) if 0 < val < 5.0]
        
        if not valid_data:
            return False
        
        min_value = min(valid_data, key=lambda x: x[1])[1]
        min_indices = [i for i, val in valid_data if abs(val - min_value) < 0.1]
        min_index = int(np.mean(min_indices))
        
        # Robot is parallel if minimum is in center range
        return 15 <= min_index <= 25
    
    def start_navigation(self):
        """Initialize and start the navigation state machine"""
        
        self.get_logger().info("🎯 Starting Macadamia Field Navigation...")
        self.get_logger().info(f"📍 Target: {self.total_target_trees} trees in 2x2 grid")
        
        # Create state machine
        sm = MacadamiaStateMachine()
        
        # Add states
        sm.add_state("move_forward", self.move_forward_handler)
        sm.add_state("correctright", self.correct_right_handler)
        sm.add_state("correctleft", self.correct_left_handler)
        sm.add_state("clockwise", self.correct_clockwise_handler)
        sm.add_state("counterclockwise", self.correct_counterclockwise_handler)
        sm.add_state("tree_from_side", self.tree_from_side_handler)
        sm.add_state("navigate_to_next_row", self.navigate_to_next_row_handler)
        sm.add_state("end_state", self.terminate_handler, end_state=True)
        
        # Set starting state
        sm.set_start("move_forward")
        
        # Initialize system state
        system_state = {"mission_status": "active", "start_time": time.time()}
        
        # Run the state machine
        try:
            sm.run(system_state)
        except KeyboardInterrupt:
            self.get_logger().info("🛑 Navigation interrupted by user")
            self.cmd_vel.publish(self.stop_cmd)

def main(args=None):
    """Main function to run the macadamia field robot"""
    
    rclpy.init(args=args)
    
    try:
        # Create and run the robot
        robot = MacadamiaFieldRobot()
        
        # Give time for all systems to initialize
        time.sleep(2)
        
        # Start navigation
        robot.start_navigation()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Macadamia Field Robot...")
    except Exception as e:
        print(f"❌ Robot error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
