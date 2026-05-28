#!/usr/bin/env python3
"""
Simple PD joint command tool for CASBOT02.
Publishes position targets to /motion/joint_cmd so hl_motion forwards them to hardware.

Usage:
  # Lift right hand (preset)
  python3 scripts/test_joint_cmd.py --preset right_hand_up

  # Lift left hand
  python3 scripts/test_joint_cmd.py --preset left_hand_up

  # Set specific joints
  python3 scripts/test_joint_cmd.py --joints right_shoulder_pitch_joint,right_elbow_pitch_joint --targets 0.35,0.87

  # Hold current position (just publish current state as target)
  python3 scripts/test_joint_cmd.py --hold

  # Return to zero
  python3 scripts/test_joint_cmd.py --zero
"""

import argparse
import sys
import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# ---------------------------------------------------------------------------
# Presets — reference positions (radians) for common test poses
# Values from wbc_fsm/State_FixedStand.h and motion_params.yaml
# ---------------------------------------------------------------------------
PRESETS = {
    "right_hand_up": {
        "right_shoulder_pitch_joint": 0.35,
        "right_shoulder_roll_joint": -0.18,
        "right_elbow_pitch_joint": 0.87,
    },
    "left_hand_up": {
        "left_shoulder_pitch_joint": 0.35,
        "left_shoulder_roll_joint": 0.18,
        "left_elbow_pitch_joint": 0.87,
    },
    "both_hands_up": {
        "left_shoulder_pitch_joint": 0.35,
        "left_shoulder_roll_joint": 0.18,
        "left_elbow_pitch_joint": 0.87,
        "right_shoulder_pitch_joint": 0.35,
        "right_shoulder_roll_joint": -0.18,
        "right_elbow_pitch_joint": 0.87,
    },
    "t_pose": {
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.3,
        "left_elbow_pitch_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": -0.3,
        "right_elbow_pitch_joint": 0.0,
    },
}

# All 28 joints in the order hl_motion publishes /motion/joint_state
ALL_JOINTS = [
    "leg_l1_joint", "leg_l2_joint", "leg_l3_joint",
    "leg_l4_joint", "leg_l5_joint", "leg_l6_joint",
    "leg_r1_joint", "leg_r2_joint", "leg_r3_joint",
    "leg_r4_joint", "leg_r5_joint", "leg_r6_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_wrist_yaw_joint",
    "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_wrist_yaw_joint",
    "right_wrist_pitch_joint", "right_wrist_roll_joint",
    "head_yaw_joint", "head_pitch_joint",
]


class JointCmdPublisher(Node):
    def __init__(self, target_map, duration=2.0, hold=False, zero=False):
        super().__init__("test_joint_cmd")
        self.pub = self.create_publisher(JointState, "/motion/joint_cmd", 10)
        self.sub = self.create_subscription(
            JointState, "/motion/joint_state", self._js_cb, 10
        )

        self._current_pos = {}
        self._current_received = False
        self._target_map = target_map
        self._duration = duration
        self._hold = hold
        self._zero = zero
        self._start_time = None
        self._start_pos = {}
        self._done = threading.Event()

        rate = 50  # Hz
        self._timer = self.create_timer(1.0 / rate, self._tick)
        self._dt = 1.0 / rate

    def _js_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self._current_pos[name] = pos
        if not self._current_received:
            self.get_logger().info(f"Received joint_state: {len(msg.name)} joints")
            self._current_received = True

    def _tick(self):
        if not self._current_received:
            self.get_logger().warn("Waiting for /motion/joint_state ...", throttle_duration_sec=2)
            return

        if self._start_time is None:
            self._start_time = self.get_clock().now()
            # Snapshot starting positions for joints we will move
            for name in self._target_map:
                self._start_pos[name] = self._current_pos.get(name, 0.0)
            self.get_logger().info(
                f"Starting interpolation to targets over {self._duration}s"
            )
            for name, target in self._target_map.items():
                start = self._start_pos.get(name, 0.0)
                self.get_logger().info(f"  {name}: {start:.3f} -> {target:.3f}")

        elapsed = (self.get_clock().now() - self._start_time).nanoseconds * 1e-9

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ALL_JOINTS.copy()

        if self._zero:
            # Publish all zeros
            msg.position = [0.0] * len(ALL_JOINTS)
        elif self._hold:
            # Publish current positions as targets (hold in place)
            msg.position = [self._current_pos.get(n, 0.0) for n in ALL_JOINTS]
        elif elapsed >= self._duration:
            # Interpolation done, hold at target
            msg.position = [
                self._target_map.get(n, self._current_pos.get(n, 0.0))
                for n in ALL_JOINTS
            ]
            if not self._done.is_set():
                self._done.set()
                self.get_logger().info("Target reached. Holding position. Ctrl-C to exit.")
        else:
            # Linear interpolation
            t = elapsed / self._duration
            # Smoothstep for less jerk
            t = t * t * (3.0 - 2.0 * t)
            msg.position = []
            for name in ALL_JOINTS:
                if name in self._target_map:
                    s = self._start_pos.get(name, 0.0)
                    e = self._target_map[name]
                    msg.position.append(s + (e - s) * t)
                else:
                    msg.position.append(self._current_pos.get(name, 0.0))

        msg.velocity = [0.0] * len(ALL_JOINTS)
        msg.effort = [0.0] * len(ALL_JOINTS)
        self.pub.publish(msg)


def main():
    parser = argparse.ArgumentParser(description="CASBOT02 joint command test tool")
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        help="Use a predefined pose",
    )
    parser.add_argument(
        "--joints",
        type=str,
        help="Comma-separated joint names (e.g. right_shoulder_pitch_joint,right_elbow_pitch_joint)",
    )
    parser.add_argument(
        "--targets",
        type=str,
        help="Comma-separated target positions in radians (must match --joints count)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Interpolation duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--hold",
        action="store_true",
        help="Hold current joint positions (no movement, just lock)",
    )
    parser.add_argument(
        "--zero",
        action="store_true",
        help="Command all joints to zero (DANGER: only use when robot is supported!)",
    )
    args = parser.parse_args()

    target_map = {}

    if args.preset:
        target_map = dict(PRESETS[args.preset])
    elif args.joints:
        names = [n.strip() for n in args.joints.split(",")]
        if not args.targets:
            print("ERROR: --targets required with --joints")
            sys.exit(1)
        targets = [float(t.strip()) for t in args.targets.split(",")]
        if len(names) != len(targets):
            print(f"ERROR: {len(names)} joint names but {len(targets)} targets")
            sys.exit(1)
        target_map = dict(zip(names, targets))
    elif not args.hold and not args.zero:
        parser.print_help()
        print("\nTIP: use --preset right_hand_up for a quick test")
        sys.exit(0)

    # Validate joint names
    for name in target_map:
        if name not in ALL_JOINTS:
            print(f"WARNING: '{name}' not in known joint list, will still try to command it")

    rclpy.init(args=sys.argv)
    node = JointCmdPublisher(target_map, duration=args.duration,
                             hold=args.hold, zero=args.zero)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
