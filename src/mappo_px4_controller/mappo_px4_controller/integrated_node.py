"""ROS 2 controller scaffold for connecting MAPPO core to PX4.

PX4 subscriptions, offboard publishers, and MAPPO inference are intentionally
left for the next migration stage. This scaffold publishes no flight commands.
"""

from typing import Optional, Sequence

import rclpy
from rclpy.node import Node

from .state_store import AssignmentState, ControllerState, DefenderState


class MappoPx4Controller(Node):
    """Own controller parameters and mutable state without module globals."""

    def __init__(self) -> None:
        super().__init__('mappo_px4_controller')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('ndef', 4),
                ('k_max', 5),
                ('decision_period_sec', 1.0),
                ('setpoint_rate_hz', 20.0),
                ('origin_xy', [0.0, 0.0]),
                ('use_estimated', True),
                ('deterministic', True),
                ('velocity_min_mps', 3.0),
                ('velocity_max_mps', 5.0),
                ('checkpoint_path', ''),
                ('vehicle_namespaces', [
                    '/px4_1',
                    '/px4_2',
                    '/px4_3',
                    '/px4_4',
                ]),
                ('enable_px4_commands', False),
            ],
        )

        ndef = self.get_parameter('ndef').get_parameter_value().integer_value
        if ndef < 1:
            raise ValueError('parameter ndef must be at least 1')

        self.state = ControllerState(
            defenders={
                defender_id: DefenderState(vehicle_id=defender_id)
                for defender_id in range(ndef)
            },
            assignments={
                defender_id: AssignmentState(
                    defender_id_0based=defender_id,
                )
                for defender_id in range(ndef)
            },
        )

        if self.get_parameter('enable_px4_commands').value:
            self.get_logger().warning(
                'PX4 command output is not implemented in this scaffold; '
                'no flight commands will be published.'
            )


def main(args: Optional[Sequence[str]] = None) -> None:
    """Run the controller node."""

    rclpy.init(args=args)
    node = MappoPx4Controller()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
