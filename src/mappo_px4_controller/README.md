# mappo_px4_controller

ROS 2/PX4 integration package for the MAPPO interceptor system.

The stage-six scaffold contains typed state storage, coordinate conversion
helpers, parameters, and launch/install metadata. PX4 subscriptions, offboard
publishers, timers, and MAPPO inference are intentionally deferred to the next
migration stage.

The default configuration keeps `enable_px4_commands` set to `false`.
No arm, mode-switch, trajectory-setpoint, or vehicle-command publisher is
implemented in this scaffold.
