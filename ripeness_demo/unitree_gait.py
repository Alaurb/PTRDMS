"""Firmware-configured Unitree SDK gait adapter, independent of ROS."""
class UnitreeGaitAdapter:
    def __init__(self, client, normal_id, terrain_id):
        if not callable(getattr(client, "SwitchGait", None)):
            raise ValueError("This SDK does not expose SwitchGait; use the firmware-matched SDK/driver")
        if not isinstance(normal_id, int) or not isinstance(terrain_id, int) or normal_id == terrain_id:
            raise ValueError("Supply distinct normal and terrain gait IDs for the installed Go2-W firmware")
        self.client = client
        self.ids = {"normal": normal_id, "terrain": terrain_id}
        self.confirmed = None

    def request(self, mode):
        if mode not in self.ids:
            raise ValueError("Unknown gait mode")
        if mode == self.confirmed:
            return True
        # A zero SDK result confirms command acceptance; telemetry remains the
        # authority for the robot's actual mode.
        code = self.client.SwitchGait(self.ids[mode])
        if code != 0:
            self.confirmed = None
            return False
        self.confirmed = mode
        return True

    def observe(self, gait_id):
        self.confirmed = next((name for name, value in self.ids.items() if value == gait_id), None)
        return self.confirmed
