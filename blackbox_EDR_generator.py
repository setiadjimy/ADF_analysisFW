#!/usr/bin/env python3

import carla
import can
import cantools
import pandas as pd
import pygame
import time
import logging
import math
import threading
from datetime import datetime
from collections import deque
from pathlib import Path
import queue
import os

# ======================================================
# CONFIGURATION
# ======================================================
CONFIG = {
    "CAN_INTERFACE": "virtual",
    "CAN_CHANNEL": "vcan0",
    "DBC_FILE": Path("bmw.dbc"),
    "CAN_LOG_FILE": "vehicle_can_data.log",
    "EDR_BUFFER_SECONDS": 20,
    "EDR_TICK_INTERVAL": 0.05,  # 20Hz
    "EDR_CSV_FILE": "edr_data",
    "CARLA_HOST": "localhost",
    "CARLA_PORT": 2000,
    "VEHICLE_BLUEPRINT": "vehicle.*model3*",
    "LOG_LEVEL": "INFO",
    "AUTOPILOT": False,
    "COLLISION_COOLDOWN": 5.0  # Seconds between EDR exports
}

# ======================================================
# PHYSICS & UTILITY
# ======================================================
class VehiclePhysics:
    @staticmethod
    def calculate_speed_kmh(velocity):
        return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6

    @staticmethod
    def calculate_magnitude(vector):
        return math.sqrt(vector.x**2 + vector.y**2 + vector.z**2)

# ======================================================
# EDR BUFFER (Thread-safe)
# ======================================================
class EDRBuffer:
    def __init__(self, max_seconds, tick_interval):
        self.maxlen = int(max_seconds / tick_interval)
        self.buffer = deque(maxlen=self.maxlen)
        self.lock = threading.Lock()
        
    def add(self, record):
        with self.lock:
            self.buffer.append(record)
    
    def export_csv(self, filename):
        with self.lock:
            if not self.buffer:
                logging.warning("EDR buffer empty; skipping export.")
                return
            df = pd.DataFrame(list(self.buffer))
            df.to_csv(filename, index=False)
            logging.info(f"[EDR] Exported {len(df)} records to {filename}")

# ======================================================
# CAN BUS MANAGER
# ======================================================
class CANManager:
    def __init__(self, dbc_file, interface='virtual', channel='vcan0', gps_provider=None):
        self.gps_provider = gps_provider
		
        try:
            self.db = cantools.database.load_file(dbc_file)
        except Exception as e:
            logging.error(f"Failed to load DBC: {e}")
            raise
            
        self.bus = can.Bus(interface=interface, channel=channel)
        self.message_queue = queue.Queue(maxsize=500)
        self.running = True
        self.send_thread = threading.Thread(target=self._send_worker, daemon=True)
        self.send_thread.start()
        
        self.message_cache = {}
        self._cache_messages()

        # ----- CAN log file setup  ---------------------------------
        self.can_logger = logging.getLogger("can_log")
        self.can_logger.setLevel(logging.INFO)

        # Only add handler once even if CANManager is re‑created
        if not self.can_logger.handlers:
            file_handler = logging.FileHandler(CONFIG["CAN_LOG_FILE"], mode="a")
            file_formatter = logging.Formatter(
                "%(asctime)s %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            self.can_logger.addHandler(file_handler)
        # ------------------------------------------------------------

    def _cache_messages(self):
        for name in ["Speed", "AccPedal", "EngineAndBrake", "StatusDSC_KCAN", "SteeringWheelAngle", "TurnSignals", "Crash", "GPS"]:
            try:
                self.message_cache[name] = self.db.get_message_by_name(name)
            except KeyError:
                pass

    def encode_and_queue(self, msg_name, signals):
        if msg_name in self.message_cache:
            try:
                msg_def = self.message_cache[msg_name]
                data = msg_def.encode(signals)
                msg = can.Message(arbitration_id=msg_def.frame_id, data=data, is_extended_id=False)
                self.message_queue.put(msg, block=False)
            except (queue.Full, Exception) as e:
                logging.debug(f"CAN Queue Error {msg_name}: {e}")

    def _send_worker(self):
        while self.running:
            try:
                msg = self.message_queue.get(timeout=1.0)
                data_str = " ".join(f"{b:02x}" for b in msg.data)
                
                lat = lon = None
                if self.gps_provider is not None:
                    gd = getattr(self.gps_provider, "gps_data", None)
                    if gd is not None:
                        lat = gd.get("lat", None)
                        lon = gd.get("lon", None)
						
                if lat is not None and lon is not None:
                    self.can_logger.info(
                        f"<0x{msg.arbitration_id:03x}> [{msg.dlc}] {data_str} "
                        f"GPS(lat={lat:.7f}, lon={lon:.7f})")
                else:
                    self.can_logger.info(
                    f"<0x{msg.arbitration_id:03x}> [{msg.dlc}] {data_str}")
                # -----------------------------------------------------

                self.bus.send(msg)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"CAN Bus Error: {e}")

    def shutdown(self):
        self.running = False
        self.bus.shutdown()

# ======================================================
# INTEGRATED MONITOR
# ======================================================
class IntegratedVehicleMonitor:
    def __init__(self, config):
        self.config = config

        logging.basicConfig(
            level=config["LOG_LEVEL"],
            format='%(levelname)s: %(message)s'
        )
        
        self.can_manager = CANManager(config["DBC_FILE"], config["CAN_INTERFACE"], config["CAN_CHANNEL"], gps_provider=self)
        self.edr_buffer = EDRBuffer(config["EDR_BUFFER_SECONDS"], config["EDR_TICK_INTERVAL"])
        
        self.client = None
        self.world = None
        self.vehicle = None
        self.sensors = []
        
        # Internal State
        self.last_export_time = 0
        self.gps_data = {'lat': 0.0, 'lon': 0.0, 'timestamp': 0.0}
        
    def connect(self):
        self.client = carla.Client(self.config["CARLA_HOST"], self.config["CARLA_PORT"])
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Use synchronous mode
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = self.config["EDR_TICK_INTERVAL"]
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        
        bp = self.world.get_blueprint_library().filter(self.config["VEHICLE_BLUEPRINT"])[0]
        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(bp, spawn_point)
        self.vehicle.set_autopilot(self.config["AUTOPILOT"])
        
        # Setup Sensors
        gnss_bp = self.world.get_blueprint_library().find('sensor.other.gnss')
        col_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        
        gnss = self.world.spawn_actor(gnss_bp, carla.Transform(), attach_to=self.vehicle)
        col = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        
        gnss.listen(lambda data: self._on_gnss(data))
        col.listen(lambda event: self._on_collision(event))
        
        self.sensors.extend([gnss, col])
        return True

    def _on_gnss(self, data):
        self.gps_data = {'lat': data.latitude, 'lon': data.longitude, 'timestamp': data.timestamp}

    def _on_collision(self, event):
        now = time.time()
        if now - self.last_export_time > self.config["COLLISION_COOLDOWN"]:
            logging.warning("Collision Detected! Triggering EDR Export.")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.edr_buffer.export_csv(f"{self.config['EDR_CSV_FILE']}_collision_{ts}.csv")
            self.last_export_time = now
            
            # Send High Priority CAN Crash Message
            impulse = VehiclePhysics.calculate_magnitude(event.normal_impulse)
            self.can_manager.encode_and_queue("Crash", {
                'CollisionSeverity': min(impulse, 255),
                'CrashDetected': 1,
                'AirbagTriggered': 1 if impulse > 5000 else 0,
                'EmergencyCall': 1
            })

    def _process_can_frames(self):
        v = self.vehicle.get_velocity()
        c = self.vehicle.get_control()
        a = self.vehicle.get_acceleration()
        l = self.vehicle.get_light_state()
        
        # Speed Data
        self.can_manager.encode_and_queue("Speed", {
            'VehicleSpeed': VehiclePhysics.calculate_speed_kmh(v),
            'MovingForward': 1 if v.x > 0.1 else 0,
            'MovingReverse': 0, 'AccY': a.y, 'AccX': a.x, 'YawRate': self.vehicle.get_angular_velocity().z,
            'Counter_416': 0, 'Checksum_416': 0
        })
        
        # Throttle Data
        self.can_manager.encode_and_queue("AccPedal", {
            'EngineSpeed': c.throttle * 6500,
            'AcceleratorPedalPercentage': c.throttle * 100,
            'AcceleratorPedalPressed': 1 if c.throttle > 0.01 else 0,
            'Checksum_170': 0, 'Counter_170': 0, 'CruisePedalInactive': 1, 'CruisePedalActive': 0, 'KickDownPressed': 0, 'ThrottlelPressed': 0
        })
        
        # Brake Data via EngineAndBrake (0x168)
        self.can_manager.encode_and_queue("EngineAndBrake", {
            'BrakePressed': 1 if c.brake > 0.01 else 0,
            'Brake_active2': 1 if c.brake > 0.01 else 0,
            'EngineTorque': 0.0,            # or map from your powertrain if desired
            'EngineTorqueWoInterv': 0.0
        })
        
        # Brake pressure via StatusDSC_KCAN (0x414)
        brake_pressure_raw = int(c.brake * 255)
        self.can_manager.encode_and_queue("StatusDSC_KCAN", {
            'BrakePressure': brake_pressure_raw,
            'BrakeStates': 1 if c.brake > 0.01 else 0,
            'Checksum_414': 0,
            'Counter_414': 0,
            'DTC_on': 0,
            'DSC_full_off': 0
        })
        
        #GPS data
        lat = self.gps_data['lat']
        lon = self.gps_data['lon']
        self.can_manager.encode_and_queue("GPS", {
			'Latitude': int(lat * 1e7),
			'Longitude': int(lon * 1e7)
		})

    def run(self):
        if not self.connect(): return
        
        pygame.init()
        screen = pygame.display.set_mode((200, 100))
        pygame.display.set_caption("EDR Controller")
        clock = pygame.time.Clock()
        
        logging.info("System Online. Press 'E' for manual export, 'ESC' to quit.")
        
        try:
            while True:
                self.world.tick()
                
                # 1. Capture Inputs
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE: return
                        if event.key == pygame.K_e:
                            self.edr_buffer.export_csv(f"manual_export_{int(time.time())}.csv")

                # 2. Manual Driving (if not autopilot)
                if not self.config["AUTOPILOT"]:
                    keys = pygame.key.get_pressed()
                    ctrl = carla.VehicleControl()
                    ctrl.throttle = 1.0 if keys[pygame.K_w] else 0.0
                    ctrl.brake = 1.0 if keys[pygame.K_s] else 0.0
                    ctrl.steer = -0.5 if keys[pygame.K_a] else (0.5 if keys[pygame.K_d] else 0.0)
                    self.vehicle.apply_control(ctrl)

                # 3. CAN Logging
                self._process_can_frames()
                
                # 4. EDR Logging
                v = self.vehicle.get_velocity()
                a = self.vehicle.get_acceleration()
                ang = self.vehicle.get_angular_velocity()
                t = self.vehicle.get_transform()
                c = self.vehicle.get_control()

                snapshot = {
                    'timestamp': datetime.now().isoformat(),
                    'sim_time': self.world.get_snapshot().timestamp.elapsed_seconds,
                    'velocity_x': v.x,
                    'velocity_y': v.y,
                    'velocity_z': v.z,
                    'speed_kmh': VehiclePhysics.calculate_speed_kmh(v),
            
                    # IMU Data: Linear Acceleration (m/s^2)
                    'accel_x': a.x,
                    'accel_y': a.y,
                    'accel_z': a.z,
                    'total_g_force': VehiclePhysics.calculate_magnitude(a) / 9.81,
            
                    # IMU Data: Angular Velocity (deg/s)
                    'gyro_roll': ang.x,
                    'gyro_pitch': ang.y,
                    'gyro_yaw': ang.z,
            
                    # Orientation
                    'pitch': t.rotation.pitch,
                    'roll': t.rotation.roll,
                    'yaw': t.rotation.yaw,
            
                    # Driver Inputs
                    'throttle': c.throttle,
                    'brake': c.brake,
                    'steer': c.steer,
                    'hand_brake': c.hand_brake,
            
                    # Location
                    'lat': self.gps_data['lat'],
                    'lon': self.gps_data['lon']
                }
                self.edr_buffer.add(snapshot)

                clock.tick(1.0 / self.config["EDR_TICK_INTERVAL"])

        except KeyboardInterrupt:
            logging.info("User interrupted simulation.")
        finally:
            self.cleanup()

    def cleanup(self):
        logging.info("Cleaning up...")
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        
        for s in self.sensors: s.destroy()
        if self.vehicle: self.vehicle.destroy()
        self.can_manager.shutdown()
        pygame.quit()

if __name__ == '__main__':
    monitor = IntegratedVehicleMonitor(CONFIG)
    monitor.run()
