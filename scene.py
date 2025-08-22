import os
import time
from datetime import datetime
from os.path import join
import sys
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import torch
import operator, random
from copy import deepcopy
from gymnasium import spaces

from isaacgym import gymapi, gymtorch, gymutil
from isaacgymenvs.utils.torch_jit_utils import to_torch
from isaacgymenvs.utils.dr_utils import get_property_setter_map, get_property_getter_map, \
    get_default_setter_args, apply_random_samples, check_buckets, generate_random_samples

# get default set of parameters
# sim_params = gymapi.SimParams()

# # set common parameters
# sim_params.dt = 1 / 60
# sim_params.substeps = 2
# sim_params.up_axis = gymapi.UP_AXIS_Z
# sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# # set PhysX-specific parameters
# sim_params.physx.use_gpu = True
# sim_params.physx.solver_type = 1
# sim_params.physx.num_position_iterations = 6
# sim_params.physx.num_velocity_iterations = 1
# sim_params.physx.contact_offset = 0.01
# sim_params.physx.rest_offset = 0.0

SCREEN_CAPTURE_RESOLUTION = (1027, 768)

class Scene:
    def __init__(self, n_envs, spacing, sim_params, physics_engine, device_id, graphics_device_id, device, headless, virtual_screen_capture: bool = False, force_render: bool = False):
        self.n_envs = n_envs
        self.spacing = spacing
        self.device_id = device_id
        self.graphics_device_id = graphics_device_id
        self.physics_engine = physics_engine
        self.sim_params = sim_params
        self.device = device
        self.headless = headless
        self.envs = []
        self.envs_per_row = int(np.sqrt(n_envs))
        self.entity_list = []
        self.camera_list = []
        self.total_rigid_body = 0
        self.total_rigid_shape = 0

        self.total_train_env_frames: int = 0

        # number of control steps
        self.dt = sim_params.dt

        self.virtual_screen_capture = virtual_screen_capture
        self.virtual_display = None
        if self.virtual_screen_capture:
            from pyvirtualdisplay.smartdisplay import SmartDisplay
            self.virtual_display = SmartDisplay(size=SCREEN_CAPTURE_RESOLUTION)
            self.virtual_display.start()
            
        self.force_render = force_render
        self.control_freq_inv = 1.0
        self.control_steps: int = 0

        self.render_fps: int = -1
        self.last_frame_time: float = 0.0

        self.record_frames: bool = False
        self.record_frames_dir = join("recorded_frames", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        self._setup()

    def _setup(self):
        self.gym = gymapi.acquire_gym()
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = self.gym.create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        # Add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1) # Z-up
        plane_params.distance = 0
        self.gym.add_ground(self.sim, plane_params)

    def build(self, cam_pos = (0.0, 1.5, 1.8), cam_target = (0.0, 0.75, 1.3)):
        # Define env grid
        env_lower = gymapi.Vec3(-self.spacing, -self.spacing, 0.0) # Adjusted lower bound for more space
        env_upper = gymapi.Vec3(self.spacing, self.spacing, self.spacing) # Adjusted upper bound
        for i in range(self.n_envs):
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, self.envs_per_row)
            self.envs.append(env_ptr)
            self.gym.begin_aggregate(env_ptr, self.total_rigid_body, self.total_rigid_shape, True)

            """Add entities"""
            for j, entity in enumerate(self.entity_list):

                # Set position and orientation
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(
                    entity.init_pos[0],
                    entity.init_pos[1],
                    entity.init_pos[2]
                )
                pose.r = gymapi.Quat(
                    entity.init_quat[0],
                    entity.init_quat[1],
                    entity.init_quat[2],
                    entity.init_quat[3]
                )
                # Add entity to environment
                handle = self.gym.create_actor(env_ptr, entity.asset, pose, f"entity-{j}", i, 0)
                actor_index = self.gym.get_actor_index(env_ptr, handle, gymapi.DOMAIN_SIM)

                dof_list = []
                num_dof = self.gym.get_asset_dof_count(entity.asset)
                for k in range(num_dof):
                    index = self.gym.get_actor_dof_index(env_ptr, handle, k, gymapi.DOMAIN_SIM)
                    dof_list.append(index)

                rigid_bodies_list = []
                num_bodies = self.gym.get_asset_rigid_body_count(entity.asset)
                for k in range(num_bodies):
                    index = self.gym.get_actor_rigid_body_index(env_ptr, handle, k, gymapi.DOMAIN_SIM)
                    rigid_bodies_list.append(index)

                # Link dict
                entity.link_dict = self.gym.get_asset_rigid_body_dict(entity.asset)

                shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
                for k, s in enumerate(shape_props):
                    s.filter = (1<<(j+1))
                    s.friction = entity.friction
                    s.rolling_friction = entity.rolling_friction
                    s.torsion_friction = entity.torsion_friction

                self.gym.set_actor_rigid_shape_properties(env_ptr, handle, shape_props)

                if actor_index < 0:
                    raise RuntimeError(
                        f"Failed to create actor {j} in environment {i}. Actor index: {actor_index}")

                entity.handles.append(handle)
                entity.actor_indices.append(actor_index)
                entity.dof_indices.append(dof_list)
                entity.rigid_body_indices.append(rigid_bodies_list)

                # Set DOF properties (drive mode, stiffness, damping) if not None
                dof_props = self.gym.get_actor_dof_properties(env_ptr, handle)
                entity.n_dofs = len(dof_props)

                if entity.stiffness is not None:
                    dof_props['stiffness'] = entity.stiffness
                if entity.damping is not None:
                    dof_props['damping'] = entity.damping

                if i == 0:  # Only set limits for the first environment
                    for k in range(entity.n_dofs):
                        entity.lower_limits.append(dof_props['lower'][k])
                        entity.upper_limits.append(dof_props['upper'][k])

                self.gym.set_actor_dof_properties(env_ptr, handle, dof_props)

            """Add cameras"""
            for camera in self.camera_list:
                # Add camera to environment
                cam_props = gymapi.CameraProperties()
                cam_props.width = camera.res[0]
                cam_props.height = camera.res[1]
                cam_props.enable_tensors = True
                handle = self.gym.create_camera_sensor(env_ptr, cam_props)
                self.gym.set_camera_location(handle, handle, camera.pos, camera.lookat)
                camera.handles.append(handle)

            self.gym.end_aggregate(env_ptr)

        self.gym.prepare_sim(self.sim)

        """Initialize state tensors"""
        self.actor_root_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim)).reshape(-1, 13)
        self.dof_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)).reshape(-1, 2)
        self.dof_force_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_force_tensor(self.sim)).reshape(-1, 1)
        self.rigid_body_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)).reshape(-1, 13)
        self.net_contact_force_tensor = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim)).reshape(-1, 3)

        self.set_actor_root_state_tensor = torch.zeros_like(self.actor_root_state_tensor, device=self.device)
        self.set_dof_state_tensor = torch.zeros_like(self.dof_state_tensor, device=self.device)
        self.control_dof_state_tensor = torch.zeros_like(self.dof_state_tensor, device=self.device)
        self.set_rigid_body_state_tensor = torch.zeros_like(self.rigid_body_state_tensor, device=self.device)

        self.set_actor_root_state_indices = []
        self.set_dof_state_indices = []
        self.control_dof_state_indices = []
        self.set_rigid_body_state_indices = []

        """Set up viewer"""
        self.set_viewer(cam_pos, cam_target)

    def set_viewer(self, cam_pos, cam_target):
        """Create the viewer."""

        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_R, "record_frames")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)

            cam_pos = gymapi.Vec3([cam_pos[0], cam_pos[1], cam_pos[2]])
            cam_target = gymapi.Vec3([cam_target[0], cam_target[1], cam_target[2]])

            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

    def add_entity(self, asset_root, asset_file, pos, quat, fixed, robot = False, disable_gravity = False, drive_mode = None, friction = 1.0, rolling_friction = 0.01, torsion_friction = 0.01):
        print(f"Adding entity: {asset_file}")
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = fixed
        asset_options.flip_visual_attachments = False # Often needed for URDFs
        asset_options.disable_gravity = disable_gravity
        asset_options.thickness = 0.001
        if drive_mode is None:
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        elif entity.drive_mode == 'pos':
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        elif entity.drive_mode == 'vel':
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL

        if robot == True:
            asset_options.angular_damping = 20
            asset_options.linear_damping = 20
            asset_options.collapse_fixed_joints = False
            asset_options.use_mesh_materials = True
        else:
            asset_options.use_mesh_materials = False
            asset_options.override_com = True
            asset_options.override_inertia = True
            asset_options.convex_decomposition_from_submeshes = True
            asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 200000
            asset_options.density = 200  # * the average density of low-fill-rate 3D-printed models
            

        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(asset)
        self.gym.set_asset_rigid_shape_properties(asset, rigid_shape_props_asset)
        self.total_rigid_body += self.gym.get_asset_rigid_body_count(asset)
        self.total_rigid_shape += self.gym.get_asset_rigid_shape_count(asset)
        entity = Rigid_entity(self, asset, pos, quat, friction, 
                              rolling_friction, torsion_friction)
        self.entity_list.append(entity)
        return entity
    
    def add_camera(self, res, pos, lookat):
        camera = Camera(self, res, pos, lookat)
        self.camera_list.append(camera)
        return camera
    
    def step(self):
        self._apply_buffer()
        self._step()
        self._refresh_buffer()

    def _apply_buffer(self):
        if len(self.set_actor_root_state_indices) > 0:
            indices = torch.tensor(self.set_actor_root_state_indices, dtype=torch.int32, device=self.device)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, 
                gymtorch.unwrap_tensor(self.set_actor_root_state_tensor),
                gymtorch.unwrap_tensor(indices),
                len(indices)
            )
        if len(self.set_dof_state_indices) > 0:
            indices = torch.tensor(self.set_dof_state_indices, dtype=torch.int32, device=self.device)
            self.gym.set_dof_state_tensor_indexed(
                self.sim, 
                gymtorch.unwrap_tensor(self.set_dof_state_tensor),
                gymtorch.unwrap_tensor(indices),
                len(indices)
            )
        if len(self.control_dof_state_indices) > 0:
            indices = torch.tensor(self.control_dof_state_indices, dtype=torch.int32, device=self.device)
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim, 
                gymtorch.unwrap_tensor(self.control_dof_state_tensor[:, 0].contiguous()),
                gymtorch.unwrap_tensor(indices),
                len(indices)
            )

    def _refresh_buffer(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.set_actor_root_state_indices = []
        self.set_dof_state_indices = []
        self.control_dof_state_indices = []
        self.set_rigid_body_state_indices = []

    def _step(self):
        # Step the simulation
        self.gym.simulate(self.sim)

    def render(self, mode="rgb_array"):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "record_frames" and evt.value > 0:
                    self.record_frames = not self.record_frames

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

                # it seems like in some cases sync_frame_time still results in higher-than-realtime framerate
                # this code will slow down the rendering to real time
                now = time.time()
                delta = now - self.last_frame_time
                if self.render_fps < 0:
                    # render at control frequency
                    render_dt = self.dt * self.control_freq_inv  # render every control step
                else:
                    render_dt = 1.0 / self.render_fps

                if delta < render_dt:
                    time.sleep(render_dt - delta)

                self.last_frame_time = time.time()

            else:
                self.gym.poll_viewer_events(self.viewer)

            if self.record_frames:
                if not os.path.isdir(self.record_frames_dir):
                    os.makedirs(self.record_frames_dir, exist_ok=True)

                self.gym.write_viewer_image_to_file(self.viewer, join(self.record_frames_dir, f"frame_{self.control_steps}.png"))

            if self.virtual_display and mode == "rgb_array":
                img = self.virtual_display.grab()
                return np.array(img)
            
    """
    Domain Randomization methods
    """

    def get_actor_params_info(self, dr_params: Dict[str, Any], env):
        """Generate a flat array of actor params, their names and ranges.

        Returns:
            The array
        """

        if "actor_params" not in dr_params:
            return None
        params = []
        names = []
        lows = []
        highs = []
        param_getters_map = get_property_getter_map(self.gym)
        for actor, actor_properties in dr_params["actor_params"].items():
            handle = self.gym.find_actor_handle(env, actor)
            for prop_name, prop_attrs in actor_properties.items():
                if prop_name == 'color':
                    continue  # this is set randomly
                props = param_getters_map[prop_name](env, handle)
                if not isinstance(props, list):
                    props = [props]
                for prop_idx, prop in enumerate(props):
                    for attr, attr_randomization_params in prop_attrs.items():
                        name = prop_name+'_' + str(prop_idx) + '_'+attr
                        lo_hi = attr_randomization_params['range']
                        distr = attr_randomization_params['distribution']
                        if 'uniform' not in distr:
                            lo_hi = (-1.0*float('Inf'), float('Inf'))
                        if isinstance(prop, np.ndarray):
                            for attr_idx in range(prop[attr].shape[0]):
                                params.append(prop[attr][attr_idx])
                                names.append(name+'_'+str(attr_idx))
                                lows.append(lo_hi[0])
                                highs.append(lo_hi[1])
                        else:
                            params.append(getattr(prop, attr))
                            names.append(name)
                            lows.append(lo_hi[0])
                            highs.append(lo_hi[1])
        return params, names, lows, highs

    def apply_randomizations(self, dr_params):
        """Apply domain randomizations to the environment.

        Note that currently we can only apply randomizations only on resets, due to current PhysX limitations

        Args:
            dr_params: parameters for domain randomization to use.
        """

        # If we don't have a randomization frequency, randomize every step
        rand_freq = dr_params.get("frequency", 1)

        # First, determine what to randomize:
        #   - non-environment parameters when > frequency steps have passed since the last non-environment
        #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
        #   - on the first call, randomize everything
        self.last_step = self.gym.get_frame_count(self.sim)
        if self.first_randomization:
            do_nonenv_randomize = True
            env_ids = list(range(self.num_envs))
        else:
            do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
            rand_envs = torch.where(self.randomize_buf >= rand_freq, torch.ones_like(self.randomize_buf), torch.zeros_like(self.randomize_buf))
            rand_envs = torch.logical_and(rand_envs, self.reset_buf)
            env_ids = torch.nonzero(rand_envs, as_tuple=False).squeeze(-1).tolist()
            self.randomize_buf[rand_envs] = 0

        if do_nonenv_randomize:
            self.last_rand_step = self.last_step

        param_setters_map = get_property_setter_map(self.gym)
        param_setter_defaults_map = get_default_setter_args(self.gym)
        param_getters_map = get_property_getter_map(self.gym)

        # On first iteration, check the number of buckets
        if self.first_randomization:
            check_buckets(self.gym, self.envs, dr_params)

        for nonphysical_param in ["observations", "actions"]:
            if nonphysical_param in dr_params and do_nonenv_randomize:
                dist = dr_params[nonphysical_param]["distribution"]
                op_type = dr_params[nonphysical_param]["operation"]
                sched_type = dr_params[nonphysical_param]["schedule"] if "schedule" in dr_params[nonphysical_param] else None
                sched_step = dr_params[nonphysical_param]["schedule_steps"] if "schedule" in dr_params[nonphysical_param] else None
                op = operator.add if op_type == 'additive' else operator.mul

                if sched_type == 'linear':
                    sched_scaling = 1.0 / sched_step * \
                        min(self.last_step, sched_step)
                elif sched_type == 'constant':
                    sched_scaling = 0 if self.last_step < sched_step else 1
                else:
                    sched_scaling = 1

                if dist == 'gaussian':
                    mu, var = dr_params[nonphysical_param]["range"]
                    mu_corr, var_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        mu *= sched_scaling
                        var *= sched_scaling
                        mu_corr *= sched_scaling
                        var_corr *= sched_scaling
                    elif op_type == 'scaling':
                        var = var * sched_scaling  # scale up var over time
                        mu = mu * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                        var_corr = var_corr * sched_scaling  # scale up var over time
                        mu_corr = mu_corr * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * params['var_corr'] + params['mu_corr']
                        return op(
                            tensor, corr + torch.randn_like(tensor) * params['var'] + params['mu'])

                    self.dr_randomizations[nonphysical_param] = {'mu': mu, 'var': var, 'mu_corr': mu_corr, 'var_corr': var_corr, 'noise_lambda': noise_lambda}

                elif dist == 'uniform':
                    lo, hi = dr_params[nonphysical_param]["range"]
                    lo_corr, hi_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        lo *= sched_scaling
                        hi *= sched_scaling
                        lo_corr *= sched_scaling
                        hi_corr *= sched_scaling
                    elif op_type == 'scaling':
                        lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        lo_corr = lo_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi_corr = hi_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * (params['hi_corr'] - params['lo_corr']) + params['lo_corr']
                        return op(tensor, corr + torch.rand_like(tensor) * (params['hi'] - params['lo']) + params['lo'])

                    self.dr_randomizations[nonphysical_param] = {'lo': lo, 'hi': hi, 'lo_corr': lo_corr, 'hi_corr': hi_corr, 'noise_lambda': noise_lambda}

        if "sim_params" in dr_params and do_nonenv_randomize:
            prop_attrs = dr_params["sim_params"]
            prop = self.gym.get_sim_params(self.sim)

            if self.first_randomization:
                self.original_props["sim_params"] = {
                    attr: getattr(prop, attr) for attr in dir(prop)}

            for attr, attr_randomization_params in prop_attrs.items():
                apply_random_samples(
                    prop, self.original_props["sim_params"], attr, attr_randomization_params, self.last_step)

            self.gym.set_sim_params(self.sim, prop)

        # If self.actor_params_generator is initialized: use it to
        # sample actor simulation params. This gives users the
        # freedom to generate samples from arbitrary distributions,
        # e.g. use full-covariance distributions instead of the DR's
        # default of treating each simulation parameter independently.
        extern_offsets = {}
        if self.actor_params_generator is not None:
            for env_id in env_ids:
                self.extern_actor_params[env_id] = \
                    self.actor_params_generator.sample()
                extern_offsets[env_id] = 0

        # randomise all attributes of each actor (hand, cube etc..)
        # actor_properties are (stiffness, damping etc..)

        # Loop over actors, then loop over envs, then loop over their props 
        # and lastly loop over the ranges of the params 

        for actor, actor_properties in dr_params["actor_params"].items():

            # Loop over all envs as this part is not tensorised yet 
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, actor)
                extern_sample = self.extern_actor_params[env_id]

                # randomise dof_props, rigid_body, rigid_shape properties 
                # all obtained from the YAML file
                # EXAMPLE: prop name: dof_properties, rigid_body_properties, rigid_shape properties  
                #          prop_attrs: 
                #               {'damping': {'range': [0.3, 3.0], 'operation': 'scaling', 'distribution': 'loguniform'}
                #               {'stiffness': {'range': [0.75, 1.5], 'operation': 'scaling', 'distribution': 'loguniform'}
                for prop_name, prop_attrs in actor_properties.items():
                    if prop_name == 'color':
                        num_bodies = self.gym.get_actor_rigid_body_count(
                            env, handle)
                        for n in range(num_bodies):
                            self.gym.set_rigid_body_color(env, handle, n, gymapi.MESH_VISUAL,
                                                          gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
                        continue

                    if prop_name == 'scale':
                        setup_only = prop_attrs.get('setup_only', False)
                        if (setup_only and not self.sim_initialized) or not setup_only:
                            attr_randomization_params = prop_attrs
                            sample = generate_random_samples(attr_randomization_params, 1,
                                                             self.last_step, None)
                            og_scale = 1
                            if attr_randomization_params['operation'] == 'scaling':
                                new_scale = og_scale * sample
                            elif attr_randomization_params['operation'] == 'additive':
                                new_scale = og_scale + sample
                            self.gym.set_actor_scale(env, handle, new_scale)
                        continue

                    prop = param_getters_map[prop_name](env, handle)
                    set_random_properties = True

                    if isinstance(prop, list):
                        if self.first_randomization:
                            self.original_props[prop_name] = [
                                {attr: getattr(p, attr) for attr in dir(p)} for p in prop]
                        for p, og_p in zip(prop, self.original_props[prop_name]):
                            for attr, attr_randomization_params in prop_attrs.items():
                                setup_only = attr_randomization_params.get('setup_only', False)
                                if (setup_only and not self.sim_initialized) or not setup_only:
                                    smpl = None
                                    if self.actor_params_generator is not None:
                                        smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                            extern_sample, extern_offsets[env_id], p, attr)
                                    apply_random_samples(
                                        p, og_p, attr, attr_randomization_params,
                                        self.last_step, smpl)
                                else:
                                    set_random_properties = False
                    else:
                        if self.first_randomization:
                            self.original_props[prop_name] = deepcopy(prop)
                        for attr, attr_randomization_params in prop_attrs.items():
                            setup_only = attr_randomization_params.get('setup_only', False)
                            if (setup_only and not self.sim_initialized) or not setup_only:
                                smpl = None
                                if self.actor_params_generator is not None:
                                    smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                        extern_sample, extern_offsets[env_id], prop, attr)
                                apply_random_samples(
                                    prop, self.original_props[prop_name], attr,
                                    attr_randomization_params, self.last_step, smpl)
                            else:
                                set_random_properties = False

                    if set_random_properties:
                        setter = param_setters_map[prop_name]
                        default_args = param_setter_defaults_map[prop_name]
                        setter(env, handle, prop, *default_args)

        if self.actor_params_generator is not None:
            for env_id in env_ids:  # check that we used all dims in sample
                if extern_offsets[env_id] > 0:
                    extern_sample = self.extern_actor_params[env_id]
                    if extern_offsets[env_id] != extern_sample.shape[0]:
                        print('env_id', env_id,
                              'extern_offset', extern_offsets[env_id],
                              'vs extern_sample.shape', extern_sample.shape)
                        raise Exception("Invalid extern_sample size")

        self.first_randomization = False



class Rigid_entity:
    def __init__(self, scene, asset, init_pos, init_quat, friction, rolling_friction, torsion_friction):
        self.scene = scene
        self.asset = asset
        self.handles = []
        self.actor_indices = []
        self.dof_indices = []
        self.rigid_body_indices = []
        self.lower_limits = []
        self.upper_limits = []
        self.init_pos = init_pos
        self.init_quat = init_quat
        self.friction = friction
        self.rolling_friction = rolling_friction
        self.torsion_friction = torsion_friction
        self.n_dof = None
        self.stiffness = None
        self.damping = None
        self.link_dict = None

    def set_friction(self, friction, env_ids = None):
        if env_ids is None:
            env_ids = [i for i in range(self.scene.n_envs)]
        for i, env_ptr in enumerate(self.scene.envs):
            handle = self.handles[i]
            shape_props = self.scene.gym.get_actor_rigid_shape_properties(env_ptr, handle)
            for s in shape_props:
                s.friction = friction
            self.scene.gym.set_actor_rigid_shape_properties(env_ptr, handle, shape_props)

    def set_mass(self, mass):
        pass

    def set_dofs_kp(self, kp):
        self.stiffness = kp

    def set_dofs_kv(self, kv):
        self.damping = kv

    def set_state(self, state, env_ids = None):
        if env_ids is None:
            env_ids = [i for i in range(self.scene.n_envs)]
        actor_indices = []
        for env_idx in env_ids:
            actor_indices.append(self.actor_indices[env_idx])

        self.scene.set_actor_root_state_tensor[actor_indices, :13] = state
        self.scene.set_actor_root_state_indices.extend(actor_indices)

    def set_pose(self, pos, quat, env_ids = None):
        if env_ids is None:
            env_ids = [i for i in range(self.scene.n_envs)]
        actor_indices = []
        for env_idx in env_ids:
            actor_indices.append(self.actor_indices[env_idx])

        self.scene.set_actor_root_state_tensor[actor_indices, 0:3] = pos
        self.scene.set_actor_root_state_tensor[actor_indices, 3:7] = quat
        self.scene.set_actor_root_state_tensor[actor_indices, 7:13] = 0.0
        self.scene.set_actor_root_state_indices.extend(actor_indices)

    def set_vel(self, vel, env_ids = None):
        if env_ids is None:
            env_ids = [i for i in range(self.scene.n_envs)]
        actor_indices = []
        for env_idx in env_ids:
            actor_indices.append(self.actor_indices[env_idx])

        self.scene.set_actor_root_state_tensor[actor_indices, 0:7] = 0.0
        self.scene.set_actor_root_state_tensor[actor_indices, 7:13] = vel
        self.scene.set_actor_root_state_indices.extend(actor_indices)

    def set_dofs_state(self, state, env_ids = None):
        if env_ids is None:
            env_ids = [i for i in range(self.scene.n_envs)]
        dof_indices = []
        for env_idx in env_ids:
            dof_indices.extend(self.dof_indices[env_idx])

        actor_indices = []
        for env_idx in env_ids:
            actor_indices.append(self.actor_indices[env_idx])

        state = state.reshape(-1, 2)
        self.scene.set_dof_state_tensor[dof_indices] = state
        self.scene.set_dof_state_indices.extend(actor_indices)

    def set_dofs_position(self, position, env_ids = None):
        if env_ids is None:
            env_ids = [i for i in range(self.scene.n_envs)]
        dof_indices = []
        for env_idx in env_ids:
            dof_indices.extend(self.dof_indices[env_idx])

        actor_indices = []
        for env_idx in env_ids:
            actor_indices.append(self.actor_indices[env_idx])

        self.scene.set_dof_state_tensor[dof_indices, 0] = position.reshape(-1)
        self.scene.set_dof_state_tensor[dof_indices, 1] = 0.0
        self.scene.set_dof_state_indices.extend(actor_indices)

    def control_dofs_position(self, position, env_ids = None):
        if env_ids is None:
            env_ids = [i for i in range(self.scene.n_envs)]
        dof_indices = []
        for env_idx in env_ids:
            dof_indices.extend(self.dof_indices[env_idx])

        actor_indices = []
        for env_idx in env_ids:
            actor_indices.append(self.actor_indices[env_idx])

        self.scene.control_dof_state_tensor[dof_indices, 0] = position.reshape(-1)
        self.scene.control_dof_state_tensor[dof_indices, 1] = 0.0
        self.scene.control_dof_state_indices.extend(actor_indices)

    def get_state(self, env_ids = None):
        if env_ids is None:
            env_ids = [i for i in range(self.scene.n_envs)]
        actor_indices = []
        for env_idx in env_ids:
            actor_indices.append(self.actor_indices[env_idx])

        state = self.scene.actor_root_state_tensor[actor_indices, :13]
        return state

    def get_pose(self, env_ids = None):
        if env_ids is None:
            env_ids = [i for i in range(self.scene.n_envs)]
        actor_indices = []
        for env_idx in env_ids:
            actor_indices.append(self.actor_indices[env_idx])

        pose = self.scene.actor_root_state_tensor[actor_indices, 0:7]
        return pose
    
    def get_vel(self, env_ids = None):
        if env_ids is None:
            env_ids = [i for i in range(self.scene.n_envs)]
        actor_indices = []
        for env_idx in env_ids:
            actor_indices.append(self.actor_indices[env_idx])

        vel = self.scene.actor_root_state_tensor[actor_indices, 7:13]
        return vel
    
    def get_dofs_state(self, env_ids = None):
        if env_ids is None:
            env_ids = [i for i in range(self.scene.n_envs)]
        dof_indices = []
        for env_idx in env_ids:
            dof_indices.extend(self.dof_indices[env_idx])

        state = self.scene.dof_state_tensor[dof_indices, :]
        return state.reshape(len(env_ids), -1, 2)
    
    def get_dofs_force(self, env_ids = None):
        if env_ids is None:
            env_ids = [i for i in range(self.scene.n_envs)]
        dof_indices = []
        for env_idx in env_ids:
            dof_indices.extend(self.dof_indices[env_idx])

        force = self.scene.dof_force_tensor[dof_indices, :]
        return force.reshape(len(env_ids), -1)

    def get_dofs_position(self, env_ids = None):
        if env_ids is None:
            env_ids = [i for i in range(self.scene.n_envs)]
        dof_indices = []
        for env_idx in env_ids:
            dof_indices.extend(self.dof_indices[env_idx])

        position = self.scene.dof_state_tensor[dof_indices, 0]
        return position.reshape(len(env_ids), -1)
    
    def get_dofs_velocity(self, env_ids = None):
        if env_ids is None:
            env_ids = [i for i in range(self.scene.n_envs)]
        dof_indices = []
        for env_idx in env_ids:
            dof_indices.extend(self.dof_indices[env_idx])

        velocity = self.scene.dof_state_tensor[dof_indices, 1]
        return velocity.reshape(len(env_ids), -1)

    def get_rigid_body_state(self, env_ids = None):
        if env_ids is None:
            env_ids = [i for i in range(self.scene.n_envs)]
        rigid_body_indices = []
        for env_idx in env_ids:
            rigid_body_indices.extend(self.rigid_body_indices[env_idx])

        state = self.scene.rigid_body_state_tensor[rigid_body_indices, :13]
        return state.reshape(len(env_ids), -1, 13)

    def get_link_pose(self, link_name, env_ids = None):
        if env_ids is None:
            env_ids = [i for i in range(self.scene.n_envs)]
        link_index = self.link_dict[link_name]
        rigid_body_state = self.scene.rigid_body_state_tensor.reshape(self.scene.n_envs, -1, 13)

        return rigid_body_state[env_ids, link_index, 0:7]
    
    def get_link_state(self, link_name, env_ids = None):
        if env_ids is None:
            env_ids = [i for i in range(self.scene.n_envs)]
        link_index = self.link_dict[link_name]
        rigid_body_state = self.scene.rigid_body_state_tensor.reshape(self.scene.n_envs, -1, 13)

        return rigid_body_state[env_ids, link_index, :13]
    
    def get_link_contact_force(self, link_name, env_ids = None):
        if env_ids is None:
            env_ids = [i for i in range(self.scene.n_envs)]
        link_index = self.link_dict[link_name]
        contact_state = self.scene.net_contact_force_tensor.reshape(self.scene.n_envs, -1, 3)

        return contact_state[env_ids, link_index, 0:3]
    
    def get_dofs_limit(self):
        return torch.tensor(self.lower_limits, device=self.scene.device), torch.tensor(self.upper_limits, device=self.scene.device)
    
    def get_dofs_name(self):
        env = self.scene.envs[0]
        actor_handle = self.handles[0]
        joint_names = self.scene.gym.get_actor_dof_names(env, actor_handle)
        return joint_names

    def inverse_kinematics_multilink(self):
        pass

def look_at_rotation(camera_pos, target_pos, up_vec=np.array([0,0,1])):
    forward = (target_pos - camera_pos)
    forward = forward / np.linalg.norm(forward)

    right = np.cross(up_vec, forward)
    right = right / np.linalg.norm(right)

    up = np.cross(forward, right)

    # Rotation matrix
    rot_mat = np.array([right, up, forward]).T  # 3x3

    from scipy.spatial.transform import Rotation as R
    r = R.from_matrix(rot_mat)
    q = r.as_quat()  # [x, y, z, w]

    return (q[0], q[1], q[2], q[3])

class Camera:
    def __init__(self, scene, res, pos, lookat):
        self.scene = scene
        self.handles = []
        self.res = res
        self.pos = pos
        self.lookat = lookat

    def render(self):
        self.scene.gym.render_all_camera_sensors(self.sim)
        self.scene.gym.fetch_results(self.sim, True)
        self.scene.gym.step_graphics(self.sim)
        # get image tensor:
        imgs = []
        for i in range(self.scene.n_envs):
            env_ptr = self.scene.envs[i]
            handle = self.handles[i]
            img = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, handle, gymapi.IMAGE_COLOR)
            img_tensor = gymtorch.wrap_tensor(img)
            imgs.append(img_tensor)
        imgs = torch.stack(imgs, dim=0)
        imgs = imgs.reshape(self.scene.n_envs, self.res[0], self.res[1], -1)
        return imgs



