from __future__ import annotations

import argparse
import glob
import os
import time
from dataclasses import dataclass, field
import numpy as np
import h5py
import pyvista as pv
import vtk
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class PlaybackState:
    current_frame: int = 0
    n_frames: int = 0
    n_particles: int = 0
    fps: int = 30
    playback_speed: int = 1
    radius: float | None = None
    selected_particle_idx: int | None = None
    is_playing: bool = False
    timer_id: int = -1
    vtk_timer_id: int = -1
 
@dataclass
class UIComponents:
    """Class to hold references to UI widgets."""
    info_text: any = None
    help_text: any = None
    slider: any = None
    speed_widget: any = None
    
class InteractivePlayer:
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.state = PlaybackState()
        self.ui = UIComponents()
        self._read_data()

        self.plotter = pv.Plotter(window_size=[1280, 960], title="3D Interactive Trajectory Player (Sphere Mesh)")
        
        self._setup_scene()
        self._init_widgets()
        self._bind_events()
        
        self._render_frame(0)
        self.plotter.reset_camera()

    def _read_data(self):
        try:
            self.h5 = h5py.File(self.hdf5_path, "r")
        except Exception as exc:
            raise RuntimeError("This script requires h5py. Install it via: pip install h5py") from exc
        try:
            self.positions = self.h5["position"][:]
            self.velocities = self.h5["velocity"][:]
            self.time_steps = self.h5["time"][:]
            self.sim_steps = self.h5["step"][:] if "step" in self.h5 else np.arange(self.time_steps.size, dtype=np.int64)
        except Exception as exc:
            raise RuntimeError("This script requires PyVista + VTK. Install via: pip install pyvista vtk") from exc

        if self.time_steps.size == 0:
            raise ValueError("Trajectory has no frames.")

        self.state.n_frames = int(self.time_steps.size)
        self.state.n_particles = int(self.positions.shape[1])
        self.speeds = np.linalg.norm(self.velocities, axis=-1)
        self.speed_min = 0.0
        self.speed_max = float(np.nanmax(self.speeds[0]))

        radius_value = self._read_particle_radius()
        if radius_value is None:
            radius_value = 1.0

        if np.isscalar(radius_value):
            self.radii = np.full(self.state.n_particles, float(radius_value))
        else:
            radius_array = np.asarray(radius_value, dtype=float)
            if radius_array.size == self.state.n_particles:
                self.radii = radius_array
            else:
                self.radii = np.full(self.state.n_particles, float(np.mean(radius_array)))

        self.sphere_template = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10)

    def _read_particle_radius(self):
        if self.state.radius is not None:
            try:
                val = float(self.state.radius)
                if np.isfinite(val) and val > 0.0:
                    return val
            except Exception:
                pass

        attrs = self.h5.attrs
        for key in ("particle_radius", "r"):
            if key in attrs:
                try:
                    val = float(attrs[key])
                    if np.isfinite(val) and val > 0.0:
                        return val
                except Exception:
                    pass

        if "meta" in self.h5:
            meta_attrs = self.h5["meta"].attrs
            for key in ("materials.r", "r", "particle_radius"):
                if key in meta_attrs:
                    try:
                        val = float(meta_attrs[key])
                        if np.isfinite(val) and val > 0.0:
                            return val
                    except Exception:
                        pass
        return None

    def _setup_scene(self):
        """Sets up the initial scene, including lights and the main particle actor."""
        self.plotter.add_light(pv.Light(position=(10, 10, 10), intensity=1.5))
        self.plotter.add_light(pv.Light(position=(-10, -10, -10), intensity=1.5))

        initial_positions = self.positions[0]
        initial_speeds = self.speeds[0]
        
        self.glyph_mesh = self._build_glyph_mesh(initial_positions, initial_speeds)
        
        self.particle_actor = self.plotter.add_mesh(
            self.glyph_mesh,
            scalars='speed',
            cmap='viridis',
            clim=(self.speed_min, self.speed_max),
            scalar_bar_args={
                'title': "Speed (m/s)",
                'vertical': True,
                'position_x': 0.9,
                'position_y': 0.1,
                'height': 0.8,
                'width': 0.08,
                'title_font_size': 20,
                'label_font_size': 16,
                'fmt': '%.2e'
            }
        )
        
        self.selection_marker = pv.Sphere(radius=1.0)
        self.selection_actor = self.plotter.add_mesh(self.selection_marker, color='red', opacity=0.5)
        try:
            self.selection_actor.SetVisibility(False)
        except AttributeError:
            self.selection_actor.visibility = False
        
        self.plotter.show_grid()

    def _build_glyph_mesh(self, positions, speeds):
        """Builds a glyph mesh for the particles."""
        source_mesh = pv.PolyData(positions)
        source_mesh['speed'] = speeds
        source_mesh['radius'] = self.radii
        
        # The glyph filter will copy scalar data from the source points to the glyphs.
        # It's important that the sphere_template does not have active scalars itself.
        return source_mesh.glyph(
            orient=False,  # No orientation data
            scale='radius',
            factor=1.0,
            geom=self.sphere_template
        )

    def _init_widgets(self):
        """Initializes all UI widgets."""
        # --- Info Text (Top Left) ---
        self.ui.info_text = self.plotter.add_text(
            "", position='upper_left', font_size=14, font='courier'
        )

        # --- Help Text (Bottom Left) ---
        self.ui.help_text = self.plotter.add_text(
            "", position=(0.02, 0.02), font_size=11, font='courier', viewport=True
        )
        self._update_help_text()

        # --- Slider ---
        self.ui.slider = self.plotter.add_slider_widget(
            callback=self._on_slider_update,
            rng=[float(self.time_steps[0]), float(self.time_steps[-1])],
            value=float(self.time_steps[0]),
            pointa=(0.25, 0.03),
            pointb=(0.75, 0.03),
            style='modern',
            color='gray',
            title="Time (s)"
        )
        try:
            slider_rep = self.ui.slider.GetRepresentation()
            slider_rep.SetLabelFormat("%.2e")
        except Exception:
            pass


    def _bind_events(self):
        """Binds keyboard and mouse events."""
        self.plotter.add_key_event('space', lambda: self._toggle_play())
        self.plotter.add_key_event('r', lambda: self.plotter.reset_camera())
        self.plotter.add_key_event('1', lambda: self._set_speed(1))
        self.plotter.add_key_event('2', lambda: self._set_speed(2))
        self.plotter.add_key_event('5', lambda: self._set_speed(5))
        self.plotter.add_key_event('plus', lambda: self._set_speed(self.state.playback_speed + 5))
        self.plotter.add_key_event('minus', lambda: self._set_speed(max(1, self.state.playback_speed - 5)))
        
        self.plotter.enable_point_picking(
            callback=self._on_pick,
            use_mesh=True,
            show_point=False,
            left_clicking=True,
            show_message=False
        )
        
        # Double-click to zoom
        try:
            # Modern pyvista versions
            self.plotter.iren.interactor.add_observer(
                vtk.vtkCommand.LeftButtonDoubleClickEvent, self._on_double_click
            )
        except AttributeError:
            # Older pyvista versions might need a different approach
            print("Warning: Could not bind double-click event. Your VTK/PyVista version might be old.")


    def _on_double_click(self, obj, event):
        """Handle double-click event to zoom on a particle."""
        pos = self.plotter.pick_mouse_position()
        if pos:
            picked_actor, picked_id = self.plotter.pick_closest_point(pos, use_mesh=True)
            if picked_actor is self.particle_actor and picked_id is not None:
                # The picked_id is the point ID in the glyph mesh, not the particle index.
                # We need to find which particle this point belongs to.
                point_id_in_glyph = self.glyph_mesh['vtkPointId'][picked_id]
                self._zoom_to_particle(point_id_in_glyph)

    def _on_pick(self, *args):
        if self.plotter.picked_point is not None and self.plotter.picked_mesh is self.particle_actor:
            point_id = self.plotter.picked_mesh['vtkPointId'][self.plotter.picked_point_id]
            self.state.selected_particle_idx = point_id

            # Update selection marker
            self.selection_marker.points = pv.Sphere(
                radius=self.radii[point_id] * 1.2,
                center=self.positions[self.state.current_frame, point_id]
            ).points
            self.selection_marker.modified()
            self.selection_actor.visibility = True
        else:
            self.state.selected_particle_idx = None
            self.selection_actor.visibility = False

        self._update_info_text()

    def _on_slider_update(self, value):
        time_value = float(value)
        frame = int(np.argmin(np.abs(self.time_steps - time_value)))
        if frame != self.state.current_frame:
            self._render_frame(frame)

    def _toggle_play(self, *args):
        self.state.is_playing = not self.state.is_playing
        if self.state.is_playing:
            if self.state.vtk_timer_id == -1:
                self._init_vtk_timer_fallback()
        else:
            # This part is tricky as stopping the vtk timer is not straightforward
            # from the callback. We'll just let it run but the _on_timer will do nothing.
            pass
        self._update_info_text()

    def _init_vtk_timer_fallback(self):
        """A robust way to create a repeating timer using the underlying VTK interactor."""
        if self.state.vtk_timer_id != -1:
            return 
        
        try:
            iren = self.plotter.iren.interactor
            interval_ms = max(1, int(1000 / (self.state.fps * self.state.playback_speed)))
            self.state.vtk_timer_id = iren.CreateRepeatingTimer(interval_ms)
            iren.AddObserver(vtk.vtkCommand.TimerEvent, self._on_timer)
            print(f"VTK repeating timer started with ID: {self.state.vtk_timer_id}")
        except Exception as e:
            print(f"Failed to create VTK timer: {e}")
            self.state.vtk_timer_id = -1

    def _set_speed(self, speed: int):
        speed = max(1, min(500, speed))
        if self.state.playback_speed == speed:
            return
        self.state.playback_speed = speed
        self._update_help_text()
        if self.state.is_playing and self.plotter.iren is not None:
            try:
                if self.state.vtk_timer_id != -1:
                    self.plotter.iren.interactor.DestroyTimer(self.state.vtk_timer_id)
                    self.state.vtk_timer_id = -1
                self._init_vtk_timer_fallback()
            except Exception:
                pass


    def _on_timer(self, *args):
        """Callback for the timer, advances the frame if playing."""
        if not self.state.is_playing:
            return
        steps = max(1, int(self.state.playback_speed))
        for _ in range(steps):
            self._next_frame(render=False)
        
        # We must manually call render if we are using the VTK timer
        # because it doesn't automatically trigger a render in the same
        # way pyvista's `add_callback` does.
        self.plotter.render()


    def _next_frame(self, render=True):
        current = self.state.current_frame
        n_frames = self.state.n_frames
        new_frame = (current + 1) % n_frames
        if new_frame != current:
            self._render_frame(new_frame, update_slider=True)

    def _prev_frame(self):
        current = self.state.current_frame
        n_frames = self.state.n_frames
        new_frame = (current - 1 + n_frames) % n_frames
        if new_frame != current:
            self._render_frame(new_frame, update_slider=True)

    def _render_frame(self, frame, update_slider=False):
        """Updates the scene to a specific frame."""
        self.state.current_frame = frame
        
        # Update particle positions
        new_positions = self.positions[frame]
        new_speeds = self.speeds[frame]
        
        # Rebuild the glyph mesh. This is more robust than updating points.
        self.glyph_mesh = self._build_glyph_mesh(new_positions, new_speeds)
        array_names = set(self.glyph_mesh.array_names)
        point_ids = None
        for name in ("vtkPointId", "vtkOriginalPointIds", "vtkGlyph3DOriginalPointIds"):
            if name in array_names:
                point_ids = self.glyph_mesh[name]
                break

        if point_ids is not None:
            point_speeds = new_speeds[point_ids]
        else:
            points_per_glyph = int(self.sphere_template.n_points)
            expected_points = self.state.n_particles * points_per_glyph
            if self.glyph_mesh.n_points == expected_points:
                point_speeds = np.repeat(new_speeds, points_per_glyph)
            else:
                # Fallback if point mapping is unavailable.
                point_speeds = np.full(self.glyph_mesh.n_points, float(np.mean(new_speeds)))

        self.glyph_mesh['speed'] = point_speeds
        try:
            self.glyph_mesh.set_active_scalars('speed')
        except AttributeError:
            pass

        updated_with_mapper = False
        try:
            self.particle_actor.mapper.SetInputData(self.glyph_mesh)
            try:
                self.particle_actor.mapper.SetScalarModeToUsePointFieldData()
            except AttributeError:
                self.particle_actor.mapper.SetScalarModeToUsePointData()
            self.particle_actor.mapper.SelectColorArray('speed')
            self.particle_actor.mapper.ScalarVisibilityOn()
            self.particle_actor.mapper.SetScalarRange(self.speed_min, self.speed_max)
            self.particle_actor.mapper.Update()
            updated_with_mapper = True
        except AttributeError:
            pass

        if not updated_with_mapper:
            try:
                self.plotter.update_coordinates(self.glyph_mesh.points, mesh=self.particle_actor)
                try:
                    self.plotter.update_scalars(point_speeds, mesh=self.particle_actor, name='speed')
                except TypeError:
                    self.plotter.update_scalars(point_speeds, mesh=self.particle_actor)
            except AttributeError:
                pass

        # Update selection marker if a particle is selected
        if self.state.selected_particle_idx is not None:
            self.selection_marker.points = pv.Sphere(
                radius=self.radii[self.state.selected_particle_idx] * 1.2,
                center=new_positions[self.state.selected_particle_idx]
            ).points
            self.selection_marker.modified()

        if update_slider:
            target_time = float(self.time_steps[frame])
            try:
                self.ui.slider.SetValue(target_time)
            except AttributeError:
                slider_rep = self.ui.slider.GetRepresentation()
                slider_rep.SetValue(target_time)
            
        self._update_info_text()

    def _update_info_text(self):
        """Updates the information text display."""
        frame = self.state.current_frame
        step = self.sim_steps[frame]
        time = self.time_steps[frame]
        
        info_lines = []
        info_lines.append("--- SIMULATION INFO ---")
        info_lines.append(f"Frame : {frame:>5d} / {self.state.n_frames - 1}")
        info_lines.append(f"Step  : {step:>5d}")
        info_lines.append(f"Time  : {time:.6e} s")
        info_lines.append(f"N     : {self.state.n_particles}")
        info_lines.append("-" * 23)
        
        if self.state.selected_particle_idx is not None:
            idx = self.state.selected_particle_idx
            pos = self.positions[frame, idx]
            vel = self.velocities[frame, idx]
            speed = self.speeds[frame, idx]
            info_lines.append("--- SELECTED PARTICLE ---")
            info_lines.append(f"Index : {idx}")
            info_lines.append(f"Pos   : {pos[0]:.2e}, {pos[1]:.2e}, {pos[2]:.2e}")
            info_lines.append(f"Vel   : {vel[0]:.2e}, {vel[1]:.2e}, {vel[2]:.2e}")
            info_lines.append(f"Speed : {speed:.3f} m/s")
        else:
            info_lines.append("Selected: None")

        info_text = "\n".join(info_lines)
        self._set_text(self.ui.info_text, info_text)

    def _set_text(self, actor, text: str) -> None:
        try:
            actor.SetText(text)
            return
        except TypeError:
            try:
                actor.SetText(2, text)
                return
            except Exception:
                pass
        except AttributeError:
            pass

        try:
            actor.SetInput(text)
        except Exception:
            pass


    def _zoom_to_particle(self, particle_idx):
        if particle_idx < self.state.n_particles:
            print(f"Zooming to particle {particle_idx}")
            center = self.positions[self.state.current_frame, particle_idx]
            radius = self.radii[particle_idx]
            
            # Set camera position to be slightly away from the particle
            cam_pos = center + np.array([radius * 10, radius * 10, radius * 10])
            
            self.plotter.set_camera_position(
                [cam_pos, center, (0, 0, 1)], reset=False
            )
            self.plotter.render()

    def _update_help_text(self):
        """Updates the help text to show current speed."""
        help_str = (
            "Controls:\n"
            "  Play/Pause : Space Bar\n"
            "  Reset View : R\n"
            f"  Speed      : {self.state.playback_speed}x"
        )
        self._set_text(self.ui.help_text, help_str)

    def run(self):
        """Starts the interactive player."""
        print("Starting interactive player...")
        self.plotter.show()
        # Clean up the timer when the window is closed
        try:
            if self.state.vtk_timer_id != -1 and self.plotter.iren is not None:
                self.plotter.iren.interactor.DestroyTimer(self.state.vtk_timer_id)
                print(f"VTK timer {self.state.vtk_timer_id} destroyed.")
        except Exception as e:
            print(f"Could not destroy VTK timer: {e}")


if __name__ == '__main__':
    import argparse
    import os

    DEFAULT_FOLDER = 'data/N10_2.5e-10dt_10e-5_verlet3_h54'
    DEFAULT_FILENAME = 'trajectory.h5'

    parser = argparse.ArgumentParser(description='Interactive 3D Trajectory Player')
    parser.add_argument(
        '--folder',
        type=str,
        default=DEFAULT_FOLDER,
        help=f'Path to the folder containing the HDF5 file. Defaults to {DEFAULT_FOLDER}.'
    )
    parser.add_argument(
        '--file',
        type=str,
        default=DEFAULT_FILENAME,
        help=f'Name of the HDF5 file. Defaults to {DEFAULT_FILENAME}.'
    )
    args = parser.parse_args()

    h5_path = os.path.join(args.folder, args.file)

    if not os.path.exists(h5_path):
        print(f"Error: HDF5 file not found at {h5_path}")
    else:
        player = InteractivePlayer(hdf5_path=h5_path)
        player.run()
