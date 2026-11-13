
import open3d as o3d
from open3d.visualization import gui
from open3d.visualization import rendering

from pathlib import Path
import numpy as np
import subprocess
import threading

PRESETS = {
    "RS 2": {"ckpt":"my_models/rs2_b8_e160.pth", "ds_mode": "rs", "ds_num": "4096"},
    "RS 4": {"ckpt":"my_models/rs4_b8_e160.pth", "ds_mode": "rs", "ds_num": "2048"},
    "RS 8": {"ckpt":"my_models/rs8_b8_e160.pth", "ds_mode": "rs", "ds_num": "1024"},

    "FPS 2": {"ckpt":"my_models/fps2_b8_e160.pth", "ds_mode": "fps", "ds_num": "4096"},
    "FPS 4": {"ckpt":"my_models/fps4_b8_e160.pth", "ds_mode": "fps", "ds_num": "2048"},
    "FPS 8": {"ckpt":"my_models/fps8_b8_e160.pth", "ds_mode": "fps", "ds_num": "1024"},

    "SGWRS 2": {"ckpt":"my_models/sgwrs2_b8_e160.pth", "ds_mode": "sgwrs", "ds_num": "4096"},
    "SGWRS 4": {"ckpt":"my_models/sgwrs4_b8_e160.pth", "ds_mode": "sgwrs", "ds_num": "2048"},
    "SGWRS 8": {"ckpt":"my_models/sgwrs8_b8_e160.pth", "ds_mode": "sgwrs", "ds_num": "1024"},

    "CASNet 2": {"ckpt":"my_models/cn2_b8_e160.pth", "ds_mode": "cn", "ds_num": "4096"},
    "CASNet 4": {"ckpt":"my_models/cn4_b8_e160.pth", "ds_mode": "cn", "ds_num": "2048"},
    "CASNet 8": {"ckpt":"my_models/cn8_b8_e160.pth", "ds_mode": "cn", "ds_num": "1024"},
}

INPUTS = [
    ("data_root", "KITTI root path:", "kitti_data"),
    ("ckpt", "Checkpoint path:", "pretrained_legacy/epoch_160.pth"),
    ("batch_size", "Batch size:", "8"),
    ("num_workers", "DataLoader workers:", "4"),
    ("ds_mode", "Subsampling mode (none, rs, fps, cn, sgwrs):", "none"),
    ("ds_num", "Number of retained points:", "1024"),
    ("no_cuda", "Disable CUDA (yes/no):", "no"),
]

CONTAINER_NAME = "sgwrs"
WORKDIR = "/home/appuser/sgwrs"

PROJECT_ROOT = Path(__file__).resolve().parent
ORIGINAL_DIR = (PROJECT_ROOT / "sgwrs_ds/kitti_data/training/velodyne_cropped").resolve()
OUTPUT_PATH = "_results_tmp_gui/tmp_gui_vis"
DS_DIR = (PROJECT_ROOT / "sgwrs" / OUTPUT_PATH / "ds_pcd").resolve()

def on_run_clicked(params, window, run_button, status, state, update_clouds):
    try:
        batch_size = int(params["batch_size"])
        num_workers = int(params["num_workers"])
        ds_num = int(params["ds_num"])
        if batch_size <= 0 or num_workers < 0 or ds_num <= 0 : raise ValueError
    except ValueError:
        status.text = "Error: Invalid numeric parameter"
        return

    ds_mode = params["ds_mode"].lower()
    if ds_mode not in ("none", "rs", "fps", "cn", "sgwrs"):
        status.text = "Error: Invalid subsampling mode"
        return

    no_cuda = params["no_cuda"].lower()
    if no_cuda not in ("yes", "no"):
        status.text = "Error: Use yes or no"
        return

    if not params["data_root"] or not params["ckpt"]:
        status.text = "Error: Missing path"
        return

    command = [
        "docker", "exec",
        "-w", WORKDIR,
        CONTAINER_NAME,
        "python", "-u", "evaluate.py",
        "--data_root", params["data_root"],
        "--ckpt", params["ckpt"],
        "--saved_path", OUTPUT_PATH,
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
        "--ds_mode", ds_mode,
        "--ds_num", str(ds_num),
    ]

    if no_cuda == "yes": command.append("--no_cuda")

    command.append("--save_ds_pcd")

    print("Starting evaluation...", flush=True)
    print("Command:", command, flush=True)

    state["running"] = True

    run_button.enabled = False
    status.text = "Evaluation running..."

    def clear_old_pointclouds():
        expected = (PROJECT_ROOT / "sgwrs" / "_results_tmp_gui" / "tmp_gui_vis" / "ds_pcd").absolute()
        if DS_DIR != expected: 
            raise RuntimeError("Unexpected point cloud directory")
        DS_DIR.mkdir(parents=True, exist_ok=True)
        for file in DS_DIR.glob("*.bin"):
            if file.is_file() and not file.is_symlink():
                file.unlink()

    def watch_pointclouds(stop_event, on_new_cloud):
        last_file = None
        while True:
            files = [
                p for p in DS_DIR.glob("*.bin")
                if p.stem.isdigit() and len(p.stem) == 6
            ]
            if files:
                latest = max(files, key=lambda p: p.stat().st_mtime_ns)
                original_path = ORIGINAL_DIR / latest.name
                if latest.name != last_file and original_path.is_file():
                    try:
                        original = np.fromfile(original_path, dtype=np.float32).reshape(-1, 4)
                        downsampled = np.fromfile(latest, dtype=np.float32).reshape(-1, 4)
                        if len(original) > 0 and len(downsampled) > 0:
                            on_new_cloud(
                                original[:, :3].copy(),
                                downsampled[:, :3].copy(),
                                latest.stem
                            )
                            last_file = latest.name
                    except (OSError, ValueError) as error:
                        print(f"Point cloud read error: {error}")
            if stop_event.wait(0.05):
                break

    def worker():
        stop_event = threading.Event()
        watcher = None

        try:
            clear_old_pointclouds()

            def on_new_cloud(original, downsampled, frame_id):
                gui.Application.instance.post_to_main_thread(
                    window,
                    lambda: update_clouds(original, downsampled, frame_id)
                )

            watcher = threading.Thread(
                target=watch_pointclouds,
                args=(stop_event, on_new_cloud),
                daemon=True
            )
            watcher.start()

            process = subprocess.Popen(command)
            return_code = process.wait()

            if return_code == 0: message = "Evaluation completed successfully"
            else: message = f"Evaluation failed (exit code: {return_code})"

        except OSError as error: message = f"Failed to start: {error}"
        except Exception as error: message = f"Error: {error}"

        finally:
            stop_event.set()
            if watcher is not None:
                watcher.join(timeout=2)

        print(message, flush=True)

        def finish():
            state["running"] = False
            status.text = message
            run_button.enabled = True

        gui.Application.instance.post_to_main_thread(window, finish)

    threading.Thread(target=worker, daemon=True).start()

def main():
    app = gui.Application.instance
    app.initialize()

    window = app.create_window("SGWRS | Evaluate", 1600, 950) # Name, W, H
    
    em = window.theme.font_size # Font Scale Unit

    state = {
        "running": False,
        "live": True,
        "current_frame": None
    }

    # Background

    BG = gui.Color(0.18, 0.18, 0.18, 1.0)

    strips = []
    for _ in range(7):
        strip = gui.Vert()
        strip.background_color = BG
        window.add_child(strip)
        strips.append(strip)

    L_layout = gui.Vert(0.8 * em, gui.Margins(0, 0, 0, 0)) # Space, Margins(L,T,R,B)

    L_layout.background_color = BG

    # = LEFT LAYOUT ======

    # PRESETS

    center_label = gui.Horiz()
    center_label.add_stretch()
    center_label.add_child(gui.Label("PRESETS"))
    center_label.add_stretch()
    
    L_layout.add_child(center_label)
    
    preset_grid = gui.VGrid(3, 0.2 * em) # Colums, Space

    def apply_preset(name):
        preset = PRESETS[name]
        for key, value in preset.items():
            args_fields[key].text_value = value

    for name in PRESETS:
        button = gui.Button(name)
        button.horizontal_padding_em = 1.5 # W
        button.set_on_clicked(lambda name=name: apply_preset(name))
        preset_grid.add_child(button)

    L_layout.add_child(preset_grid)

    # INPUTS

    args_fields = {}

    def create_input(label_text, placeholder):
        input_layout = gui.Vert(0.1 * em) # Space
        label = gui.Label(label_text)
        input_layout.add_child(label)
        text_input = gui.TextEdit()
        text_input.text_value = placeholder
        input_layout.add_child(text_input)
        return input_layout, text_input

    for key, label_text, default in INPUTS:
        input_layout, text_input = create_input(label_text, default)
        L_layout.add_child(input_layout)
        args_fields[key] = text_input

    L_layout.add_stretch()

    # RUN

    def container_is_running():
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
            capture_output=True,
            text=True
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def run():
        if state["running"]: return
        if not container_is_running():
            window.show_message_box(
                "Docker warning",
                "The 'sgwrs' container is not running!\n\n"
                "Please start the container before running evaluation."
            )
            return
        state["live"] = True
        state["current_frame"] = None
        params = {
            key: field.text_value.strip()
            for key, field in args_fields.items()
        }
        on_run_clicked(params, window, run_button, status, state, update_clouds)

    run_button = gui.Button("Run")
    run_button.vertical_padding_em = 0.5 # H
    run_button.set_on_clicked(run)

    L_layout.add_child(run_button)


    # SYNC

    def sync_views():
        ds_viewer.scene.camera.copy_from(
            original_viewer.scene.camera
        )
        ds_viewer.force_redraw()

    sync_button = gui.Button("Sync")
    sync_button.set_on_clicked(sync_views)

    L_layout.add_child(sync_button)

    # CONTROL

    control_layout = gui.Horiz(0.7 * em, gui.Margins(0, 0, 0, 0)) # Space, Margins(L,T,R,B)

    def stop_live():
        state["live"] = False

    stop_button = gui.Button("Stop")
    stop_button.set_on_clicked(stop_live)
    stop_button.horizontal_padding_em = 1.3 # W

    def resume_live():
        state["live"] = True
        frames = get_available_frames()
        if frames: show_frame(frames[-1])

    resume_button = gui.Button("Live")
    resume_button.set_on_clicked(resume_live)
    resume_button.horizontal_padding_em = 1.4 # W

    def step_frame(direction):
        state["live"] = False
        frames = get_available_frames()
        if not frames: return
        current = state["current_frame"]
        if current not in frames: target = frames[-1]
        else:
            index = frames.index(current)
            new_index = max(0, min(len(frames) - 1, index + direction))
            target = frames[new_index]
        show_frame(target)

    prev_button = gui.Button("Prev")
    next_button = gui.Button("Next")
    prev_button.set_on_clicked(lambda: step_frame(-1))
    next_button.set_on_clicked(lambda: step_frame(1))
    prev_button.horizontal_padding_em = 1.3 # W
    next_button.horizontal_padding_em = 1.3 # W

    control_layout.add_child(stop_button)
    control_layout.add_child(prev_button)
    control_layout.add_child(next_button)
    control_layout.add_child(resume_button)

    L_layout.add_child(control_layout)

    frame_label = gui.Label("Frame: -")
    L_layout.add_child(frame_label)

    # LOG

    status = gui.Label("Ready")

    L_layout.add_child(status)

    # ADD TO ROOT

    window.add_child(L_layout)

    # =====================

    # = RIGHT LAYOUT ======

    def load_point_cloud(path):
        path = Path(path)
        if not path.is_file(): raise FileNotFoundError(f"Point cloud not found: {path}")
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        if pcd.is_empty(): raise ValueError(f"Empty point cloud: {path}")
        return pcd

    def create_pointcloud_viewer():
        viewer = gui.SceneWidget()
        viewer.scene = rendering.Open3DScene(window.renderer)
        viewer.setup_camera(
            60.0,
            o3d.geometry.AxisAlignedBoundingBox(
                [-10, -10, -10],
                [10, 10, 10]
            ),
            [0, 0, 0]
        )
        camera = viewer.scene.camera
        view = camera.get_view_matrix()
        inv_view = np.linalg.inv(view)
        eye = inv_view[:3, 3]
        right = inv_view[:3, 0]
        up = inv_view[:3, 1]
        horizontal_offset = 30.0 
        zoom_out = 50.0 
        offset = right * horizontal_offset
        forward = -inv_view[:3, 2]
        viewer.look_at(
            eye + offset + forward * 10.0,
            eye + offset - forward * zoom_out,
            up
        )
        return viewer

    def update_clouds(original_xyz, ds_xyz, frame_id, force=False):
        if not state["live"] and not force: return
        state["current_frame"] = frame_id
        frame_label.text = f"Frame: {frame_id}"
        def update_viewer(viewer, xyz):
            viewer.scene.remove_geometry("points")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            material.point_size = 2.5 * window.scaling
            material.base_color = [0.4, 0.4, 0.4, 1.0]
            viewer.scene.add_geometry("points", pcd, material)
            viewer.force_redraw()
        update_viewer(original_viewer, original_xyz)
        update_viewer(ds_viewer, ds_xyz)

    def get_available_frames():
        frames = []
        for file in DS_DIR.glob("*.bin"):
            if not (file.stem.isdigit() and len(file.stem) == 6): continue
            original_path = ORIGINAL_DIR / file.name
            if original_path.is_file(): frames.append(file.stem)
        return sorted(frames)

    def show_frame(frame_id):
        original_path = ORIGINAL_DIR / f"{frame_id}.bin"
        ds_path = DS_DIR / f"{frame_id}.bin"
        try:
            original = np.fromfile(
                original_path,
                dtype=np.float32
            ).reshape(-1, 4)
            downsampled = np.fromfile(
                ds_path,
                dtype=np.float32
            ).reshape(-1, 4)
            if len(original) == 0 or len(downsampled) == 0: return
            update_clouds(
                original[:, :3].copy(),
                downsampled[:, :3].copy(),
                frame_id,
                force=True
            )
        except (OSError, ValueError) as error:
            print(f"Failed to load frame {frame_id}: {error}")

    # ORIGINAL PCD

    original_label = gui.Label("Original point cloud")
    original_viewer = create_pointcloud_viewer()

    original_label.background_color = BG

    window.add_child(original_label)
    window.add_child(original_viewer)

    # DOWNSAMPLED PCD

    ds_label = gui.Label("Downsampled point cloud")
    ds_viewer = create_pointcloud_viewer()

    ds_label.background_color = BG

    window.add_child(ds_label)
    window.add_child(ds_viewer)

    # =====================

    def on_layout(ctx):
        r = window.content_rect

        margin = 1.5 * em
        spacing = 1.5 * em

        left_width = 21.25 * em

        # LEFT PANEL
        L_layout.frame = gui.Rect(
            r.x + margin,
            r.y + margin,
            left_width,
            r.height - 2 * margin
        )

        # RIGHT PANEL
        right_x = r.x + margin + left_width + spacing
        right_width = max(1, r.get_right() - margin - right_x)

        label_height = 1.5 * em
        available_height = r.height - 2 * margin
        viewer_height = max(1, (available_height - 2 * label_height - spacing) / 2)

        # ORIGINAL
        original_label.frame = gui.Rect(
            right_x,
            r.y + margin,
            right_width,
            label_height
        )

        original_viewer.frame = gui.Rect(
            right_x,
            r.y + margin + label_height,
            right_width,
            viewer_height
        )

        # DOWNSAMPLED
        ds_y = r.y + margin + label_height + viewer_height + spacing

        ds_label.frame = gui.Rect(
            right_x,
            ds_y,
            right_width,
            label_height
        )

        ds_viewer.frame = gui.Rect(
            right_x,
            ds_y + label_height,
            right_width,
            viewer_height
        )

        # BG STRIPS
        R = gui.Rect
        rects = [
            R(r.x, r.y, r.width, margin),                                                                   # top
            R(r.x, r.get_bottom() - margin, r.width, margin),                                               # bottom
            R(r.x, r.y, margin, r.height),                                                                  # left edge
            R(r.get_right() - margin, r.y, margin, r.height),                                               # right edge
            R(r.x + margin + left_width, r.y + margin, spacing, r.height - 2 * margin),                     # panel | viewers gap
            R(right_x, r.y + margin + label_height + viewer_height, right_width, spacing + label_height),   # viewer 1 | viewer 2
        ]
        for strip, rect in zip(strips, rects):
            strip.frame = rect

    window.set_on_layout(on_layout)

    def on_close():
        if state["running"]:
            window.show_message_box(
                "Evaluation is running",
                "Please wait until the evaluation finishes."
            )
            return False
        return True

    window.set_on_close(on_close)
    app.run()

if __name__ == "__main__":
    main()
