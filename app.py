"""Desktop entry point for the PTRDMS panorama-processing workflow.

Run with: python app.py
"""
from __future__ import annotations

import queue
import math
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from tkinter import END, BooleanVar, StringVar, Text, Tk
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parent


def build_command(
    python: str,
    input_dir: str,
    map_path: str,
    output_dir: str,
    detector_weights: str,
    classifier_weights: str,
    confidence: str,
    max_frames: str,
    pose_csv: str = "",
    range_manifest: str = "",
    export_six_faces: bool = True,
    classifier_imgsz: str = "",
    camera_yaw_offset_deg: str = "0",
    pose_match_mode: str = "filename",
    frame_times_csv: str = "",
    pose_timestamp_tolerance_s: str = ".05",
) -> list[str]:
    """Create a transparent command line for a GUI processing run."""
    command = [
        python,
        str(ROOT / "demo.py"),
        "--input",
        input_dir,
        "--map",
        map_path,
        "--output",
        output_dir,
        "--detector",
        "two-stage",
        "--detector-weights",
        detector_weights,
        "--classifier-weights",
        classifier_weights,
        "--confidence",
        confidence,
        "--max-frames",
        max_frames,
    ]
    if pose_csv:
        command.extend(["--pose-csv", pose_csv])
    if pose_match_mode == "timestamp":
        command.extend([
            "--pose-match-mode", "timestamp",
            "--frame-times-csv", frame_times_csv,
            "--pose-timestamp-tolerance-s", pose_timestamp_tolerance_s,
        ])
    if range_manifest:
        command.extend(["--range-manifest", range_manifest])
    if export_six_faces:
        command.append("--export-six-faces")
    if classifier_imgsz:
        command.extend(["--classifier-imgsz", classifier_imgsz])
    if camera_yaw_offset_deg.strip():
        command.extend(["--camera-yaw-offset-deg", camera_yaw_offset_deg])
    return command


class PrmsApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("PTRDMS Panoramic Tomato Ripeness Detection and Mapping")
        self.root.minsize(780, 610)
        self.messages: queue.Queue[str] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.last_result: Path | None = None
        self.busy = False
        self.pending_result: Path | None = None

        self.input_dir = StringVar()
        self.map_path = StringVar()
        self.output_dir = StringVar(value=str(ROOT / "outputs" / "desktop_run"))
        self.pose_csv = StringVar()
        self.pose_match_mode = StringVar(value="filename")
        self.frame_times_csv = StringVar()
        self.pose_timestamp_tolerance_s = StringVar(value="0.05")
        self.range_manifest = StringVar()
        self.camera_yaw_offset_deg = StringVar(value="0")
        self.detector_weights = StringVar(value=str(ROOT / "models" / "tomato_detector.pt"))
        self.classifier_weights = StringVar(value=str(ROOT / "models" / "tomato_ripeness_classifier.pt"))
        self.confidence = StringVar(value="0.25")
        self.max_frames = StringVar(value="0")
        self.classifier_imgsz = StringVar(value="")
        self.export_six_faces = BooleanVar(value=True)
        self.status = StringVar(value="Ready. The bundled classifier uses the paper's four-stage maturity taxonomy.")

        frame = ttk.Frame(root, padding=14)
        frame.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        self._path_row(frame, 0, "Panorama folder", self.input_dir, self._choose_directory)
        self._path_row(frame, 1, "Map image", self.map_path, self._choose_file)
        self._path_row(frame, 2, "Output folder", self.output_dir, self._choose_output)
        self._path_row(frame, 3, "Camera pose CSV (optional)", self.pose_csv, self._choose_file)
        self._path_row(frame, 4, "Frame timestamps CSV (timestamp mode)", self.frame_times_csv, self._choose_file)
        ttk.Label(frame, text="Pose binding mode").grid(row=5, column=0, sticky="w", pady=3)
        ttk.Combobox(frame, textvariable=self.pose_match_mode, values=("filename", "timestamp"), state="readonly").grid(row=5, column=1, sticky="ew", padx=(10, 8), pady=3)
        ttk.Label(frame, text="Timestamp tolerance (s)").grid(row=6, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.pose_timestamp_tolerance_s).grid(row=6, column=1, sticky="ew", padx=(10, 8), pady=3)
        self._path_row(frame, 7, "Range manifest (optional)", self.range_manifest, self._choose_file)
        ttk.Label(frame, text="Camera yaw offset (deg)").grid(row=8, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.camera_yaw_offset_deg).grid(row=8, column=1, sticky="ew", padx=(10, 8), pady=3)
        self._path_row(frame, 9, "Tomato detector weights", self.detector_weights, self._choose_file)
        self._path_row(frame, 10, "Ripeness classifier weights", self.classifier_weights, self._choose_file)

        options = ttk.Frame(frame)
        options.grid(row=11, column=0, columnspan=3, sticky="ew", pady=(8, 8))
        ttk.Label(options, text="Confidence").grid(row=0, column=0, sticky="w")
        ttk.Entry(options, textvariable=self.confidence, width=8).grid(row=0, column=1, padx=(6, 18))
        ttk.Label(options, text="Frames (0 = all)").grid(row=0, column=2, sticky="w")
        ttk.Entry(options, textvariable=self.max_frames, width=8).grid(row=0, column=3, padx=(6, 18))
        ttk.Checkbutton(options, text="Export all six cube faces", variable=self.export_six_faces).grid(row=0, column=4, sticky="w")
        ttk.Label(options, text="Classifier size (blank = model)").grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Entry(options, textvariable=self.classifier_imgsz, width=8).grid(row=1, column=2, sticky="w")

        actions = ttk.Frame(frame)
        actions.grid(row=12, column=0, columnspan=3, sticky="ew")
        self.run_button = ttk.Button(actions, text="Run detection and mapping", command=self.run_inference)
        self.run_button.grid(row=0, column=0, padx=(0, 8))
        ttk.Button(actions, text="Replay included results", command=self.replay).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(actions, text="Verify included data", command=self.verify).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(actions, text="Open latest result", command=self.open_result).grid(row=0, column=3)

        ttk.Label(frame, textvariable=self.status).grid(row=13, column=0, columnspan=3, sticky="w", pady=(12, 4))
        self.log = Text(frame, height=17, wrap="word", state="disabled")
        self.log.grid(row=14, column=0, columnspan=3, sticky="nsew")
        frame.rowconfigure(14, weight=1)
        self.root.after(100, self._drain_messages)

    def _path_row(self, parent: ttk.Frame, row: int, label: str, variable: StringVar, chooser) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=(10, 8), pady=3)
        ttk.Button(parent, text="Browse", command=lambda: chooser(variable)).grid(row=row, column=2, pady=3)

    def _choose_directory(self, variable: StringVar) -> None:
        value = filedialog.askdirectory(initialdir=variable.get() or str(ROOT))
        if value:
            variable.set(value)

    def _choose_output(self, variable: StringVar) -> None:
        value = filedialog.askdirectory(initialdir=str(ROOT / "outputs"), title="Choose output folder")
        if value:
            variable.set(value)

    def _choose_file(self, variable: StringVar) -> None:
        value = filedialog.askopenfilename(initialdir=str(ROOT), title="Choose file")
        if value:
            variable.set(value)

    def _append(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(END, text)
        self.log.see(END)
        self.log.configure(state="disabled")

    def _validate_number(self) -> bool:
        try:
            confidence = float(self.confidence.get())
            frames = int(self.max_frames.get())
            classifier_size = int(self.classifier_imgsz.get()) if self.classifier_imgsz.get().strip() else 224
            yaw_offset = float(self.camera_yaw_offset_deg.get())
            timestamp_tolerance = float(self.pose_timestamp_tolerance_s.get())
        except ValueError:
            messagebox.showerror("Invalid options", "Confidence must be a number; frames and classifier size must be integers.")
            return False
        if (not 0.0 <= confidence <= 1.0 or frames < 0 or classifier_size < 32
                or not -360 <= yaw_offset <= 360 or not math.isfinite(timestamp_tolerance) or timestamp_tolerance <= 0):
            messagebox.showerror("Invalid options", "Confidence must be 0–1, frames nonnegative, classifier size at least 32, yaw offset within -360 to 360 degrees, and timestamp tolerance positive.")
            return False
        return True

    def _ensure_paths(self, required: tuple[StringVar, ...]) -> bool:
        missing = [value.get() for value in required if not Path(value.get()).exists()]
        if missing:
            messagebox.showerror("Missing input", "The following path does not exist:\n" + missing[0])
            return False
        if self.range_manifest.get() and not self.pose_csv.get():
            messagebox.showerror("Range data requires poses", "Provide a camera pose CSV when using a range manifest.")
            return False
        if self.pose_match_mode.get() == "timestamp":
            if not self.pose_csv.get() or not self.frame_times_csv.get():
                messagebox.showerror("Timestamp binding requires inputs", "Provide both a pose CSV and a frame timestamps CSV.")
                return False
            if not Path(self.pose_csv.get()).exists() or not Path(self.frame_times_csv.get()).exists():
                messagebox.showerror("Missing timestamp input", "The pose CSV or frame timestamps CSV does not exist.")
                return False
        return True

    def run_inference(self) -> None:
        if self.busy:
            return
        if not self._validate_number() or not self._ensure_paths((self.input_dir, self.map_path, self.detector_weights, self.classifier_weights)):
            return
        command = build_command(
            sys.executable, self.input_dir.get(), self.map_path.get(), self.output_dir.get(),
            self.detector_weights.get(), self.classifier_weights.get(), self.confidence.get(), self.max_frames.get(),
            self.pose_csv.get(), self.range_manifest.get(), self.export_six_faces.get(),
            self.classifier_imgsz.get().strip(), self.camera_yaw_offset_deg.get().strip(),
            self.pose_match_mode.get(), self.frame_times_csv.get(), self.pose_timestamp_tolerance_s.get().strip(),
        )
        self.pending_result = Path(self.output_dir.get()) / "index.html"
        self._start(command, "Running two-stage detection and spatial mapping…")

    def replay(self) -> None:
        if self.busy:
            return
        output = ROOT / "outputs" / "desktop_replay"
        self.pending_result = output / "index.html"
        self._start([sys.executable, str(ROOT / "scripts" / "replay_observations.py"), "--source", str(ROOT / "demo"), "--output", str(output)], "Replaying the included saved observations…")

    def verify(self) -> None:
        if self.busy:
            return
        self.pending_result = None
        self._start([sys.executable, str(ROOT / "scripts" / "verify_dataset.py")], "Checking included file hashes…")

    def _start(self, command: list[str], status: str) -> None:
        if self.busy:
            return
        self.busy = True
        self.status.set(status)
        self.run_button.configure(state="disabled")
        self._append("\n$ " + subprocess.list2cmdline(command) + "\n")
        threading.Thread(target=self._run_process, args=(command,), daemon=True).start()

    def _run_process(self, command: list[str]) -> None:
        try:
            self.process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.messages.put(line)
            code = self.process.wait()
            self.messages.put(("__DONE__", code))
        except OSError as exc:
            self.messages.put(("__ERROR__", str(exc)))

    def _drain_messages(self) -> None:
        try:
            while True:
                item = self.messages.get_nowait()
                if isinstance(item, tuple):
                    kind, payload = item
                    self.process = None
                    self.busy = False
                    self.run_button.configure(state="normal")
                    if kind == "__DONE__":
                        if payload == 0 and self.pending_result and self.pending_result.is_file():
                            self.last_result = self.pending_result
                        self.status.set("Completed successfully." if payload == 0 else f"Stopped with exit code {payload}.")
                    else:
                        self.status.set("Could not start the command.")
                        self._append(f"ERROR: {payload}\n")
                    self.pending_result = None
                else:
                    self._append(item)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_messages)

    def open_result(self) -> None:
        candidates = [self.last_result, ROOT / "outputs" / "reviewer_fixed_final" / "index.html",
                      ROOT / "outputs" / "desktop_replay" / "index.html", ROOT / "demo" / "index.html"]
        for candidate in candidates:
            if candidate and candidate.exists():
                webbrowser.open(candidate.resolve().as_uri())
                self.status.set(f"Opened {candidate}")
                return
        messagebox.showinfo("No result yet", "Run detection or replay the included results first.")


def main() -> None:
    root = Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    PrmsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
