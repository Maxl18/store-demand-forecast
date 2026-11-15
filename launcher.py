

import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

# - project folder = this file's folder
ROOT = Path(__file__).resolve().parent
NOTEBOOK = "store_demand_forecast.ipynb"
HTML_NAME = "store_demand_forecast.html"
LOG_FILE = ROOT / "results" / "final_test_log.json"
KAGGLE_URL = "https://www.kaggle.com/c/demand-forecasting-kernels-only"

NOTEBOOK_WARNING = (
    "This re-runs the whole frozen pipeline from start to finish, including the "
    "test period (the last quarter of 2017).\n\n"
    "- It takes under a minute.\n"
    "- It rewrites the notebook file and the files in the results folder.\n"
    "- It adds an entry to the run log (results\\final_test_log.json). "
    "Do not delete or edit that log.\n"
    "- The guard may refuse to run, or report a mismatch, if your train.csv or "
    "your library versions differ from the original run.\n\n"
    "Run it now?"
)
DRY_RUN_NOTE = "DRY RUN MODE (test hook LAUNCHER_DRY_RUN=1): the notebook button only prints 'dry run'."

# - keep the window fast on long output
MAX_MESSAGES_PER_TICK = 300   # - lines per 50 ms tick
MAX_OUTPUT_LINES = 3000       # - older lines dropped


# - functions up to self_test() need no window (--self-test)
def venv_python():
    """.venv Python path, or None."""
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python"):
        candidate = ROOT / rel
        if candidate.is_file():
            return candidate
    return None


def python_to_use():
    """(python path, note); falls back to the current Python."""
    venv = venv_python()
    if venv is not None:
        return str(venv), ""
    return sys.executable, "NOTE: .venv was not found, so the Python that started this window is used."


def dry_run_on():
    """True if test hook LAUNCHER_DRY_RUN=1."""
    return os.environ.get("LAUNCHER_DRY_RUN") == "1"


def notebook_dialog_text():
    """Confirm-dialog text (dry-run warning if on)."""
    if dry_run_on():
        return DRY_RUN_NOTE + " Nothing real will run.\n\n" + NOTEBOOK_WARNING
    return NOTEBOOK_WARNING


def kill_process_tree(proc):
    """Stop a command and its children (e.g. the kernel).

    - taskkill /T /F first: terminate() ends only the top process on Windows
    """
    try:
        done = subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                              capture_output=True,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if done.returncode == 0:
            return
    except OSError:
        pass
    try:
        proc.terminate()
    except OSError:
        pass


def command_checks():
    """The two validation-only test scripts."""
    py, _ = python_to_use()
    return [[py, str(Path("src") / "test_leakage.py")],
            [py, str(Path("src") / "test_simulation.py")]]


def command_notebook():
    """Notebook run (dry-run command if the hook is on)."""
    py, _ = python_to_use()
    if dry_run_on():
        return [[py, "-c", "print('dry run')"]]
    return [[py, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
             "--inplace", NOTEBOOK]]


def command_export_html():
    """Notebook to HTML, no re-run."""
    py, _ = python_to_use()
    return [[py, "-m", "jupyter", "nbconvert", "--to", "html", NOTEBOOK,
             "--output", HTML_NAME]]


def read_run_log_summary():
    """One-line run log summary (plain json, no guard)."""
    if not LOG_FILE.is_file():
        return "no run log yet (the final test has not been run here)"
    try:
        data = json.loads(LOG_FILE.read_text(encoding="utf-8"))
        entries = data.get("entries", [])
        if not isinstance(entries, list):  # - report odd shapes, never crash
            raise ValueError("entries is not a list")
        entries = [e for e in entries if isinstance(e, dict)]
    except (OSError, ValueError, AttributeError, TypeError):
        return "run log exists but could not be read (do not edit it)"
    started = sum(1 for e in entries if e.get("event") == "started")
    finished = sum(1 for e in entries if e.get("event") == "finished")
    other = len(entries) - started - finished
    text = f"{started} run(s) logged, {finished} finished, {len(entries)} log entries"
    if other:
        text += f" ({other} refused, failed or other)"
    return text


def get_status():
    """Status rows (label, ok, text)."""
    train = ROOT / "train.csv"
    if train.is_file():
        mb = train.stat().st_size / 1_000_000
        train_row = ("train.csv", True, f"found ({mb:.1f} MB)")
    else:
        train_row = ("train.csv", False,
                     "MISSING. Download it from " + KAGGLE_URL
                     + " (Data tab) and put it in the project folder.")
    venv = venv_python()
    venv_row = (".venv", venv is not None,
                "found" if venv else "missing (run.bat creates it)")
    nb_row = ("Notebook", (ROOT / NOTEBOOK).is_file(),
              "found" if (ROOT / NOTEBOOK).is_file() else "MISSING: " + NOTEBOOK)
    log_text = read_run_log_summary()
    log_row = ("Run log", LOG_FILE.is_file(), log_text)
    return [train_row, venv_row, nb_row, log_row]


def snapshot_results():
    """SHA-256 of every file in results\\."""
    shots = {}
    base = ROOT / "results"
    if base.is_dir():
        for path in sorted(base.rglob("*")):
            if path.is_file():
                shots[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return shots


def describe_results_change(before, after):
    """Did results change during the command?"""
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    if not changed:
        return "results folder: unchanged (all files identical to before)."
    shown = ", ".join(changed[:8]) + (" ..." if len(changed) > 8 else "")
    return f"results folder: {len(changed)} file(s) CHANGED: {shown}"


def open_with_default_app(path):
    """Open with the Windows default program."""
    opener = getattr(os, "startfile", None)
    if opener is None:
        raise OSError("opening files this way only works on Windows")
    opener(str(path))


def self_test():
    """No window: print status and button commands; returns 0."""
    print("launcher self-test")
    print("project folder:", ROOT.name, "(found next to launcher.py)")
    py, note = python_to_use()
    print("python to use :", "the .venv Python" if note == "" else "fallback (current Python)")
    if note:
        print(note)
    print()
    print("Status panel:")
    for label, ok, text in get_status():
        print(f"  [{'ok' if ok else '!!'}] {label}: {text}")
    if dry_run_on():
        print("  [!!] " + DRY_RUN_NOTE)
    print()
    print("Commands the buttons would run (all start in the project folder):")
    print("  Run checks, step 1 :", command_checks()[0])
    print("  Run checks, step 2 :", command_checks()[1])
    print("  Run the notebook   :", command_notebook()[0])
    print("  Export to HTML     :", command_export_html()[0])
    if dry_run_on():
        print("  (LAUNCHER_DRY_RUN=1 is set, so the notebook button is harmless)")
    print()
    print("self-test finished: nothing was executed.")
    return 0


class LauncherApp:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import scrolledtext, ttk
        self.tk = tk
        self.root = root
        root.title("Store demand forecast launcher")
        root.geometry("820x640")
        root.minsize(640, 480)

        self.busy = False              # - one command at a time
        self.proc = None               # - running subprocess
        self.closing = False           # - window closed mid-run
        self.lock = threading.Lock()   # - start step vs close: no race
        self.messages = queue.Queue()  # - worker -> window (tkinter not thread-safe)
        self.buttons = []
        self.status_labels = []

        # - status panel
        status_box = ttk.LabelFrame(root, text="Status", padding=8)
        status_box.pack(fill="x", padx=10, pady=(10, 4))
        for _ in range(4):
            label = ttk.Label(status_box, text="", wraplength=760, justify="left")
            label.pack(anchor="w")
            self.status_labels.append(label)
        # - shown only with the test hook
        self.mode_label = ttk.Label(status_box, text=DRY_RUN_NOTE, foreground="#b00020",
                                    wraplength=760, justify="left")

        # - buttons
        button_box = ttk.Frame(root, padding=(10, 4))
        button_box.pack(fill="x")
        self.add_button(button_box, "Run checks (leakage and simulation tests)", self.on_checks, 0, 0, 2)
        self.add_button(button_box, "Run the notebook", self.on_notebook, 0, 2, 1)
        self.add_button(button_box, "Open notebook", self.on_open_notebook, 1, 0, 1)
        self.add_button(button_box, "Open README", self.on_open_readme, 1, 1, 1)
        self.add_button(button_box, "Open results folder", self.on_open_results, 1, 2, 1)
        self.add_button(button_box, "Export notebook to HTML and open it", self.on_export_html, 2, 0, 3)
        for col in range(3):
            button_box.columnconfigure(col, weight=1)

        # - output box
        self.output = scrolledtext.ScrolledText(root, height=20, wrap="char", state="disabled",
                                                font=("Consolas", 9))
        self.output.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        self.output.tag_config("good", foreground="#0a7a2f")
        self.output.tag_config("bad", foreground="#b00020")
        self.output.tag_config("note", foreground="#555555")

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.refresh_status()
        self.say("Ready. Pick a button above. Nothing runs until you click.\n", "note")
        root.after(50, self.poll)

    def add_button(self, parent, text, command, row, col, span):
        from tkinter import ttk
        button = ttk.Button(parent, text=text, command=command)
        button.grid(row=row, column=col, columnspan=span, sticky="ew", padx=3, pady=3)
        self.buttons.append(button)

    def say(self, text, tag=None):
        """Append text to the output box, scroll to end."""
        self.output.configure(state="normal")
        self.output.insert("end", text, tag or ())
        # - drop oldest lines: keeps the window fast
        lines = int(self.output.index("end-1c").split(".")[0])
        if lines > MAX_OUTPUT_LINES:
            self.output.delete("1.0", f"{lines - MAX_OUTPUT_LINES + 1}.0")
        self.output.see("end")
        self.output.configure(state="disabled")

    def refresh_status(self):
        for label, (name, ok, text) in zip(self.status_labels, get_status()):
            label.configure(text=f"{'OK     ' if ok else 'PROBLEM'}  {name}: {text}",
                            foreground="#0a7a2f" if ok else "#b00020")
        if dry_run_on():
            self.mode_label.pack(anchor="w")
        else:
            self.mode_label.pack_forget()

    def set_busy(self, busy):
        self.busy = busy
        for button in self.buttons:
            button.state(["disabled"] if busy else ["!disabled"])

    def run_job(self, title, steps, on_success=None):
        """Run `steps` in order without freezing the window."""
        if self.busy:  # - never two at once
            return
        self.set_busy(True)
        _, note = python_to_use()
        self.say(f"\n=== {title} ===\n", "note")
        if note:
            self.say(note + "\n", "bad")
        if dry_run_on():
            self.say("(test hook LAUNCHER_DRY_RUN=1 is on)\n", "note")
        self.before = snapshot_results()
        self.on_success = on_success
        threading.Thread(target=self.worker, args=(steps,), daemon=True).start()

    def worker(self, steps):
        """Background thread; talks to the window only via the queue."""
        codes = []
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        for argv in steps:
            self.messages.put(("out", "$ " + " ".join(Path(a).name if i == 0 else a
                                                     for i, a in enumerate(argv)) + "\n"))
            try:
                with self.lock:
                    if self.closing:  # - window closed: no new step
                        break
                    self.proc = subprocess.Popen(argv, cwd=str(ROOT), env=env,
                                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                 text=True, encoding="utf-8", errors="replace",
                                                 creationflags=flags)
                proc = self.proc
                for line in proc.stdout:
                    self.messages.put(("out", line))
                code = proc.wait()
            except OSError as problem:
                self.messages.put(("out", f"Could not start the command: {problem}\n"))
                code = -1
            codes.append(code)
            if self.closing:
                break
        self.proc = None
        self.messages.put(("done", codes))

    def poll(self):
        """Every 50 ms in the window thread: show output, handle the end."""
        text = []
        finished = None
        try:
            for _ in range(MAX_MESSAGES_PER_TICK):
                kind, payload = self.messages.get_nowait()
                if kind == "out":
                    text.append(payload)
                else:
                    finished = payload
                    break
        except queue.Empty:
            pass
        if text:
            self.say("".join(text))
        if finished is not None:
            self.finish(finished)
        self.root.after(50, self.poll)

    def finish(self, codes):
        ok = all(code == 0 for code in codes)
        shown = ", ".join(str(c) for c in codes)
        word = "exit code" if len(codes) == 1 else "exit codes (one per step)"
        if self.closing:
            self.say("\nSTOPPED by the user (window closed).\n", "bad")
        elif ok:
            self.say(f"\nSUCCESS: finished with {word} {shown}.\n", "good")
        else:
            self.say(f"\nFAILED: {word} {shown}. Read the output above for the reason.\n", "bad")
        self.say(describe_results_change(self.before, snapshot_results()) + "\n", "note")
        self.refresh_status()
        self.set_busy(False)
        if ok and self.on_success:
            self.on_success()

    def on_checks(self):
        self.run_job("Run checks (leakage and simulation tests)", command_checks())

    def on_notebook(self):
        from tkinter import messagebox
        if self.busy:
            return
        if messagebox.askyesno("Run the notebook?", notebook_dialog_text(), icon="warning", default="no"):
            self.start_notebook()

    def start_notebook(self):
        """Notebook button after 'yes'."""
        self.run_job("Run the notebook", command_notebook())

    def on_export_html(self):
        self.run_job("Export notebook to HTML (no re-run)", command_export_html(),
                     on_success=lambda: self.open_path(ROOT / HTML_NAME))

    def on_open_notebook(self):
        self.open_path(ROOT / NOTEBOOK)

    def on_open_readme(self):
        self.open_path(ROOT / "README.md")

    def on_open_results(self):
        self.open_path(ROOT / "results")

    def open_path(self, path):
        try:
            if not path.exists():
                raise OSError(f"{path.name} does not exist yet")
            open_with_default_app(path)
            self.say(f"Opened {path.name}\n", "note")
        except OSError as problem:
            self.say(f"Could not open {path.name}: {problem}\n", "bad")

    def on_close(self):
        from tkinter import messagebox
        if self.busy:
            sure = messagebox.askyesno(
                "A command is still running",
                "A command is still running. Closing now stops it, and everything it "
                "started, half way. If it is the notebook run, the run log will show a "
                "started run with no finish.\n\nClose anyway?", icon="warning", default="no")
            if not sure:
                return
            self.stop_running_command()
        self.root.destroy()

    def stop_running_command(self):
        """Kill the running command tree; no later steps."""
        with self.lock:
            self.closing = True      # - worker checks this before each step
            proc = self.proc
        if proc is not None:
            kill_process_tree(proc)
            proc.wait()
        # - window closing: also print to console
        note = "STOPPED by the user: the window was closed while a command was running."
        self.say("\n" + note + "\n", "bad")
        print(note)


def main():
    if "--self-test" in sys.argv[1:]:
        return self_test()
    import tkinter as tk
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
