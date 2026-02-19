"""
Experiment Results Comparison Tool

Opens a tkinter UI with a checkbox directory tree to select run directories,
finds evaluation.csv files in them, and aggregates summaries into a
timestamped output CSV.

Usage:
    python experiments/compare_results.py
"""
import os
import csv
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def read_summary(eval_csv_path):
    """Read the SUMMARY row from an evaluation.csv file."""
    try:
        with open(eval_csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == "SUMMARY":
                    return {
                        "episodes": row[1],
                        "avg_reward": row[2],
                        "crash_pct": row[3],
                        "arrival_pct": row[4],
                    }
    except (OSError, IndexError):
        pass
    return None


def generate_csv(selected_dirs):
    """Find evaluation CSVs in selected dirs, aggregate into output CSV."""
    seen = set()
    eval_files = []
    for d in selected_dirs:
        for root, dirs, files in os.walk(d):
            if "evaluation.csv" in files:
                real = os.path.realpath(os.path.join(root, "evaluation.csv"))
                if real not in seen:
                    seen.add(real)
                    eval_files.append(real)
    eval_files.sort()

    if not eval_files:
        return None, 0

    rows = []
    for ef in eval_files:
        summary = read_summary(ef)
        if summary:
            rel = os.path.relpath(ef, PROJECT_ROOT)
            summary["source"] = rel
            rows.append(summary)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(SCRIPT_DIR, f"comparison_{timestamp}.csv")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Selected directories:"])
        for d in selected_dirs:
            writer.writerow([os.path.relpath(d, PROJECT_ROOT)])
        writer.writerow([])
        writer.writerow(["source", "episodes", "avg_reward", "crash_pct", "arrival_pct"])
        for r in rows:
            writer.writerow([
                r["source"], r["episodes"], r["avg_reward"],
                r["crash_pct"], r["arrival_pct"],
            ])

    return output_path, len(eval_files)


class CheckboxTree:
    """Treeview with checkboxes for directory selection."""

    TAG_CHECKED = "checked"
    TAG_UNCHECKED = "unchecked"
    TAG_EVAL = "has_eval"

    def __init__(self, parent, root_path):
        self.root_path = root_path
        self.checked = set()  # set of iids that are checked

        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))

        scrollbar_y = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        style = ttk.Style()
        style.configure("Dir.Treeview", font=("sans-serif", 13), rowheight=50)

        self.tree = ttk.Treeview(
            frame, columns=("check", "info"), show="tree",
            yscrollcommand=scrollbar_y.set,
            xscrollcommand=scrollbar_x.set,
            style="Dir.Treeview",
        )
        self.tree.pack(fill=tk.BOTH, expand=True)
        scrollbar_y.config(command=self.tree.yview)
        scrollbar_x.config(command=self.tree.xview)

        self.tree.column("#0", width=700, stretch=True)

        self.tree.tag_configure(self.TAG_EVAL, foreground="#16a34a")

        # Populate root
        root_name = os.path.basename(root_path)
        root_iid = self.tree.insert("", tk.END, text=f"[ ] {root_name}",
                                     open=True, values=(root_path,))
        self._populate(root_iid, root_path)

        # Bind click to toggle checkbox
        self.tree.bind("<Button-1>", self._on_click)
        # Bind expand to lazy-load deeper levels
        self.tree.bind("<<TreeviewOpen>>", self._on_expand)

    def _populate(self, parent_iid, parent_path):
        """Add immediate subdirectories as children."""
        try:
            entries = sorted(os.listdir(parent_path))
        except OSError:
            return

        for name in entries:
            if name.startswith("."):
                continue
            full = os.path.join(parent_path, name)
            if not os.path.isdir(full):
                continue

            has_eval = os.path.isfile(os.path.join(full, "evaluation.csv"))
            label = f"[ ] {name}"
            if has_eval:
                label += "  [eval]"

            tags = (self.TAG_EVAL,) if has_eval else ()
            iid = self.tree.insert(parent_iid, tk.END, text=label,
                                   values=(full,), tags=tags)

            # Add a dummy child so the expand arrow shows
            if self._has_subdirs(full):
                self.tree.insert(iid, tk.END, text="")

    def _has_subdirs(self, path):
        try:
            for name in os.listdir(path):
                if name.startswith("."):
                    continue
                if os.path.isdir(os.path.join(path, name)):
                    return True
        except OSError:
            pass
        return False

    def _on_expand(self, event):
        """Lazy-load children when a node is expanded."""
        iid = self.tree.focus()
        children = self.tree.get_children(iid)
        # If there's exactly one child with empty text, it's the dummy
        if len(children) == 1 and self.tree.item(children[0], "text") == "":
            self.tree.delete(children[0])
            path = self.tree.item(iid, "values")[0]
            self._populate(iid, path)

    def _on_click(self, event):
        """Toggle checkbox on click."""
        iid = self.tree.identify_row(event.y)
        if not iid:
            return

        # Toggle
        if iid in self.checked:
            self.checked.discard(iid)
            self._update_label(iid, checked=False)
        else:
            self.checked.add(iid)
            self._update_label(iid, checked=True)

    def _update_label(self, iid, checked):
        text = self.tree.item(iid, "text")
        if checked:
            text = text.replace("[ ]", "[x]", 1)
        else:
            text = text.replace("[x]", "[ ]", 1)
        self.tree.item(iid, text=text)

    def get_selected_paths(self):
        """Return absolute paths of all checked directories."""
        paths = []
        for iid in self.checked:
            vals = self.tree.item(iid, "values")
            if vals:
                paths.append(vals[0])
        paths.sort()
        return paths


class CompareApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Compare Experiment Results")
        self.root.geometry("950x650")
        self.root.resizable(True, True)

        # --- Header ---
        header = ttk.Frame(root)
        header.pack(fill=tk.X, padx=10, pady=(10, 5))
        ttk.Label(header, text="Select run directories (click to check/uncheck):",
                  font=("sans-serif", 11)).pack(anchor=tk.W)

        # --- Checkbox tree ---
        self.tree = CheckboxTree(root, PROJECT_ROOT)

        # --- Bottom frame: Done button ---
        bottom = ttk.Frame(root)
        bottom.pack(fill=tk.X, padx=10, pady=10)

        done_btn = tk.Button(bottom, text="Done - Generate CSV", command=self.done,
                             padx=16, pady=6, bg="#16a34a", fg="white",
                             font=("sans-serif", 11, "bold"))
        done_btn.pack(side=tk.RIGHT)

        count_label_text = "Directories with [eval] contain evaluation.csv"
        ttk.Label(bottom, text=count_label_text, foreground="#16a34a",
                  font=("sans-serif", 9)).pack(side=tk.LEFT)

    def done(self):
        selected = self.tree.get_selected_paths()
        if not selected:
            messagebox.showwarning("No directories",
                                   "Please check at least one directory.")
            return

        output_path, eval_count = generate_csv(selected)

        if output_path is None:
            messagebox.showerror("No results",
                                 "No evaluation.csv files found in selected directories.")
            return

        rel_output = os.path.relpath(output_path, PROJECT_ROOT)
        messagebox.showinfo("Done",
                            f"Found {eval_count} evaluation.csv file(s).\n\n"
                            f"Saved to:\n{rel_output}")
        print(f"Comparison CSV saved to: {rel_output}")
        print(f"Found {eval_count} evaluation.csv file(s).")
        self.root.destroy()


def main():
    root = tk.Tk()
    CompareApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
