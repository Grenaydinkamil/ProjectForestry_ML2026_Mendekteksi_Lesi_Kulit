"""
Cek nama variabel yang dipakai SEBELUM didefinisikan.

Kenapa perlu script ini?
  Notebook adalah skrip yang dieksekusi sel demi sel. Kesalahan paling
  umum di notebook bukan salah ketik, tapi "variabel dipakai di sel
  sebelum sel yang membuatnya dijalankan" -- atau sebelum variabel itu
  dibuat sama sekali. Kesalahan seperti `NameError: name 'train_ds' is
  not defined` baru ketahuan setelah notebook berjalan 10 menit.

  Script ini membaca sumber .py notebook secara statis (tanpa eksekusi)
  dan menandai:
    - nama yang dipakai di level modul tapi tidak pernah didefinisikan
      di file itu (dan bukan builtin)
    - nama yang dipakai SEBELUM baris tempat ia didefinisikan

Cara kerja:
  1. Parse sumber jadi AST.
  2. Telusuri statement tingkat atas BERURUTAN.
  3. Untuk tiap statement, kumpulkan nama yang "diikat" (assignment,
     import, def, class, argumen fungsi, for/with/except target).
  4. Bandingkan dengan kumpulan nama yang sudah terikat sebelumnya.

Jalankan:  python scripts/check_undefined_names.py
"""

from __future__ import annotations

import ast
import builtins
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

BUILTINS = set(dir(builtins))

# Nama yang hanya ada di lingkungan notebook (IPython) atau di level modul
# Python, bukan variabel yang salah define.
KHUUSAN_LINGKUNGAN = {
    "__file__", "__name__", "__doc__", "__package__", "__spec__",
    "__builtins__", "__loader__", "__debug__",
    # IPython / Jupyter
    "display", "get_ipython", "In", "Out", "_", "__", "___", "exit", "quit",
}


class Analyzer(ast.NodeVisitor):
    """Mengumpulkan nama yang diikat oleh sebuah node."""

    def __init__(self) -> None:
        self.bound: set[str] = set()

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._bind_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._bind_target(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._bind_target(node.target)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._bind_target(node.target)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._bind_target(node.target)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._bind_target(node.target)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self._bind_target(item.optional_vars)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self._bind_target(item.optional_vars)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.bound.add((alias.asname or alias.name).split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.bound.add(alias.asname or alias.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.bound.add(node.name)
        # Argumen dan isi fungsi ikut "mengikat" nama, supaya nama yang
        # hanya dipakai di dalam fungsi tidak dilaporkan sebagai masalah.
        for arg in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
            self.bound.add(arg.arg)
        if node.args.vararg:
            self.bound.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.bound.add(node.args.kwarg.arg)
        for sub in node.body:
            Analyzer().visit(sub)
            for n in _names_bound(sub):
                self.bound.add(n)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bound.add(node.name)
        for sub in node.body:
            for n in _names_bound(sub):
                self.bound.add(n)

    def visit_Global(self, node: ast.Global) -> None:
        self.bound.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.bound.update(node.names)

    def _bind_target(self, target: ast.AST) -> None:
        for n in ast.walk(target):
            if isinstance(n, ast.Name):
                self.bound.add(n.id)


def _names_bound(node: ast.AST) -> set[str]:
    """Nama-nama yang diikat oleh satu statement (untuk>def/class)."""
    a = Analyzer()
    a.visit(node)
    return a.bound


def _loads(node: ast.AST) -> set[str]:
    """
    Nama-nama yang dibaca (bukan ditulis) oleh sebuah statement.

    Penting: badan fungsi, kelas, dan lambda TIDAK dihitung, karena
    nama di dalam scope itu punya aturan sendiri (dipakai sebelum
    didefinisikan di dalam fungsi juga boleh, mis. rekursi).
    """
    loads: set[str] = set()

    def walk(n: ast.AST) -> None:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef, ast.Lambda)):
            # hanya argumen default yang dievaluasi di scope luar
            for child in ast.iter_child_nodes(n):
                if isinstance(child, ast.arguments):
                    for d in child.defaults + [x for x in child.kw_defaults
                                               if x is not None]:
                        loads.update(_loads(d))
                elif isinstance(child, ast.AnnAssign):
                    loads.update(_loads(child))
            return
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            loads.add(n.id)
        for child in ast.iter_child_nodes(n):
            walk(child)

    walk(node)
    return loads


def _comprehension_targets(node: ast.AST) -> set[str]:
    """Nama yang diikat oleh target list comprehension (mis. `for x in ...`)."""
    hasil: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp,
                          ast.GeneratorExp)):
            for gen in n.generators:
                for t in ast.walk(gen.target):
                    if isinstance(t, ast.Name):
                        hasil.add(t.id)
    return hasil


def check_file(path: pathlib.Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    masalah: list[str] = []
    bound: set[str] = set()

    for stmt in tree.body:
        # Nama yang terikat oleh statement ini (termasuk di dalam bodinya)
        stmt_bound = _names_bound(stmt) | _comprehension_targets(stmt)

        # Nama yang dibaca pada statement ini, KECUALI yang diikat oleh
        # statement ini juga (nama boleh dipakai di dalam blok tempat
        # ia didefinisikan, misalnya di dalam loop).
        used = _loads(stmt) - _comprehension_targets(stmt) - stmt_bound

        for name in sorted(used):
            if name in bound or name in BUILTINS or name in KHUUSAN_LINGKUNGAN:
                continue
            masalah.append(
                f"{path.relative_to(REPO_ROOT)}:{stmt.lineno} "
                f"'{name}' dipakai tapi belum didefinisikan"
            )

        bound |= stmt_bound

    return masalah


def main() -> int:
    pola = ("src/*.py", "app/*.py", "scripts/*.py",
            "scripts/notebook_src/*.py")
    total = 0
    file_diperiksa = 0

    for pola_file in pola:
        for path in sorted(REPO_ROOT.glob(pola_file)):
            file_diperiksa += 1
            try:
                masalah = check_file(path)
            except SyntaxError as exc:
                print(f"[X] {path.name}: SyntaxError {exc}")
                total += 1
                continue
            for m in masalah:
                print(f"[X] {m}")
                total += 1

    print(f"\nFile diperiksa: {file_diperiksa} | masalah: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
