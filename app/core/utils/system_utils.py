from pathlib import Path
from datetime import datetime, timedelta
import shutil
import fnmatch
from typing import List


def bootstrap_dirs(app_base: Path) -> None:
    """Garante que os diretórios necessários existem."""
    models = app_base / "models"
    results = app_base / "results"
    models.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    print(f"✅ Diretórios garantidos: {models} e {results}")


def cleanup_workspace(project_root: Path, days: int = 30,
                      dry_run: bool = False, delete_datasets: bool = False) -> None:
    """Limpa arquivos antigos e temporários do workspace."""
    app_base = project_root / "app"
    cutoff = datetime.now() - timedelta(days=days)

    targets: List[Path] = []

    for log_file in project_root.glob("*.log"):
        targets.append(log_file)

    patterns = [
        "*_results.json",
        "*_optimization_*.json",
    ]
    for pattern in patterns:
        for p in project_root.glob(pattern):
            targets.append(p)

    for p in (app_base / "results").glob("*.json"):
        targets.append(p)

    image_patterns = [
        "*evaluation*.png",
        "*confusion_matrix*.png",
        "*training_history*.png",
        "*spectrograms*.png",
        "*architecture*.png",
    ]
    for img in project_root.glob("*.png"):
        for pat in image_patterns:
            if fnmatch.fnmatch(img.name, pat):
                targets.append(img)
                break

    pycache_dirs = list(app_base.rglob("__pycache__"))

    if delete_datasets:
        ds_root = project_root / "datasets"
        ds_app = app_base / "datasets"
        if ds_root.exists():
            targets.append(ds_root)
        if ds_app.exists():
            targets.append(ds_app)

    removed = 0
    for path in targets:
        try:
            if path.is_dir():
                if dry_run:
                    print(f"DRY-RUN: rmtree {path}")
                else:
                    shutil.rmtree(path)
                    removed += 1
            elif path.is_file():
                if path.stat().st_mtime < cutoff.timestamp():
                    if dry_run:
                        print(f"DRY-RUN: unlink {path}")
                    else:
                        path.unlink()
                        removed += 1
                else:
                    if dry_run:
                        print(f"DRY-RUN: keep (recent) {path}")
            else:
                continue
        except Exception as e:
            print(f"⚠️ Falha ao remover {path}: {e}")

    for d in pycache_dirs:
        try:
            if dry_run:
                print(f"DRY-RUN: rmtree {d}")
            else:
                shutil.rmtree(d)
                removed += 1
        except Exception as e:
            print(f"⚠️ Falha ao remover {d}: {e}")

    print(f"✅ Limpeza concluída, itens removidos: {removed}")
