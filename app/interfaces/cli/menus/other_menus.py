import json
import os
from pathlib import Path
from typing import Any, Dict, List

from app import __version__
from app.interfaces.cli.menus.base_menu import BaseMenu


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _fmt(value: Any, spec: str = ".4f") -> str:
    return format(value, spec) if isinstance(value, (int, float)) else "—"


class ResultsMenu(BaseMenu):
    """Resumo dos runs de benchmark e dos relatórios de treino da CLI."""

    def _run_summaries(self) -> List[Path]:
        root = self.context.results_dir
        return sorted(
            p
            for p in root.rglob("run_summary.json")
            if len(p.relative_to(root).parts) <= 3
        )

    def show(self):
        self._print_header("RESULTADOS")
        summaries = self._run_summaries()
        if summaries:
            print(f"\n📈 Runs de benchmark em {self.context.results_dir}:")
        for path in summaries:
            data = _read_json(path)
            print(
                f"\n▶ {path.parent.relative_to(self.context.results_dir)}"
                f"  [status={data.get('status', '?')},"
                f" escopo={data.get('experiment_scope', '?')}]"
            )
            print(f"   {'modelo':26s} {'status':8s} {'EER':>8s} {'AUC':>8s}")
            for model in data.get("models") or []:
                clean = model.get("clean") or {}
                print(
                    f"   {str(model.get('model', '?')):26s}"
                    f" {str(model.get('status', '?')):8s}"
                    f" {_fmt(clean.get('eer')):>8s} {_fmt(clean.get('auc_roc')):>8s}"
                )

        reports = sorted(self.context.results_dir.glob("training_report_*.json"))
        if reports:
            print("\n📝 Relatórios de treino da CLI:")
        for path in reports:
            data = _read_json(path)
            accuracy = (data.get("results") or {}).get("accuracy")
            print(
                f"   - {data.get('model_name', path.stem)}"
                f" ({data.get('architecture', '?')}), acurácia={_fmt(accuracy, '.2%')}"
            )

        if not summaries and not reports:
            print("\n   Nenhum resultado encontrado.")
        input("\nPressione Enter para continuar...")


class SettingsMenu(BaseMenu):
    """Configuração efetiva do processo (somente leitura)."""

    _ENV_KEYS = (
        "DEEPFAKE_ENV",
        "XFAKE_STORAGE_DIR",
        "MODELS_DIR",
        "DEEPFAKE_MODELS_DIR",
        "XFAKE_MODELS_DIR",
        "DATABASE_URL",
        "XFAKE_CREATE_DEFAULT_MODELS",
        "XFAKE_MAX_UPLOAD_MB",
    )

    def show(self):
        self._print_header("CONFIGURAÇÕES")
        paths = self.context.paths
        print(f"\nSpeechChecker {__version__}")
        print("\n📁 Diretórios operacionais:")
        for label, path in (
            ("dados", paths.data),
            ("datasets", paths.datasets),
            ("modelos", paths.models),
            ("resultados", paths.results),
            ("uploads", paths.uploads),
            ("logs", paths.logs),
            ("banco", paths.database),
        ):
            print(f"   {label:11s} {path}")

        print("\n🔧 Variáveis de ambiente (definidas):")
        defined = [(k, os.environ[k]) for k in self._ENV_KEYS if os.environ.get(k)]
        for key, value in defined:
            print(f"   {key}={value}")
        if not defined:
            print("   (nenhuma — usando os padrões)")

        models = self.context.detection_service.get_available_models()
        print(f"\n🤖 Modelos carregáveis: {len(models)}")
        print(
            f"🧠 Arquiteturas treináveis: {len(self.context.available_architectures)}"
        )
        print("\nPara alterar, edite o .env (veja .env.example) e reinicie.")
        input("\nPressione Enter para continuar...")
