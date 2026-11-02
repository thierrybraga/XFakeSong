import json
from pathlib import Path
from typing import Optional

from app.interfaces.cli.menus.base_menu import BaseMenu


class TrainingMenu(BaseMenu):
    """Menu para treinamento de modelos.

    Usa o mesmo contrato do ``POST /api/v1/training/start``: um ``.npz`` com
    ``X_train``/``y_train`` (e, opcionalmente, ``X_val``/``y_val``) no formato
    de entrada da arquitetura escolhida, treinado pelo ``TrainingService``.
    """

    def show(self):
        while True:
            print("\n🤖 TREINAMENTO DE MODELO")
            print("-" * 40)
            print("1. Treinar novo modelo")
            print("2. Listar modelos disponíveis")
            print("3. Voltar ao menu principal")

            choice = input("\nEscolha uma opção: ").strip()

            if choice == "1":
                self.train_model()
            elif choice == "2":
                self.list_trained_models()
            elif choice == "3":
                break
            else:
                print("❌ Opção inválida!")

    def _ask_dataset(self) -> Optional[Path]:
        candidates = sorted(self.context.datasets_dir.rglob("*.npz"))
        if candidates:
            print("\n📦 Datasets .npz encontrados:")
            for i, path in enumerate(candidates, 1):
                print(f"{i:2d}. {path.relative_to(self.context.datasets_dir)}")
        raw = input(
            "\nNúmero do dataset ou caminho de um .npz: "
        ).strip().strip('"')
        if raw.isdigit() and 1 <= int(raw) <= len(candidates):
            return candidates[int(raw) - 1]
        path = Path(raw)
        if raw and path.is_file() and path.suffix == ".npz":
            return path
        print("❌ Dataset .npz não encontrado.")
        return None

    def _ask_int(self, label: str, default: int) -> int:
        raw = input(f"{label} (padrão: {default}): ").strip()
        return int(raw) if raw.isdigit() and int(raw) > 0 else default

    def train_model(self):
        """Treina um novo modelo pelo TrainingService."""
        print("\n🚀 Iniciando treinamento de modelo")

        architectures = self.context.available_architectures
        print("\n📋 Arquiteturas disponíveis:")
        for i, arch in enumerate(architectures, 1):
            print(f"{i:2d}. {arch}")

        try:
            arch_choice = input("\nEscolha uma arquitetura (número): ").strip()
            if not arch_choice.isdigit() or not (
                1 <= int(arch_choice) <= len(architectures)
            ):
                print("❌ Escolha inválida!")
                return
            selected_arch = architectures[int(arch_choice) - 1]
            print(f"\n✅ Arquitetura selecionada: {selected_arch}")

            dataset_path = self._ask_dataset()
            if dataset_path is None:
                return

            defaults = self.context.training_config
            epochs = self._ask_int("Número de épocas", defaults.epochs)
            batch_size = self._ask_int("Tamanho do batch", defaults.batch_size)

            print("\n🔧 Configurações:")
            print(f"   - Arquitetura: {selected_arch}")
            print(f"   - Dataset: {dataset_path}")
            print(f"   - Épocas: {epochs}")
            print(f"   - Batch size: {batch_size}")

            if input("\nIniciar treinamento? (s/N): ").strip().lower() != "s":
                print("❌ Treinamento cancelado.")
                return

            print("🚀 Treinando...")
            result = self.context.training_service.train_model(
                architecture=selected_arch,
                dataset_path=str(dataset_path),
                config={"epochs": epochs, "batch_size": batch_size},
            )
            if not result.is_success:
                print(f"❌ Erro no treinamento: {'; '.join(result.errors)}")
                return

            metadata = result.data
            print("\n✅ Modelo treinado com sucesso!")
            print(f"📁 Salvo em: {metadata.file_path}")
            print(f"📊 Acurácia: {metadata.accuracy:.2%}")

            report_path = (
                self.context.results_dir / f"training_report_{metadata.name}.json"
            )
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_name": metadata.name,
                        "architecture": selected_arch,
                        "dataset": str(dataset_path),
                        "training_config": {
                            "epochs": epochs,
                            "batch_size": batch_size,
                        },
                        "results": metadata.metrics,
                        "timestamp": metadata.created_at.isoformat(),
                    },
                    f,
                    indent=2,
                )
            print(f"📝 Relatório: {report_path}")

            # O DetectionService é compartilhado: recarrega para o modelo novo
            # aparecer na inferência sem reiniciar.
            self.context.detection_service.reload_models()

        except Exception as e:
            print(f"❌ Erro durante o treinamento: {e}")
            self.context.logger.error(f"Erro de treinamento: {e}")

    def list_trained_models(self):
        """Lista os modelos que a inferência consegue carregar."""
        print("\n📋 Modelos disponíveis:")
        models = self.context.detection_service.get_available_models()
        if not models:
            print("   Nenhum modelo encontrado.")
        for name in models:
            print(f"   - {name}")
