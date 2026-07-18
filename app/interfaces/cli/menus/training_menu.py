import json

import numpy as np

from app.domain.services.training_service import TrainingService
from app.interfaces.cli.menus.base_menu import BaseMenu


class TrainingMenu(BaseMenu):
    """Menu para treinamento de modelos."""

    def show(self):
        while True:
            print("\n🤖 TREINAMENTO DE MODELO")
            print("-" * 40)
            print("1. Treinar novo modelo")
            print("2. Listar modelos treinados")
            print("3. Avaliar modelo")
            print("4. Voltar ao menu principal")

            choice = input("\nEscolha uma opção: ").strip()

            if choice == "1":
                self.train_model()
            elif choice == "2":
                self.list_trained_models()
            elif choice == "3":
                self.evaluate_model()
            elif choice == "4":
                break
            else:
                print("❌ Opção inválida!")

    def train_model(self):
        """Treina um novo modelo."""
        print("\n🚀 Iniciando treinamento de modelo")

        features_file = self.context.datasets_dir / \
            "features" / "extracted_features.json"
        if not features_file.exists():
            print("❌ Features não encontradas. Execute a extração de features primeiro.")
            return

        print("\n📋 Arquiteturas disponíveis:")
        for i, arch in enumerate(self.context.available_architectures, 1):
            print(f"{i:2d}. {arch}")

        try:
            arch_choice_input = input(
                "\nEscolha uma arquitetura (número): ").strip()
            if not arch_choice_input.isdigit():
                print("❌ Entrada inválida!")
                return

            arch_choice = int(arch_choice_input) - 1
            if arch_choice < 0 or arch_choice >= len(
                    self.context.available_architectures):
                print("❌ Escolha inválida!")
                return

            selected_arch = self.context.available_architectures[arch_choice]
            print(f"\n✅ Arquitetura selecionada: {selected_arch}")

            epochs_input = input(
                f"Número de épocas (padrão: {self.context.training_config.epochs}): ").strip()
            epochs = int(
                epochs_input) if epochs_input else self.context.training_config.epochs

            batch_size_input = input(
                f"Tamanho do batch (padrão: {self.context.training_config.batch_size}): ").strip()
            batch_size = int(
                batch_size_input) if batch_size_input else self.context.training_config.batch_size

            print("\n🔧 Configurações:")
            print(f"   - Arquitetura: {selected_arch}")
            print(f"   - Épocas: {epochs}")
            print(f"   - Batch size: {batch_size}")

            if input("\nIniciar treinamento? (s/N): ").strip().lower() != 's':
                print("❌ Treinamento cancelado.")
                return

            with open(features_file, 'r') as f:
                data = json.load(f)

            features = data['features']
            labels = data['labels']

            print(f"\n📊 Dados carregados: {len(features)} amostras")
            print("🚀 Iniciando treinamento...")

            X = np.asarray(features, dtype=np.float32)
            y = np.asarray(labels)

            # Split treino/validação com embaralhamento (os dados chegam
            # ordenados: primeiro todas as amostras 'real', depois 'fake' —
            # sem shuffle, o split viraria 100% de uma classe em cada lado).
            rng = np.random.RandomState(42)
            indices = rng.permutation(len(X))
            split_at = max(1, int(len(X) * 0.8))
            train_idx, val_idx = indices[:split_at], indices[split_at:]

            # TrainingService espera um dataset .npz (X_train/y_train/X_val/
            # y_val), não o JSON de features — gera um arquivo derivado.
            dataset_path = (
                self.context.datasets_dir / "features" / "train_dataset.npz"
            )
            np.savez(
                dataset_path,
                X_train=X[train_idx], y_train=y[train_idx],
                X_val=X[val_idx], y_val=y[val_idx],
            )

            training_service = TrainingService(
                models_dir=str(self.context.models_dir))
            result = training_service.train_model(
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

            report_path = self.context.results_dir / \
                f"training_report_{metadata.name}.json"
            with open(report_path, 'w') as f:
                json.dump({
                    'model_name': metadata.name,
                    'architecture': selected_arch,
                    'training_config': {
                        'epochs': epochs,
                        'batch_size': batch_size
                    },
                    'results': metadata.metrics,
                    'timestamp': metadata.created_at.isoformat()
                }, f, indent=2)

        except Exception as e:
            print(f"❌ Erro durante o treinamento: {e}")
            self.context.logger.error(f"Erro de treinamento: {e}")

    def list_trained_models(self):
        """Lista os modelos já treinados."""
        print("\n📋 Modelos Treinados:")
        models = list(self.context.models_dir.glob("*.keras")) + \
            list(self.context.models_dir.glob("*.h5"))
        if not models:
            print("   Nenhum modelo encontrado.")
        else:
            for model_file in models:
                print(f"   - {model_file.name}")

    def evaluate_model(self):
        """Avalia um modelo existente (placeholder)."""
        print(
            "\n⚠️ Funcionalidade de avaliação ainda não implementada neste menu refatorado.")
