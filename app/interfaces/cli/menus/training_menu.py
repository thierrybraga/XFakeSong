import json
from datetime import datetime

import joblib
import numpy as np

from app.core.training.secure_training_pipeline import (
    SecureTrainingConfig,
    SecureTrainingPipeline,
)
from app.domain.models.architectures.factory import BaseArchitectureFactory
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

            secure_config = SecureTrainingConfig(
                test_size=0.2,
                validation_size=0.2,
                random_state=42,
                normalize_features=True,
                feature_selection=False
            )

            pipeline = SecureTrainingPipeline(secure_config)

            X = np.array(features)
            y = np.array(labels)

            train_result = pipeline.prepare_data(X, y)
            if train_result.status.name != "SUCCESS":
                print(
                    f"❌ Erro na preparação dos dados: {train_result.message}")
                return

            factory = BaseArchitectureFactory()
            model = factory.create_model(
                selected_arch,
                input_shape=(X.shape[1],),
                num_classes=2
            )

            training_result = pipeline.train_model(
                model,
                train_result.data['X_train'],
                train_result.data['y_train'],
                train_result.data['X_val'],
                train_result.data['y_val'],
                epochs=epochs,
                batch_size=batch_size
            )

            if training_result.status.name == "SUCCESS":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"{selected_arch}_{timestamp}"
                model_path = self.context.models_dir / f"{model_name}.h5"

                model.save(str(model_path))

                if hasattr(pipeline, 'scaler') and pipeline.scaler:
                    scaler_path = self.context.models_dir / \
                        f"{model_name}_scaler.pkl"
                    joblib.dump(pipeline.scaler, scaler_path)

                print("\n✅ Modelo treinado com sucesso!")
                print(f"📁 Salvo em: {model_path}")
                print(
                    f"📊 Acurácia: {training_result.data.get('accuracy', 'N/A')}")

                report_path = self.context.results_dir / \
                    f"training_report_{model_name}.json"
                with open(report_path, 'w') as f:
                    json.dump({
                        'model_name': model_name,
                        'architecture': selected_arch,
                        'training_config': {
                            'epochs': epochs,
                            'batch_size': batch_size
                        },
                        'results': training_result.data,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)

        except Exception as e:
            print(f"❌ Erro durante o treinamento: {e}")
            self.context.logger.error(f"Erro de treinamento: {e}")

    def list_trained_models(self):
        """Lista os modelos já treinados."""
        print("\n📋 Modelos Treinados:")
        models = list(self.context.models_dir.glob("*.h5"))
        if not models:
            print("   Nenhum modelo encontrado.")
        else:
            for model_file in models:
                print(f"   - {model_file.name}")

    def evaluate_model(self):
        """Avalia um modelo existente (placeholder)."""
        print(
            "\n⚠️ Funcionalidade de avaliação ainda não implementada neste menu refatorado.")
