from pathlib import Path

from app.interfaces.cli.menus.base_menu import BaseMenu


class InferenceMenu(BaseMenu):
    """Menu para inferência de áudio."""

    def show(self):
        while True:
            print("\n🔍 INFERÊNCIA DE ÁUDIO")
            print("-" * 40)
            print("1. Analisar arquivo de áudio")
            print("2. Análise em lote")
            print("3. Voltar ao menu principal")

            choice = input("\nEscolha uma opção: ").strip()

            if choice == "1":
                self.analyze_single_audio()
            elif choice == "2":
                self.analyze_batch_audio()
            elif choice == "3":
                break
            else:
                print("❌ Opção inválida!")

    def analyze_single_audio(self):
        """Analisa um único arquivo de áudio."""
        print("\n🎵 Análise de Arquivo Único")

        model_files = list(self.context.models_dir.glob("*.h5"))
        if not model_files:
            print("❌ Nenhum modelo treinado encontrado.")
            return

        print("\nModelos disponíveis:")
        for i, model_file in enumerate(model_files, 1):
            print(f"{i:2d}. {model_file.stem}")

        try:
            choice_input = input("\nEscolha um modelo (número): ").strip()
            if not choice_input.isdigit():
                print("❌ Entrada inválida!")
                return

            choice = int(choice_input) - 1
            if choice < 0 or choice >= len(model_files):
                print("❌ Escolha inválida!")
                return

            selected_model = model_files[choice]

            audio_path = input(
                "\nCaminho do arquivo de áudio: ").strip().strip('"')

            if not Path(audio_path).exists():
                print("❌ Arquivo não encontrado!")
                return

            print(f"\n🔍 Analisando: {Path(audio_path).name}")
            print(f"📊 Modelo: {selected_model.stem}")

            try:
                result = self.context.detection_service.detect_deepfake(
                    audio_path, selected_model.stem)

                if result.status.name == "SUCCESS":
                    prediction = result.data
                    confidence = prediction.get('confidence', 0)
                    is_fake = prediction.get('is_deepfake', False)

                    print("\n📊 Resultado da Análise:")
                    print(
                        f"   🎯 Classificação: {
                            '🚨 DEEPFAKE' if is_fake else '✅ REAL'}")
                    print(f"   📈 Confiança: {confidence:.2%}")

                    if confidence > 0.8:
                        print("   ✅ Alta confiança")
                    elif confidence > 0.6:
                        print("   ⚠️  Confiança moderada")
                    else:
                        print("   ❌ Baixa confiança")
                else:
                    print(f"❌ Erro na análise: {result.message}")

            except Exception as e:
                print(f"❌ Erro ao analisar áudio: {e}")
                self.context.logger.error(f"Erro na inferência: {e}")

        except ValueError:
            print("❌ Entrada inválida!")

    def analyze_batch_audio(self):
        """Analisa múltiplos arquivos de áudio."""
        print("\n📁 Análise em Lote")
        print("⚠️  Funcionalidade em desenvolvimento.")
