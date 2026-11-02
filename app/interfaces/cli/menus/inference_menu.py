from pathlib import Path
from typing import Optional

from app.domain.services.upload_service import AudioUploadService
from app.interfaces.cli.menus.base_menu import BaseMenu


class InferenceMenu(BaseMenu):
    """Menu para inferência de áudio."""

    def show(self):
        while True:
            print("\n🔍 INFERÊNCIA DE ÁUDIO")
            print("-" * 40)
            print("1. Analisar arquivo de áudio")
            print("2. Análise em lote (pasta)")
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

    def _select_model(self) -> Optional[str]:
        """Escolhe entre os modelos que o DetectionService carregou."""
        models = self.context.detection_service.get_available_models()
        if not models:
            print("❌ Nenhum modelo treinado encontrado.")
            return None
        print("\n📋 Modelos disponíveis:")
        for i, name in enumerate(models, 1):
            print(f"{i:2d}. {name}")
        raw = input("\nEscolha um modelo (número): ").strip()
        if not raw.isdigit() or not 1 <= int(raw) <= len(models):
            print("❌ Escolha inválida!")
            return None
        return models[int(raw) - 1]

    def _detect(self, audio_path: Path, model_name: str):
        result = self.context.detection_service.detect_from_file(
            audio_path, model_name=model_name
        )
        if not result.is_success:
            raise RuntimeError("; ".join(result.errors) or "erro desconhecido")
        return result.data

    def analyze_single_audio(self):
        """Analisa um único arquivo de áudio."""
        print("\n🎵 Análise de Arquivo Único")
        model_name = self._select_model()
        if model_name is None:
            return

        audio_path = Path(input("\nCaminho do arquivo de áudio: ").strip().strip('"'))
        if not audio_path.is_file():
            print("❌ Arquivo não encontrado!")
            return

        print(f"\n🔍 Analisando: {audio_path.name}")
        print(f"📊 Modelo: {model_name}")
        try:
            prediction = self._detect(audio_path, model_name)
        except Exception as e:
            print(f"❌ Erro na análise: {e}")
            self.context.logger.error(f"Erro na inferência: {e}")
            return

        confidence = prediction.confidence
        print("\n📊 Resultado da Análise:")
        print(
            f"   🎯 Classificação: {'🚨 DEEPFAKE' if prediction.is_fake else '✅ REAL'}"
        )
        print(f"   📈 Confiança: {confidence:.2%}")
        if confidence > 0.8:
            print("   ✅ Alta confiança")
        elif confidence > 0.6:
            print("   ⚠️  Confiança moderada")
        else:
            print("   ❌ Baixa confiança")

    def analyze_batch_audio(self):
        """Analisa todos os áudios suportados de uma pasta."""
        print("\n📁 Análise em Lote")
        model_name = self._select_model()
        if model_name is None:
            return

        folder = Path(input("\nPasta com os áudios: ").strip().strip('"'))
        if not folder.is_dir():
            print("❌ Pasta não encontrada!")
            return
        files = sorted(
            p
            for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in AudioUploadService.SUPPORTED_FORMATS
        )
        if not files:
            print("❌ Nenhum áudio suportado na pasta.")
            return

        n_fake = n_real = n_error = 0
        print(f"\n🔍 {len(files)} arquivo(s) com o modelo {model_name}:")
        for path in files:
            try:
                prediction = self._detect(path, model_name)
            except Exception as e:
                n_error += 1
                print(f"   ❌ {path.name}: {e}")
                continue
            if prediction.is_fake:
                n_fake += 1
            else:
                n_real += 1
            label = "DEEPFAKE" if prediction.is_fake else "REAL"
            print(f"   {label:8s} {prediction.confidence:6.1%}  {path.name}")

        print(f"\n📊 Resumo: {n_real} real, {n_fake} deepfake, {n_error} com erro")
