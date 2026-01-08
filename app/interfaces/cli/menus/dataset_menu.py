import json
from datetime import datetime
from pathlib import Path
from app.interfaces.cli.menus.base_menu import BaseMenu
from app.domain.features.extractors.segmented_feature_extractor import SegmentedFeatureExtractor, SegmentedExtractionConfig
from app.domain.features.exporters.csv_feature_exporter import CSVExportConfig, CSVFeatureExporter
from app.domain.features.extractors.spectral.spectral_features import SpectralFeatureExtractor
from app.domain.features.extractors.temporal.temporal_features import TemporalFeatureExtractor


class DatasetMenu(BaseMenu):
    """Menu para gerenciamento de dataset."""

    def show(self):
        while True:
            print("\n📊 GERENCIAMENTO DE DATASET")
            print("-" * 40)
            print("1. Verificar status do dataset")
            print("2. Carregar novo dataset")
            print("3. Extrair features")
            print("4. Voltar ao menu principal")

            choice = input("\nEscolha uma opção: ").strip()

            if choice == "1":
                self.check_dataset_status()
            elif choice == "2":
                self.load_dataset()
            elif choice == "3":
                self.extract_features()
            elif choice == "4":
                break
            else:
                print("❌ Opção inválida!")

    def check_dataset_status(self):
        """Verifica o status do dataset atual."""
        print("\n🔍 Verificando status do dataset...")

        samples_dir = self.context.datasets_dir / "samples"
        real_dir = samples_dir / "real"
        fake_dir = samples_dir / "fake"

        if not samples_dir.exists():
            print("❌ Diretório de amostras não encontrado.")
            return

        real_files = list(real_dir.glob("*.wav")) if real_dir.exists() else []
        fake_files = list(fake_dir.glob("*.wav")) if fake_dir.exists() else []

        print(f"✅ Amostras reais: {len(real_files)}")
        print(f"✅ Amostras falsas: {len(fake_files)}")
        print(f"📊 Total de amostras: {len(real_files) + len(fake_files)}")

        if len(real_files) > 0 and len(fake_files) > 0:
            print("✅ Dataset está pronto para treinamento!")
        else:
            print("⚠️  Dataset incompleto. Adicione mais amostras.")

    def load_dataset(self):
        """Carrega um novo dataset."""
        print("\n📁 Carregamento de Dataset")
        print("Coloque seus arquivos de áudio nas seguintes pastas:")
        print(
            f"- Áudios reais: {self.context.datasets_dir / 'samples' / 'real'}")
        print(
            f"- Áudios falsos: {self.context.datasets_dir / 'samples' / 'fake'}")
        print("\nFormatos suportados: .wav, .mp3, .flac")

        input("\nPressione Enter após adicionar os arquivos...")
        self.check_dataset_status()

    def extract_features(self):
        """Extrai features do dataset com opção de segmentação."""
        print("\n🔧 EXTRAÇÃO DE FEATURES")
        print("-" * 40)
        print("Escolha o método de extração:")
        print("1. 📊 Extração segmentada (intervalos de 1s)")
        print("2. 🎵 Extração do áudio completo")
        print("3. ↩️  Voltar")

        choice = input("\nEscolha uma opção: ").strip()

        if choice == "1":
            self._extract_features_segmented()
        elif choice == "2":
            self._extract_features_complete()
        elif choice == "3":
            return
        else:
            print("❌ Opção inválida!")

    def _extract_features_segmented(self):
        """Extrai features usando segmentação de áudio."""
        print("\n🔧 Extraindo features com segmentação (intervalos de 1s)...")

        try:
            samples_dir = self.context.datasets_dir / "samples"
            features_dir = self.context.datasets_dir / "features"
            features_dir.mkdir(exist_ok=True)

            real_dir = samples_dir / "real"
            fake_dir = samples_dir / "fake"

            features_data = []
            labels = []

            csv_config = CSVExportConfig(
                output_base_dir="datasets/features",
                include_metadata=True,
                decimal_places=4
            )

            config = SegmentedExtractionConfig(
                segment_duration=1.0,
                overlap_ratio=0.0,
                target_sample_rate=16000,
                extract_spectral=True,
                extract_cepstral=True,
                extract_temporal=True,
                extract_prosodic=True,
                extract_formant=True,
                extract_voice_quality=True,
                extract_perceptual=True,
                normalize_segments=True,
                remove_silence=False,
                export_csv=True,
                csv_config=csv_config
            )

            segmented_extractor = SegmentedFeatureExtractor(config)

            self._process_directory_segmented(
                real_dir, 0, segmented_extractor, features_data, labels)
            self._process_directory_segmented(
                fake_dir, 1, segmented_extractor, features_data, labels)

            features_file = features_dir / "extracted_features.json"
            with open(features_file, 'w') as f:
                json.dump({
                    'features': features_data,
                    'labels': labels,
                    'timestamp': datetime.now().isoformat(),
                    'total_samples': len(features_data),
                    'extraction_method': 'segmented',
                    'segment_duration': config.segment_duration,
                    'feature_dimension': len(features_data[0]) if features_data else 0,
                    'segmentation_config': {
                        'segment_duration': config.segment_duration,
                        'overlap_ratio': config.overlap_ratio,
                        'target_sample_rate': config.target_sample_rate,
                        'normalize_segments': config.normalize_segments,
                        'remove_silence': config.remove_silence
                    }
                }, f, indent=2)

            print(f"✅ Extração segmentada concluída!")
            print(f"   Total de segmentos: {len(features_data)}")
            print(
                f"   Dimensões das features: {len(features_data[0]) if features_data else 0}")
            print(f"   Duração dos segmentos: {config.segment_duration}s")
            print(f"   Arquivos salvos em: {features_file}")

        except Exception as e:
            print(f"❌ Erro ao extrair features: {e}")
            self.context.logger.error(f"Erro na extração de features: {e}")

    def _process_directory_segmented(self, directory: Path, label: int,
                                     extractor: SegmentedFeatureExtractor, features_data: list, labels: list):
        if directory.exists():
            files = list(directory.glob("*.wav"))
            print(
                f"📁 Processando {
                    len(files)} arquivos {
                    'reais' if label == 0 else 'falsos'}...")

            for audio_file in files:
                print(f"Processando: {audio_file.name}")
                result = extractor.extract_from_file(
                    str(audio_file), label=label)

                if result.status.name in ["SUCCESS", "PARTIAL_SUCCESS"]:
                    segments_features = result.data
                    print(
                        f"   🔍 Segmentos processados: {
                            len(segments_features)}")

                    for segment_feature in segments_features:
                        if len(segment_feature.combined_features) > 0:
                            features_data.append(
                                segment_feature.combined_features.tolist())
                            labels.append(label)
                else:
                    print(
                        f"⚠️ Erro ao extrair features de {
                            audio_file.name}: {
                            result.errors}")

    def _extract_features_complete(self):
        """Extrai features do áudio completo sem segmentação."""
        print("\n🔧 Extraindo features do áudio completo por categoria...")

        try:
            samples_dir = self.context.datasets_dir / "samples"
            features_dir = self.context.datasets_dir / "features"
            features_dir.mkdir(exist_ok=True)

            real_dir = samples_dir / "real"
            fake_dir = samples_dir / "fake"

            spectral_extractor = SpectralFeatureExtractor()
            temporal_extractor = TemporalFeatureExtractor()

            csv_config = CSVExportConfig(
                output_base_dir="datasets/features",
                include_metadata=False,
                decimal_places=6
            )

            csv_exporter = CSVFeatureExporter(csv_config)
            total_processed = 0

            total_processed += self._process_directory_complete(
                real_dir, "real", spectral_extractor, temporal_extractor, csv_exporter)
            total_processed += self._process_directory_complete(
                fake_dir, "fake", spectral_extractor, temporal_extractor, csv_exporter)

            print(f"\n✅ Extração completa por categoria concluída!")
            print(f"   Total de arquivos processados: {total_processed}")
            print(
                f"   Arquivos CSV organizados por categoria em: datasets/features/original/")

        except Exception as e:
            print(f"❌ Erro ao extrair features: {e}")
            self.context.logger.error(f"Erro na extração de features: {e}")

    def _process_directory_complete(
            self, directory: Path, label: str, spectral_extractor, temporal_extractor, csv_exporter):
        processed_count = 0
        if directory.exists():
            files = list(directory.glob("*.wav"))
            print(f"📁 Processando {len(files)} arquivos {label}...")

            for audio_file in files:
                try:
                    import librosa
                    audio_data, sample_rate = librosa.load(
                        audio_file, sr=16000)

                    spectral_dict = spectral_extractor.extract_features(
                        audio_data,
                        sample_rate) if label == 'fake' else spectral_extractor.extract_features(audio_data)
                    temporal_dict = temporal_extractor.extract_features(
                        audio_data,
                        sample_rate) if label == 'fake' else temporal_extractor.extract_features(audio_data)

                    spectral_features = self._normalize_dict(
                        spectral_dict, 'spectral_feature')
                    temporal_features = self._normalize_dict(
                        temporal_dict, 'temporal_feature')

                    result = csv_exporter.export_original_features_by_category(
                        spectral_features=spectral_features,
                        temporal_features=temporal_features,
                        filename=audio_file.stem,
                        label=label
                    )

                    if result.status.name == "SUCCESS":
                        processed_count += 1
                        print(f"✅ {audio_file.name}: Exportado")
                    else:
                        print(
                            f"⚠️ Erro ao exportar {
                                audio_file.name}: {
                                result.errors}")

                except Exception as e:
                    print(f"⚠️ Erro ao processar {audio_file.name}: {e}")
        return processed_count

    def _normalize_dict(self, data, prefix):
        if isinstance(data, dict):
            return data
        elif hasattr(data, '__iter__'):
            return {f'{prefix}_{i}': val for i, val in enumerate(data)}
        return {}
