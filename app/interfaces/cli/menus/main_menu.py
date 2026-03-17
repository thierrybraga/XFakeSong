from app.interfaces.cli.menus.base_menu import BaseMenu
from app.interfaces.cli.menus.dataset_menu import DatasetMenu
from app.interfaces.cli.menus.inference_menu import InferenceMenu
from app.interfaces.cli.menus.other_menus import ResultsMenu, SettingsMenu
from app.interfaces.cli.menus.training_menu import TrainingMenu


class MainMenu(BaseMenu):
    """Menu principal do sistema."""

    def __init__(self, context):
        super().__init__(context)
        self.dataset_menu = DatasetMenu(context)
        self.training_menu = TrainingMenu(context)
        self.inference_menu = InferenceMenu(context)
        self.results_menu = ResultsMenu(context)
        self.settings_menu = SettingsMenu(context)

    def show(self):
        print("🚀 Iniciando Sistema de Detecção de Deepfake...")

        while True:
            try:
                self._print_header("SISTEMA DE DETECÇÃO DE DEEPFAKE DE ÁUDIO")
                print("1. 📊 Gerenciar Dataset")
                print("2. 🤖 Treinar Modelo")
                print("3. 🔍 Inferência de Áudio")
                print("4. 📈 Visualizar Resultados")
                print("5. ⚙️  Configurações")
                print("6. ❌ Sair")

                choice = input("\nEscolha uma opção: ").strip()

                if choice == "1":
                    self.dataset_menu.show()
                elif choice == "2":
                    self.training_menu.show()
                elif choice == "3":
                    self.inference_menu.show()
                elif choice == "4":
                    self.results_menu.show()
                elif choice == "5":
                    self.settings_menu.show()
                elif choice == "6":
                    print("\n👋 Encerrando sistema...")
                    break
                else:
                    print("❌ Opção inválida! Tente novamente.")

            except KeyboardInterrupt:
                print("\n\n👋 Sistema interrompido pelo usuário.")
                break
            except Exception as e:
                print(f"\n❌ Erro inesperado: {e}")
                self.context.logger.error(f"Erro no menu principal: {e}")
                input("Pressione Enter para continuar...")
