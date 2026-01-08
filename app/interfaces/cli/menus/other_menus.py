from app.interfaces.cli.menus.base_menu import BaseMenu


class ResultsMenu(BaseMenu):
    """Menu para visualização de resultados."""

    def show(self):
        print("\n📈 Visualização de Resultados")
        print("⚠️  Funcionalidade em desenvolvimento.")
        input("\nPressione Enter para continuar...")


class SettingsMenu(BaseMenu):
    """Menu de configurações."""

    def show(self):
        print("\n⚙️  Configurações do Sistema")
        print("⚠️  Funcionalidade em desenvolvimento.")
        input("\nPressione Enter para continuar...")
