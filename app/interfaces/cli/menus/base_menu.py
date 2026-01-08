from abc import ABC, abstractmethod
from app.interfaces.cli.context import AppContext


class BaseMenu(ABC):
    """Classe base para menus da CLI."""

    def __init__(self, context: AppContext):
        self.context = context

    @abstractmethod
    def show(self):
        """Exibe o menu e processa a entrada do usuário."""
        pass

    def _print_header(self, title: str):
        """Imprime o cabeçalho do menu."""
        print("\n" + "=" * 60)
        print(f"    {title}")
        print("=" * 60)
