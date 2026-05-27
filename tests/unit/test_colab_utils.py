import unittest
from unittest.mock import patch, MagicMock
import sys
from app.utils.colab import is_colab, mount_drive


class TestColabUtils(unittest.TestCase):
    def test_is_colab_local(self):
        """
        Testa se is_colab retorna False localmente
        (assumindo que não estamos no Colab)
        """
        # Garante que 'google.colab' não está em sys.modules
        with patch.dict(sys.modules):
            if 'google.colab' in sys.modules:
                del sys.modules['google.colab']
            self.assertFalse(is_colab())

    def test_is_colab_simulated(self):
        """Testa se is_colab retorna True quando simulado"""
        with patch.dict(sys.modules, {'google.colab': MagicMock()}):
            self.assertTrue(is_colab())

    @patch('app.utils.colab.logger')
    def test_mount_drive_local(self, mock_logger):
        """Testa mount_drive localmente (não deve tentar montar)"""
        with patch.dict(sys.modules):
            if 'google.colab' in sys.modules:
                del sys.modules['google.colab']

            mount_drive()
            mock_logger.warning.assert_called_with(
                "Não está rodando no Colab. Ignorando montagem do Drive."
            )

    @patch('app.utils.colab.logger')
    def test_mount_drive_colab(self, mock_logger):
        """Testa mount_drive no ambiente Colab simulado"""
        mock_drive = MagicMock()
        mock_google_colab = MagicMock()
        mock_google_colab.drive = mock_drive

        modules = {
            'google.colab': mock_google_colab,
            'google.colab.drive': mock_drive
        }

        with patch.dict(sys.modules, modules):
            with patch('os.path.exists', return_value=False):
                # Recarregar módulo para pegar o novo sys.modules se necessário
                # mas aqui is_colab é chamado a cada vez.
                # O problema pode ser o import dentro da função.

                # Vamos garantir que o import dentro da função pegue nosso mock
                mount_drive()

                # Verificar se drive.mount foi chamado
                mock_drive.mount.assert_called_with('/content/drive')
                mock_logger.info.assert_called()


if __name__ == '__main__':
    unittest.main()
