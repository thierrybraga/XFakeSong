import os
import getpass
from huggingface_hub import HfApi, login, create_repo
from huggingface_hub.utils import HfHubHTTPError

def deploy_interface():
    print("===============================================================================")
    print("                        DEPLOY PARA HUGGING FACE SPACES                        ")
    print("===============================================================================")
    print("\nNota: A autenticacao requer um token de acesso (Write/Escrita).")
    print("Obtenha em: https://huggingface.co/settings/tokens\n")

    token = getpass.getpass("Cole seu Token HF (input oculto): ").strip()
    if not token:
        # Fallback para input visivel caso getpass falhe em alguns terminais
        token = input("Token (visivel): ").strip()
    
    if not token:
        print("Token nao fornecido. Abortando.")
        return

    try:
        login(token=token, add_to_git_credential=True)
        print("Login realizado com sucesso!")
    except Exception as e:
        print(f"Erro no login: {e}")
        return

    api = HfApi()
    try:
        user_info = api.whoami()
        user = user_info['name']
        print(f"\nUsuario autenticado: {user}")
    except Exception as e:
        print(f"Erro ao obter informacoes do usuario: {e}")
        return

    default_name = f"{user}/xfakesong"
    repo_name = input(f"Nome do Space para deploy [{default_name}]: ").strip()
    if not repo_name:
        repo_name = default_name
    
    if "/" not in repo_name:
        repo_name = f"{user}/{repo_name}"

    print(f"\nConfiguracao:")
    print(f"- Space: {repo_name}")
    print(f"- SDK: Docker (baseado no Dockerfile existente)")
    
    confirm = input("\nConfirmar deploy? [S/n]: ").lower()
    if confirm and confirm != 's':
        print("Cancelado.")
        return

    # Criar repo se não existir
    print("\n1. Verificando repositorio...")
    try:
        create_repo(repo_id=repo_name, repo_type="space", space_sdk="docker", exist_ok=True)
        print("   Repositorio pronto.")
    except Exception as e:
        print(f"   Erro ao criar/verificar repositorio: {e}")
        return

    # Upload
    print("\n2. Enviando arquivos... (Isso pode levar alguns minutos)")
    try:
        api.upload_folder(
            folder_path=".",
            repo_id=repo_name,
            repo_type="space",
            ignore_patterns=[
                ".git*", 
                ".venv*", 
                "__pycache__*", 
                "logs*", 
                "data/real/*", 
                "data/fake/*", 
                "*.pyc", 
                "system.log", 
                ".env", 
                "tests*",
                "start.bat"
            ]
        )
        print("\n===============================================================================")
        print("                           DEPLOY CONCLUIDO!                                   ")
        print("===============================================================================")
        print(f"Acesse seu Space em: https://huggingface.co/spaces/{repo_name}")
        print("Nota: O Space pode levar alguns minutos para construir o container.")
        print("      Verifique a aba 'Logs' no Hugging Face se houver erros.")
    except Exception as e:
        print(f"Erro no upload: {e}")

if __name__ == "__main__":
    deploy_interface()
