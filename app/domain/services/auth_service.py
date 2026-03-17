import logging

from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError

from app.core.database import SessionLocal
from app.domain.models.user import User

logger = logging.getLogger(__name__)

class AuthService:
    @staticmethod
    def register_user(username, password, email, full_name=None, phone=None):
        """Registra um novo usuário. Retorna (Success: bool, Message: str)"""
        if not username or not password or not email:
            return False, "Usuário, senha e email são obrigatórios."

        if len(password) < 6:
            return False, "A senha deve ter pelo menos 6 caracteres."

        db = SessionLocal()
        try:
            existing_user = (
                db.query(User)
                .filter(or_(User.username == username, User.email == email))
                .first()
            )

            if existing_user:
                return False, "Usuário ou Email já cadastrados."

            new_user = User(
                username=username, email=email, full_name=full_name, phone=phone
            )
            new_user.set_password(password)
            db.add(new_user)
            db.commit()

            logger.info(f"Usuário {username} registrado com sucesso.")
            return True, "Usuário registrado com sucesso!"
        except IntegrityError:
            db.rollback()
            return False, "Usuário ou Email já cadastrados."
        except Exception as e:
            logger.error(f"Erro ao registrar usuário: {e}")
            db.rollback()
            return False, f"Erro interno ao registrar: {str(e)}"
        finally:
            db.close()

    @staticmethod
    def recover_password(email):
        """Simula envio de email de recuperação."""
        db = SessionLocal()
        try:
            user = db.query(User).filter_by(email=email).first()
            if not user:
                # Retornamos sucesso mesmo se não existir para evitar
                # enumeração de usuários
                return True, (
                    "Se o email estiver cadastrado, você receberá um "
                    "link de recuperação."
                )

            # TODO: Implementar envio real de email
            logger.info(f"Solicitação de recuperação de senha para: {email}")
            return True, (
                "Email de recuperação enviado! "
                "Verifique sua caixa de entrada."
            )
        except Exception as e:
            logger.error(f"Erro na recuperação de senha: {e}")
            return False, "Erro ao processar solicitação."
        finally:
            db.close()

    @staticmethod
    def authenticate_user(username, password):
        """Autentica um usuário. Retorna (Success: bool, Message: str)"""
        db = SessionLocal()
        try:
            user = db.query(User).filter_by(username=username).first()

            if user and user.check_password(password):
                if not user.is_active:
                    return False, "Conta desativada."
                return True, "Login realizado com sucesso!"

            return False, "Usuário ou senha inválidos."

        except Exception as e:
            logger.error(f"Erro na autenticação: {e}")
            return False, f"Erro interno na autenticação: {str(e)}"
        finally:
            db.close()
