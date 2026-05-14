"""
Modèle AccountRequest — demandes de création de compte en attente d'approbation admin
"""

import enum

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy import Enum as SQLEnum

from src.core.utils import _utcnow
from src.db.database import Base


class AccountRequestStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class AccountRequest(Base):
    __tablename__ = "account_requests"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), nullable=False)
    email = Column(String(100), nullable=False, index=True)
    message = Column(String(500), nullable=True)
    role_requested = Column(String(20), nullable=False, default="user")
    status = Column(
        SQLEnum(AccountRequestStatus, values_callable=lambda obj: [e.value for e in obj]),
        default=AccountRequestStatus.PENDING,
        nullable=False,
        index=True,
    )
    rejection_reason = Column(String(300), nullable=True)
    requested_at = Column(DateTime, default=_utcnow, nullable=False)
    reviewed_at = Column(DateTime, nullable=True)
    reviewer_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    def __repr__(self) -> str:
        return f"<AccountRequest(id={self.id}, username='{self.username}', status='{self.status}')>"
