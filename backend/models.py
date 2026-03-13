from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_premium = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    exchange_keys = relationship("ExchangeKey", back_populates="owner")
    bots = relationship("Bot", back_populates="owner")


class ExchangeKey(Base):
    __tablename__ = "exchange_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    exchange_name = Column(String, nullable=False) # e.g., 'binance'
    api_key_encrypted = Column(Text, nullable=False)
    secret_key_encrypted = Column(Text, nullable=False)
    is_testnet = Column(Boolean, default=True)

    owner = relationship("User", back_populates="exchange_keys")

class Bot(Base):
    __tablename__ = "bots"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String, nullable=False)
    status = Column(String, default="draft") # draft, backtesting, live, stopped
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    owner = relationship("User", back_populates="bots")
    config = relationship("BotConfig", back_populates="bot", uselist=False)

class BotConfig(Base):
    __tablename__ = "bot_configs"

    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(Integer, ForeignKey("bots.id"), unique=True)
    config_json = Column(Text, nullable=False) # JSON tabanlı kayıt
    
    bot = relationship("Bot", back_populates="config")
