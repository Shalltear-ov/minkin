from SQLDB import SQLDB
from sqlalchemy import Column, Integer, DateTime, String, ForeignKey, Time, Date
from sqlalchemy.orm import relationship
import datetime


class UserTable(SQLDB.Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    name = Column(String(38), nullable=False)
    passport = Column(String(10), nullable=False)
    result = Column(Integer, nullable=False)




