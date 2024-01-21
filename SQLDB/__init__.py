from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm.session import Session


class SQLDB:
    Base = declarative_base()

    def __init__(self, name):
        self.engine = create_engine(f"sqlite:///{name}")
        self.session = Session(bind=self.engine)
        self.session.create_all = lambda: self.Base.metadata.create_all(self.engine)
