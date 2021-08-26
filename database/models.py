
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Item(db.Model):
    __tablename__ = "item"
    id = db.Column(db.Integer, primary_key = True)
    items = db.Column(db.String, nullable=False)

class Model(db.Model):
    __tablename__ = 'model'
    id = db.Column(db.Integer, primary_key = True)
    customer_id = db.Column(db.Integer,  db.ForeignKey('user.id'))
    image = db.Column(db.String, nullable=False)

class Generate(db.Model):
    __tablename__ = 'generate'
    id = db.Column(db.Integer, primary_key = True)
    model_image = db.Column(db.String, db.ForeignKey('model.image'))
    item_image = db.Column(db.String, db.ForeignKey('item.items'))
    generate_image = db.Column(db.String, nullable = False)

class User(db.Model):
    """ Create user table """
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key = True)
    username = db.Column(db.String(80), unique = True)
    password = db.Column(db.String(80))
    def __init__(self, username, password):
        self.username = username
        self.password = password