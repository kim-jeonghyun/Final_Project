
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Top_item(db.Model):
    __tablename__ = "top_items"
    top_id = db.Column(db.Integer, primary_key = True)
    item = db.Column(db.String, nullable=False)
    sex = db.Column(db.String, nullable=False)

class Bottom_item(db.Model):
    __tablename__ = "bottom_items"
    bottom_id = db.Column(db.Integer, primary_key = True)
    item = db.Column(db.String, nullable=False)
    sex = db.Column(db.String, nullable=False)

class User(db.Model):
    __tablename__ = "user"
    user_id = db.Column(db.Integer, primary_key = True)
    user_key = db.Column(db.Integer, nullable=False)
    model_image = db.Column(db.String, default='~~~.jpg')

class Model(db.Model):
    __tablename__ = 'model'
    model_id = db.Column(db.Integer, primary_key = True)
    top_items = db.Column(db.String, db.ForeignKey('top_items.item'), nullable=True)
    bottom_items = db.Column(db.String, db.ForeignKey('bottom_items.item'), nullable=True)
    user_key = db.Column(db.Integer, db.ForeignKey('user.user_key'), nullable=False)
    model_image = db.Column(db.String, db.ForeignKey('user.model_image'), nullable=False)
    top_generate_image = db.Column(db.String, db.ForeignKey('top_generate.top_generate_image'), nullable=True)
    bottom_generate_image = db.Column(db.String, db.ForeignKey('bottom_generate.bottom_generate_image'), nullable=True)

class Top_generate(db.Model):
    __tablename__ = 'top_generate'
    top_generate__id = db.Column(db.Integer, primary_key = True)
    user_key = db.Column(db.Integer, db.ForeignKey('user.user_key'), nullable=False)
    model_image = db.Column(db.String, db.ForeignKey('user.model_image'), nullable=True)
    item = db.Column(db.String, db.ForeignKey('top_items.item'), nullable=False)
    top_generate_image = db.Column(db.String, nullable = False)
    bottom_generate_image = db.Column(db.String, db.ForeignKey('bottom_generate.bottom_generate_image'), nullable=True)

class Bottom_generate(db.Model):
    __tablename__ = 'bottom_generate'
    bottom_generate_id = db.Column(db.Integer, primary_key = True)
    user_key = db.Column(db.Integer, db.ForeignKey('user.user_key'), nullable=False)
    model_image = db.Column(db.String, db.ForeignKey('user.model_image'), nullable=True)
    item = db.Column(db.String, db.ForeignKey('bottom_items.item'), nullable=False)
    top_generate_image = db.Column(db.String, db.ForeignKey('top_generate.top_generate_image'), nullable=True)
    bottom_generate_image = db.Column(db.String, nullable=False)