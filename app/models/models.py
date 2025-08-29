from app import db # Import the db instance from the app factory

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

# You can add more models here or create an 'app/models/' directory
# for larger applications and import them into app/__init__.py
