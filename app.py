from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
import tensorflow as tf
import logging

app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///rating.db'
db = SQLAlchemy(app)
migrate = Migrate(app, db)

class History(db.Model):
    __tablename__ = 'history'

    id = db.Column(db.Integer, primary_key=True)
    rating = db.Column(db.Integer, nullable=False)
    review = db.Column(db.String(1000), nullable=True)

with app.app_context():
    db.create_all()

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = TFBertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Define sentiment labels
sentiment_labels = ['1', '2', '3', '4', '5']

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="tf", max_length=512, truncation=True, padding=True)
    inputs = {name: tensor.numpy() for name, tensor in inputs.items()}
    logits = model.predict(inputs)[0]
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    return sentiment_labels[predicted_class]

@app.route('/result', methods=['POST'])
def analyze():
    logging.info("Request received")
    data = request.get_json()
    text = data['text']
    sentiment = predict_sentiment(text)
    history = History(rating=sentiment, review=text)
    db.session.add(history)
    db.session.commit()
    return jsonify(sentiment)

@app.route('/history', methods=['GET'])
def get_history():
    history = History.query.all()
    return jsonify([{'id': item.id, 'text': item.review, 'rating': item.rating} for item in history])

@app.route('/delete/<int:id>', methods=['DELETE'])
def delete_history(id):
    history = History.query.get(id)
    if history is None:
        return jsonify({'error': 'Not found'}), 404
    db.session.delete(history)
    db.session.commit()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)
