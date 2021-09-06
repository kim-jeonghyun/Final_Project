from flask      import Flask, request, jsonify, current_app
from flask.json import JSONEncoder
from sqlalchemy import create_engine, text

def get_user(user_id):
    user = current_app.database.execute(text("""
        SELECT 
            user_id,
            user_key,
            model_image,
            generate
        FROM User
        WHERE user_id = :user_id
    """), {
        'user_id' : user_id
    }).fetchone()

    return {
        'user_id'      : user['user_id'],
        'user_key'    : user['user_key'],
        'model_image'   : user['model_image'],
        'generate' : user['generate']
    } if user else None

# HTTP 요청을 통해 전달받은 회원가입 정보를 데이터 베이스에 저장함
def insert_user(user):
    current_app.database.execute(text("""
        INSERT INTO User (
            user_key,
            model_image,
            generate
        ) VALUES (
            :user_key,
            :model_image,
            :generate
        )
    """), user).lastrowid # 새로 사용자가 생성되면 새로 생성된 사용자의 아이디를 읽어들인다.
