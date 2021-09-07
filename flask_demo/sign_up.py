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
    return current_app.database.execute(text("""
        INSERT INTO User (
            user_key,
            model_image,
            generate
        ) VALUES (
            :user_key,
            :model_image,
            :generate
        )
    """), user).lastrowid#.inserted_primary_key # 새로 사용자가 생성되면 새로 생성된 사용자의 아이디를 읽어들인다.

def insert_user_key(user_key):
    current_app.database.execute(text("""
        INSERT INTO User_key(
            user_key
        ) VALUES (
            :user_key
        )
        """), {
            'user_key': user_key
            })


def insert_item(item):
    current_app.database.execute(text("""
        INSERT INTO Item (
            item,
            category,
            sex
        ) VALUES (
            :item,
            :category,
            :sex
        )
        """), item)

def insert_top_item(top_item):
    current_app.database.execute(text("""
        INSERT INTO Item (
            top_item
        ) VALUES (
            :top_item
        )
        """),   {
        'top_item' : top_item
    })

def insert_bottom_item(bottom_item):
    current_app.database.execute(text("""
        INSERT INTO Bottom (
            bottom_item
        ) VALUES (
            :bottom_item
        )
        """),  {
        'bottom_item' : bottom_item
    })

def insert_generate(generate):
    return current_app.database.execute(text("""
        INSERT INTO Generate(
            model_image,
            top_item,
            bottom_item,
            generate_image,
            ) VALUES (
                : model_image,
                : top_item,
                : bottom_item,
                : generate_image
            )
        """), generate)