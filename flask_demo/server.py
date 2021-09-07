from flask import Flask, render_template, request, jsonify
from sqlalchemy import create_engine, text
from werkzeug.utils import secure_filename
from sign_up import insert_user, get_user, insert_user_key
from connection import s3_connection
from s3_config import BUCKET_NAME
from make_user_key import get_user_key
app = Flask(__name__)

app.config.from_pyfile("config.py")
# 데이터 베이스와 연동해준다.
database = create_engine(app.config['DB_URL'], encoding = 'utf-8', max_overflow = 0)
app.database = database

s3 = s3_connection()

@app.route('/db')
def index():
   #new_user = request.json
   new_user = {
        'user_key'    : 'asdfszsdf2323asdxfxdfd',
        'model_image'   : 'c',
        'generate' : False
    }
   new_user_id = insert_user(new_user)
   new_user = get_user(new_user_id)
   return jsonify(new_user)

@app.route('/')
def sdsd():
   return 'Hello~~'

# 업로드 HTML 렌더링
@app.route('/upload')
def render_file():
   return render_template('upload.html')


# 파일 업로드 처리
@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']


      ###########################
      ##### 이미지 전처리 #######
      ###########################

      user_key = get_user_key(32)
      insert_user_key(user_key)

      #저장할 경로 + 파일명
      top_id, bottom_id = 0, 0
      s3_path = 'model-image/' + user_key + str(top_id) + str(bottom_id)
      s3.put_object(
         Bucket = BUCKET_NAME,
         Body = f,
         Key = s3_path,
         ContentType = f.content_type
      )
      location = s3.get_bucket_location(Bucket=BUCKET_NAME)['LocationConstraint']
      image_url = f'https://{BUCKET_NAME}.s3.{location}.amazonaws.com/{s3_path}'

      new_user ={ 
         'user_key': user_key,
         'model_image': image_url,
         'generate': False
         }
   return render_template('view.html', image_url = image_url)
   #return 'uploads 디렉토리 -> 파일 업로드 성공!' + image_url




# generate HTML 렌더링
# 상의 or 하의 item 선택 시 작동
@app.route('/generate')
def render_file():
   # 1. 해당 
   # 이미지를 클릭할 시 해당 item 의 id 받아오기




   ####################
   ######모델추론######
   ####################


   return render_template('generate.html')




if __name__ == '__main__':
    #서버 실행
   app.run(debug = True, port=8000)
