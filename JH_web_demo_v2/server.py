from flask import Flask, render_template, request, jsonify
from sqlalchemy import create_engine, text
from werkzeug.utils import secure_filename
from sign_up import get_items, get_user, insert_generate, insert_user, get_item_num, check_category, check_model, get_mask_url
from connection import s3_connection, get_s3_resource
from s3_config import BUCKET_NAME
from make_user_key import get_user_key


app = Flask(__name__)

app.config.from_pyfile("config.py")
# 데이터 베이스와 연동해준다.
database = create_engine(app.config['DB_URL'], encoding='utf-8', max_overflow=0)
app.database = database

# s3 storage 관련 변수
s3 = s3_connection()
s3_resource = get_s3_resource()
location = s3.get_bucket_location(Bucket=BUCKET_NAME)['LocationConstraint']
path = f'https://{BUCKET_NAME}.s3.{location}.amazonaws.com/'


# S3 특정 폴더 내에 있는 파일 리스트를 얻는 함수
'''def get_file_list_s3(bucket, gender, category):
    my_bucket = s3_resource.Bucket(bucket)
    prefix = gender + "/image/" + category
    file_objs = my_bucket.objects.filter(Prefix=prefix).all()
    file_names = [file_obj.key for file_obj in file_objs]
    return file_names'''

# 대문 페이지
@app.route('/')
def sss():
    return render_template('index.html')

# 테스트용 페이지
@app.route('/test')
def test():
    items= get_items("top","w")[0][2]
    print(items)
    return ""

# landing_page에 이미지 보여주기
@app.route('/select', methods=['GET'])
def get_photos():
    default_model = get_user(2)['model_url']
    default_items= get_items("top","w")
    image_urls = []
    image_urls.append(default_model)
    for i in range(len(default_items)):
        image_urls.append(default_items[i][2])
    return render_template('select.html', image_file=image_urls)


# female / male 버튼을 누르면 모델 이미지 바꿔서 보여주기
@app.route('/click_gender', methods=['GET'])
def get_model_photo():
    parameter_dict = request.args.to_dict()
    gender = parameter_dict['gender']
    if gender == "w":
        model_image = get_user(2)['model_url']
    else:
        model_image = get_user(1)['model_url']
    # html의 조각만 return해줌
    return render_template('model_image.html', image_name=model_image)


# top / bottom 버튼을 누르면 젠더/ 카테고리에 맞는 아이템 이미지 바꿔주기
@app.route('/click_category', methods=['GET'])
def get_items_photo():
    parameter_dict = request.args.to_dict()
    category = parameter_dict['category']
    gender = parameter_dict['gender']
    items_list = get_items(category,gender)
    img_urls = []
    for i in range(len(items_list)):
        img_urls.append(items_list[i][2])
    # html의 조각만 return해줌
    return render_template('item_images.html', img_urls=img_urls)


# 아이템을 눌렀을 때 모델 자리를 생성된 이미지로 바꿔주기
@app.route('/click_item', methods=['GET'])
def get_generated_photo():
    parameter_dict = request.args.to_dict()
    gender = parameter_dict['gender']
    item_id = parameter_dict['item_id']
    print(gender, item_id)
    generated_image = path + get_file_list_s3(bucket=BUCKET_NAME, gender=gender, category="model")[int(item_id[-1])]
    # html의 조각만 return해줌
    return render_template('model_image.html', image_name=generated_image)


@app.route('/db')
def index():
    # new_user = request.json
    new_user = {
        'user_key': 'asdfszsdf2323asdxfxdfd',
        'model_image': 'c',
        'generate': False
    }
    new_user_id = insert_user(new_user)
    new_user = get_user(new_user_id)
    return jsonify(new_user)


# 업로드 HTML 렌더링
# @app.route('/upload')
# def render_file():
#    return render_template('upload.html')


# 파일 업로드 처리
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        print(f)

        ###########################
        ##### 이미지 전처리 #######
        ###########################

        user_key = get_user_key(32)
        print(user_key)
        # 저장할 경로 + 파일명
        top_id, bottom_id = 0, 0
        s3_path = 'model-image/' + user_key + str(top_id) + str(bottom_id)
        s3.put_object(
            Bucket=BUCKET_NAME,
            Body=f,
            Key=s3_path,
            ContentType=f.content_type
        )
        location = s3.get_bucket_location(Bucket=BUCKET_NAME)['LocationConstraint']
        image_url = f'https://{BUCKET_NAME}.s3.{location}.amazonaws.com/{s3_path}'

        new_user = {
            'user_key': user_key,
            'model_url': image_url,
            'generate': False
        }

        new_user_id = insert_user(new_user)

    return render_template('model_image.html', image_url=image_url)
    # return 'uploads 디렉토리 -> 파일 업로드 성공!' + image_url


# generate HTML 렌더링
# 상의 or 하의 item 선택 시 작동
# @app.route('/generate')
# def render_file():
# 1. 해당
# 이미지를 클릭할 시 해당 item 의 id 받아오기


####################
######모델추론######
####################


# return render_template('generate.html')


if __name__ == '__main__':
    # 서버 실행
    app.run(debug=True, port=8000)
